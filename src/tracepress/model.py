"""The model layer: DSPy, running on its bundled lm15 engine, plus what operators need on top.

DSPy supplies signatures, prompt formatting (its JSON adapter, which asks the provider
for native structured output), parsing, retries, a response cache, and per-call cost.
This module adds concurrent batches, a progress bar, token and cost totals, and a
budget the agent can't exceed.

    lm = Model("openrouter:openai/gpt-6-luna", temperature=0, max_tokens=600)
    lm.predict(dspy.Signature("review -> gist"), [{"review": "..."}])

DSPy is always run with ``engine="lm15"``, so litellm is never imported.
"""

from __future__ import annotations

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import dspy
from dspy.lm15 import Config, Reasoning, Request, Response


def cache_dir() -> Path:
    """Where response cache entries are stored.

    ``$TRACEPRESS_HOME/cache`` when that variable is set, otherwise ``~/.tracepress/cache``.
    """
    override = os.environ.get("TRACEPRESS_HOME", "").strip()
    home = Path(override).expanduser() if override else Path.home() / ".tracepress"
    return home / "cache"


# ---------------------------------------------------------------------------- usage


@dataclass
class Usage:
    """Token and cost totals. Adds with ``+``."""

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.calls + other.calls,
            self.cached_calls + other.cached_calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )

    def as_dict(self) -> dict[str, Any]:
        """A plain dict, for manifests and printing."""
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def total_usage(usages: Sequence[Usage]) -> Usage:
    out = Usage()
    for u in usages:
        out = out + u
    return out


def _usage_of(history: list[dict[str, Any]], price: tuple[float, float] | None = None) -> Usage:
    """Usage from the history entries of one call (an LM copy's own history).

    DSPy records a cache hit with empty usage but keeps the original call's cost, so an
    entry without usage counts as cached and costs nothing.
    """
    total = Usage()
    for entry in history:
        u = entry.get("usage") or {}
        if not u:
            total = total + Usage(cached_calls=1)
            continue
        n_in, n_out = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        cost = entry.get("cost")
        if cost is None and price:  # a gateway model DSPy can't price: use the caller's $/M tokens
            cost = (n_in * price[0] + n_out * price[1]) / 1_000_000
        total = total + Usage(calls=1, input_tokens=n_in, output_tokens=n_out, cost_usd=float(cost or 0.0))
    return total


# --------------------------------------------------------------------------- budget


class BudgetExceeded(RuntimeError):
    """Raised before a batch starts when it would go over the active budget."""


class Budget:
    """A cap on model calls, and on dollars spent.

    Checked before every batch starts, so a refused batch costs nothing. The dollar cap
    compares what has been spent so far (DSPy reports each call's cost) plus, when the
    model has a ``price``, an estimate for the batch about to start.
    """

    def __init__(self, max_calls: int | None = None, max_usd: float | None = None):
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.calls = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def reserve(self, n_calls: int, est_cost: float | None) -> None:
        """Claim ``n_calls`` calls before a batch starts, or raise BudgetExceeded."""
        with self._lock:
            if self.max_calls is not None and self.calls + n_calls > self.max_calls:
                raise BudgetExceeded(
                    f"this step needs up to {n_calls} model calls but only "
                    f"{self.max_calls - self.calls} of {self.max_calls} remain"
                )
            if self.max_usd is not None:
                if self.cost_usd >= self.max_usd:
                    raise BudgetExceeded(f"the ${self.max_usd:.2f} budget is spent (${self.cost_usd:.2f})")
                if est_cost is not None and self.cost_usd + est_cost > self.max_usd:
                    raise BudgetExceeded(
                        f"this step is estimated at ${est_cost:.2f} but only "
                        f"${self.max_usd - self.cost_usd:.2f} of ${self.max_usd:.2f} remain"
                    )
            self.calls += n_calls

    def spend(self, usage: Usage) -> None:
        """Record what a finished call cost."""
        with self._lock:
            self.cost_usd += usage.cost_usd

    def remaining(self) -> str:
        """What's left, in words (shown to the agent each turn)."""
        parts = []
        if self.max_calls is not None:
            parts.append(f"{self.max_calls - self.calls} model calls")
        if self.max_usd is not None:
            parts.append(f"${self.max_usd - self.cost_usd:.2f}")
        return " and ".join(parts) or "unlimited"


_budget: ContextVar[Budget | None] = ContextVar("tracepress_budget", default=None)


@contextmanager
def budget(max_calls: int | None = None, max_usd: float | None = None) -> Iterator[Budget]:
    """Cap the model calls made inside a ``with`` block."""
    b = Budget(max_calls, max_usd)
    token = _budget.set(b)
    try:
        yield b
    finally:
        _budget.reset(token)


# --------------------------------------------------------------------------- model


@dataclass
class Prediction:
    """One row's answer: the output fields, and what the call cost."""

    outputs: dict[str, Any]
    usage: Usage


class Model:
    """A ``dspy.LM`` plus concurrency, accounting, and budgets.

    Args:
        model: ``"provider:model"``. The model id may contain slashes, e.g.
            ``"openrouter:openai/gpt-6-luna"`` or ``"anthropic:claude-opus-5-5"``.
        engine: a custom lm15 engine (anything with ``complete(Request) -> Response``),
            e.g. a test double. Defaults to DSPy's native lm15 engine.
        temperature, max_tokens: generation settings.
        reasoning: an effort word (``"off"``, ``"low"``, ... ``"max"``).
        max_workers: concurrent requests per batch.
        num_retries: DSPy's retries on rate limits and transient errors.
        cache: DSPy's response cache (on disk, under ``~/.tracepress/cache``).
        price: ``(input, output)`` dollars per million tokens. Estimates a batch before it runs,
            and, for Portkey models only, prices calls DSPy reports no cost for. Cost DSPy does report is
            kept, and a cache hit costs nothing.
        progress: show a tqdm bar for batches.
        metadata, portkey_config, virtual_key: Portkey only (``"portkey:@slug/model"``). Sent as
            ``x-portkey-metadata``, ``x-portkey-config`` and ``x-portkey-virtual-key``; unset ones
            fall back to ``PORTKEY_METADATA`` and ``PORTKEY_CONFIG``. A virtual
            key and ``@slug/model`` routing can conflict: use one or the other.
        lm_kwargs: anything else is passed to ``dspy.LM``.
    """

    def __init__(
        self,
        model: str,
        *,
        engine: Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: str | None = None,
        max_workers: int = 8,
        num_retries: int = 3,
        cache: bool = True,
        price: tuple[float, float] | None = None,
        progress: bool = True,
        metadata: Mapping[str, Any] | None = None,
        portkey_config: str | Mapping[str, Any] | None = None,
        virtual_key: str | None = None,
        **lm_kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.max_workers = max_workers
        self.price = price
        self.progress = progress
        kwargs: dict[str, Any] = dict(lm_kwargs)
        provider, model_id = _split(model)
        # DSPy can't price gateway models, so only they get `price` as a cost; elsewhere a missing
        # cost can mean "unknown" on purpose (subscription plans, tool fees) and stays 0.
        self._cost_price = price if provider == "portkey" else None
        if reasoning is not None:
            # lm15 reads each server's own spelling: OpenRouter takes a `reasoning` object.
            if provider == "openrouter":
                kwargs["reasoning"] = {"effort": reasoning}
            else:
                kwargs["reasoning_effort"] = reasoning
        if provider == "portkey":
            from .providers import portkey_model

            if not model_id:
                raise ValueError(f"{model!r} has no model id; use 'portkey:@provider-slug/model'")

            dspy_model = portkey_model(model_id, metadata=metadata, config=portkey_config, virtual_key=virtual_key)
        elif metadata is not None or portkey_config is not None or virtual_key is not None:
            raise ValueError("metadata, portkey_config and virtual_key only apply to portkey models")
        else:
            dspy_model = f"{provider}/{model_id}" if model.startswith(f"{provider}:") else model
        self.lm = dspy.LM(
            dspy_model,
            engine=engine if engine is not None else "lm15",
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
            cache=cache,
            **kwargs,
        )
        self.adapter = dspy.JSONAdapter()
        self.usage = Usage()
        self._usage_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Model({self.model!r})"

    # ---------------------------------------------------------------- accounting

    def _account(self, usage: Usage) -> None:
        with self._usage_lock:
            self.usage = self.usage + usage
        b = _budget.get()
        if b is not None:
            b.spend(usage)

    def _reserve(self, n_calls: int, prompt_chars: int) -> None:
        b = _budget.get()
        if b is None:
            return
        if b.max_usd is not None and self._cost_price is None and _split(self.model)[0] == "portkey":
            warnings.warn(
                f"{self.model} has no price=(input, output), so its cost is recorded as $0 and "
                "budget(max_usd=...) cannot stop it",
                stacklevel=3,
            )
        est = None
        if self.price:
            est = (prompt_chars / 4 * self.price[0] + n_calls * (self.max_tokens or 1024) * self.price[1]) / 1_000_000
        b.reserve(n_calls, est)

    # ------------------------------------------------------------------ predict

    def _predict_one(self, signature: type[dspy.Signature], inputs: Mapping[str, Any], examples: list | None) -> Prediction:
        lm = self.lm.copy()  # its own history, so this call's usage and cost can be read back
        predictor = dspy.Predict(signature)
        if examples:
            predictor.demos = examples
        with dspy.context(lm=lm, adapter=self.adapter):
            result = predictor(**inputs)
        usage = _usage_of(lm.history, self._cost_price)
        self._account(usage)
        return Prediction({name: result[name] for name in signature.output_fields}, usage)

    def predict(
        self,
        signature: type[dspy.Signature],
        inputs: Sequence[Mapping[str, Any]],
        *,
        examples: list[dspy.Example] | None = None,
        desc: str | None = None,
        progress: bool | None = None,
    ) -> list[Prediction]:
        """Run ``signature`` once per input mapping, concurrently, in order."""
        inputs = list(inputs)
        if not inputs:
            return []
        self._reserve(len(inputs), sum(len(str(v)) for i in inputs for v in i.values()))

        show = self.progress if progress is None else progress
        bar = _progress_bar(len(inputs), desc or self.model) if show and len(inputs) > 1 else None

        def run(item: Mapping[str, Any]) -> Prediction:
            p = self._predict_one(signature, item, examples)
            if bar is not None:
                bar.update(1)
            return p

        try:
            if self.max_workers <= 1 or len(inputs) == 1:
                return [run(i) for i in inputs]
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                return list(pool.map(run, inputs))
        finally:
            if bar is not None:
                bar.close()

    # --------------------------------------------------------- raw lm15 requests

    def config(self) -> Config:
        """This model's settings as an lm15 ``Config``, for explicit requests (the agent loop)."""
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.reasoning is not None:
            kwargs["reasoning"] = Reasoning(effort=self.reasoning)
        return Config(**kwargs)

    def send(self, request: Request) -> tuple[Response, Usage]:
        """Send one explicit lm15 request through DSPy (used by the agent's tool loop)."""
        lm = self.lm.copy()
        response = lm(request)
        usage = _usage_of(lm.history, self._cost_price)
        self._account(usage)
        return response, usage


def _split(model: str) -> tuple[str, str]:
    """``(provider, model_id)``. ``provider:id`` splits at the first colon, unless a ``/`` comes
    first, so ids that contain colons (``portkey/@bedrock/claude-v2:0``) keep them."""
    colon, slash = model.find(":"), model.find("/")
    if colon != -1 and (slash == -1 or colon < slash):
        return model[:colon], model[colon + 1 :]
    if slash != -1:
        return model[:slash], model[slash + 1 :]
    return model, ""


def _progress_bar(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, leave=False)


__all__ = ["Model", "Prediction", "Usage", "Budget", "BudgetExceeded", "budget", "total_usage"]
