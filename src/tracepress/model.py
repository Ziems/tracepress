"""The model layer: DSPy, running on its bundled lm15 engine, plus what operators need on top.

DSPy supplies signatures, prompt formatting (its XML adapter), parsing, retries, a
response cache, and per-call cost.
This module adds concurrent batches, a progress bar, token and cost totals, and a
budget that refuses batches before they overrun it.

    lm = Model("openrouter:openai/gpt-6-luna", temperature=0, max_tokens=600)
    lm.predict(dspy.Signature("review -> gist"), [{"review": "..."}])

DSPy is always run with ``engine="lm15"``, so litellm is never imported.
"""

from __future__ import annotations

import os
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import dspy
from dspy.utils.exceptions import AdapterParseError


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
    entry without usage counts as cached and costs nothing. ``price`` is ``(input, output)``
    dollars per million tokens, used when a call reports tokens but no cost.
    """
    total = Usage()
    for entry in history:
        u = entry.get("usage") or {}
        if not u:
            total = total + Usage(cached_calls=1)
            continue
        n_in, n_out = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        cost = entry.get("cost")
        if cost is None and price is not None:
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

_BARE_AMPERSAND = re.compile(r"&(?![A-Za-z]+;|#\d+;|#x[0-9A-Fa-f]+;)")


class _XMLAdapter(dspy.XMLAdapter):
    """DSPy's XML adapter, tolerant of a bare ``&`` or ``<`` inside a field.

    Answers about shell traces quote ``2>&1``, ``&&``, and ``n < 4``, which are not
    well-formed XML, so DSPy's strict parse fails and falls back to JSON. When the
    strict parse fails, those characters are escaped and the answer is parsed again.
    Only the output field tags (and ``item``, for lists) are kept as markup.
    """

    def parse(self, signature: type[dspy.Signature], completion: str) -> dict[str, Any]:
        try:
            return super().parse(signature, completion)
        except AdapterParseError:
            repaired = _escape_xml_text(completion, [*signature.output_fields, "item"])
            if repaired == completion:
                raise
            return super().parse(signature, repaired)


def _escape_xml_text(text: str, tags: Sequence[str]) -> str:
    """Escape ``&`` that is not an entity, and ``<`` that does not open or close one of ``tags``."""
    known = "|".join(re.escape(tag) for tag in tags)
    text = _BARE_AMPERSAND.sub("&amp;", text)
    return re.sub(rf"<(?!/?(?:{known})>)", "&lt;", text)


@dataclass
class Prediction:
    """One row's answer: the output fields, and what the call cost."""

    outputs: dict[str, Any]
    usage: Usage


class Model:
    """A ``dspy.LM`` plus concurrency, accounting, and budgets.

    Args:
        model: a LiteLLM id, e.g. ``"openai/gpt-5-nano"`` or
            ``"anthropic/claude-opus-5-5"``. A gateway is that id with a prefix:
            ``"openrouter:openai/gpt-6-luna"``, ``"portkey:openai/gpt-5-nano"``.
            ``portkey:`` reads ``PORTKEY_API_KEY``. ``PORTKEY_BASE_URL`` overrides
            the gateway. A gateway is priced from its own LiteLLM entry when that
            entry has a rate, and otherwise from the id after the colon.
        engine: a custom lm15 engine (anything with ``complete(Request) -> Response``),
            e.g. a test double. Defaults to DSPy's native lm15 engine.
        temperature, max_tokens: generation settings.
        reasoning: an effort word (``"off"``, ``"low"``, ... ``"max"``).
        max_workers: concurrent requests per batch.
        num_retries: DSPy's retries on rate limits and transient errors.
        parse_retries: fresh calls for a row whose response cannot be parsed, even
            after DSPy's JSON fallback. Each retry has its own cache entry, so a
            rerun replays the retry instead of the bad response. Retries are not
            counted against a ``budget``'s ``max_calls``.
        cache: DSPy's response cache (on disk, under ``~/.tracepress/cache``).
        price: ``(input, output)`` dollars per million tokens, used to estimate a
            batch before it runs. Actual cost comes from DSPy. A gateway DSPy
            cannot price uses this rate for the recorded cost; when omitted, the
            LiteLLM rate is used, falling back from the gateway entry to the id
            after the colon. ``portkey:`` ignores a leading ``@``. A missing rate
            warns and the cost stays ``$0``.
        progress: show a tqdm bar for batches.
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
        parse_retries: int = 2,
        cache: bool = True,
        price: tuple[float, float] | None = None,
        progress: bool = True,
        **lm_kwargs: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.max_workers = max_workers
        self.price = price
        self.progress = progress
        self.parse_retries = parse_retries
        kwargs: dict[str, Any] = dict(lm_kwargs)
        if model.startswith("portkey:"):
            dspy_model = _portkey_target(model, kwargs)
        else:
            dspy_model = _for_dspy(model)
        if self.price is None and _gateway_upstream(model) is not None:
            self.price = _gateway_price(model)
            if self.price is None:
                warnings.warn(
                    f"{model} has no LiteLLM price; its cost is recorded as $0. "
                    "Pass price=(input, output) in dollars per million tokens.",
                    stacklevel=2,
                )
        if reasoning is not None:
            # lm15 reads each server's own spelling: OpenRouter takes a `reasoning` object.
            if _provider(model) == "openrouter":
                kwargs["reasoning"] = {"effort": reasoning}
            else:
                kwargs["reasoning_effort"] = reasoning
        self.lm = dspy.LM(
            dspy_model,
            engine=engine if engine is not None else "lm15",
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
            cache=cache,
            **kwargs,
        )
        self.adapter = _XMLAdapter()
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
        est = None
        if self.price:
            est = (prompt_chars / 4 * self.price[0] + n_calls * (self.max_tokens or 1024) * self.price[1]) / 1_000_000
        b.reserve(n_calls, est)

    # ------------------------------------------------------------------ predict

    def _predict_one(self, signature: type[dspy.Signature], inputs: Mapping[str, Any], examples: list | None) -> Prediction:
        predictor = dspy.Predict(signature)
        if examples:
            predictor.demos = examples
        usage = Usage()
        for attempt in range(self.parse_retries + 1):
            lm = self._attempt_lm(attempt)
            try:
                with dspy.context(lm=lm, adapter=self.adapter):
                    result = predictor(**inputs)
                break
            except AdapterParseError:
                if attempt == self.parse_retries:
                    raise
            finally:
                spent = _usage_of(lm.history, self._recorded_price())
                self._account(spent)
                usage = usage + spent
        return Prediction({name: result[name] for name in signature.output_fields}, usage)

    def _attempt_lm(self, attempt: int) -> dspy.LM:
        """A copy with its own history, so each attempt's usage and cost can be read back.

        DSPy caches a response before parsing it, so an unparseable one is replayed on
        every rerun. A retry asks under its own ``rollout_id``, which is part of the
        cache key: a fresh call, cached on its own. Attempt 0 is the plain request, so
        rows that parse keep their cache entries.
        """
        if attempt == 0:
            return self.lm.copy()
        lm = self.lm.copy(rollout_id=attempt)
        # DSPy warns that rollout_id does nothing at temperature 0. It still changes
        # the cache key, which is all a retry needs.
        lm._warned_zero_temp_rollout = True
        return lm

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

    def _recorded_price(self) -> tuple[float, float] | None:
        """Rate to record when DSPy reports tokens but no cost. Gateways only."""
        if _gateway_upstream(self.model) is None:
            return None
        return self.price


def _portkey_ids(model: str) -> tuple[str, str]:
    """``(@catalog id, litellm id)`` for ``portkey:openai/gpt-5-nano`` or ``portkey:@openai/gpt-5-nano``.

    A missing ``@`` is added. One that is already there is kept. The LiteLLM id
    never includes it, so the price lookup is the same either way.
    """
    raw = model.split(":", 1)[1]
    catalog = raw if raw.startswith("@") else f"@{raw}"
    litellm_id = catalog[1:]
    slug, sep, upstream = litellm_id.partition("/")
    if not sep or not slug or not upstream:
        raise ValueError(f"{model!r} must be 'portkey:provider/model', e.g. 'portkey:openai/gpt-5-nano'")
    return catalog, litellm_id


def _portkey_target(model: str, kwargs: dict[str, Any]) -> str:
    """The DSPy model for a Portkey id. The wire model is ``openai/@provider/model``."""
    catalog, _ = _portkey_ids(model)
    if "api_key" not in kwargs:
        key = os.environ.get("PORTKEY_API_KEY", "").strip()
        if not key:
            raise ValueError(f"{model!r} needs PORTKEY_API_KEY or api_key=")
        kwargs["api_key"] = key
    base = os.environ.get("PORTKEY_BASE_URL", "").strip()
    kwargs.setdefault("api_base", base or "https://api.portkey.ai/v1")
    return f"openai/{catalog}"


def _gateway_upstream(model: str) -> str | None:
    """The LiteLLM id after a gateway prefix, or ``None`` when ``model`` is already one.

    ``portkey:@openai/gpt-5-nano`` drops the leading ``@``. A colon after the first
    slash is part of the id (``bedrock/anthropic.claude-v2:0``), not a gateway.
    """
    colon, slash = model.find(":"), model.find("/")
    if colon == -1 or slash == -1 or slash < colon:
        return None
    raw = model[colon + 1 :]
    if model.startswith("portkey:") and raw.startswith("@"):
        raw = raw[1:]
    return raw or None


def _gateway_price(model: str) -> tuple[float, float] | None:
    """``(input, output)`` dollars per million tokens for a gateway id.

    The gateway's own LiteLLM entry wins. When it has no rate, the id after the
    colon is used, so ``openrouter:openai/gpt-6-luna`` can take the OpenAI price.
    """
    upstream = _gateway_upstream(model)
    if upstream is None:
        return None
    provider = model.split(":", 1)[0]
    return _litellm_price(f"{provider}/{upstream}") or _litellm_price(upstream)


def _litellm_price(litellm_id: str) -> tuple[float, float] | None:
    """Public ``(input, output)`` dollars per million tokens for a LiteLLM id."""
    provider, _, upstream = litellm_id.partition("/")
    if not provider or not upstream:
        return None
    from dspy.clients.model_metadata import model_info, rate

    info = model_info(provider, upstream)
    per_in, per_out = rate(info, "input_cost_per_token"), rate(info, "output_cost_per_token")
    if per_in is None or per_out is None:
        return None
    return (per_in * 1_000_000, per_out * 1_000_000)


def _provider(model: str) -> str:
    """The routing provider: the text before ``:``, or before ``/`` when no colon is given."""
    head, sep, _ = model.partition(":")
    if sep:
        return head
    return model.split("/", 1)[0]


def _for_dspy(model: str) -> str:
    """DSPy routes on a LiteLLM id.

    A string with no colon, or a colon after the first slash, is that id and is
    kept (``openai/gpt-5-nano``, ``bedrock/anthropic.claude-v2:0``). A colon
    before any slash is a gateway prefix (``openrouter:openai/gpt-6-luna``).
    ``openai:gpt-5-nano`` is rejected.
    """
    colon, slash = model.find(":"), model.find("/")
    if colon == -1 or (slash != -1 and slash < colon):
        return model
    provider, model_id = model.split(":", 1)
    if "/" not in model_id:
        raise ValueError(
            f"{model!r} must be a LiteLLM id such as '{provider}/{model_id}'. "
            "A colon is only a gateway prefix, as in 'openrouter:openai/gpt-6-luna'."
        )
    return f"{provider}/{model_id}"


def _progress_bar(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, leave=False)


__all__ = ["Model", "Prediction", "Usage", "Budget", "BudgetExceeded", "budget", "total_usage"]
