"""The shared LM executor: model config, concurrency limits, usage accounting.

Caching comes from DSPy's own request cache (on disk by default), which keys on the
model, the rendered prompt and the call settings. Re-running a pipeline therefore only
pays for calls whose inputs changed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
from dspy.utils.exceptions import AdapterParseError

log = logging.getLogger("tracepress")

# The model the examples and docs suggest. There is no default: the user always picks one.
SUGGESTED_LM = "openrouter:openai/gpt-6-luna"

PORTKEY_BASE_URL = "https://api.portkey.ai/v1"

# USD per million (input, output) tokens, for models litellm has no price for. Extend
# with `tp.configure(prices={...})`, keyed by the litellm model name (`lm.model`).
PRICES = {"openrouter/openai/gpt-6-luna": (0.10, 0.50)}


@dataclass
class Config:
    # A `provider:model` string (see `resolve_lm`), a plain litellm model string, or a
    # dspy.LM. None means "read TRACEPRESS_LM", and fail if that isn't set either.
    lm: str | dspy.LM | None = None
    # How prompts and structured outputs are formatted. JSON parses more reliably than
    # DSPy's default text markers with the models we use.
    adapter: dspy.Adapter = field(default_factory=dspy.JSONAdapter)
    # Tried in order when a response can't be parsed. A different format is also a
    # different cache entry, so a cached bad response can't fail the same call forever.
    fallback_adapters: list[dspy.Adapter] = field(default_factory=lambda: [dspy.XMLAdapter()])
    max_concurrency: int = 32
    # Seconds before a call is abandoned and retried. Requests occasionally hang (litellm
    # would wait 10 minutes), and one hang stalls a whole fold. Measured on GPT-6 Luna:
    # per-row and fold-step calls p50 ~4s, worst ~15s; reduces scale with their output.
    # Enforced here rather than passed to the LM, so changing it keeps the cache valid.
    request_timeout: float = 30.0  # lm_map / lm_filter / lm_extract / fold steps / running updates
    reduce_timeout: float = 120.0  # full reduce and merge calls, whose prompts can be large
    timeout_retries: int = 2
    # Rendered size budget for one reduce call before it is split into a tree reduce.
    reduce_chunk_chars: int = 120_000
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(PRICES))
    # Anonymous usage telemetry, one event per run (see telemetry.py). Also off with
    # TRACEPRESS_TELEMETRY=0 or DO_NOT_TRACK=1.
    telemetry: bool = True


_config = Config()


def configure(**kwargs: Any) -> Config:
    """Set global defaults, e.g. `tp.configure(lm="openrouter:openai/gpt-6-luna", max_concurrency=64)`."""
    for name, value in kwargs.items():
        if not hasattr(_config, name):
            raise TypeError(f"unknown setting {name!r}")
        setattr(_config, name, value)
    return _config


def config() -> Config:
    return _config


_env_loaded = False


def _load_env() -> None:
    """Read `.env` in the working directory once. Variables already set win."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    env = Path.cwd() / ".env"
    if env.exists():
        from dotenv import load_dotenv

        load_dotenv(env)


def _require_env(name: str, spec: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{spec!r} needs {name}; set it in the environment or in .env")
    return value


def _openrouter(model: str, spec: str) -> dspy.LM:
    _require_env("OPENROUTER_API_KEY", spec)
    return _labeled(dspy.LM(f"openrouter/{model}"), "openrouter", model)


def _portkey(model: str, spec: str) -> dspy.LM:
    # Portkey's Model Catalog addresses models as `@<provider slug>/<model>`, sent to its
    # OpenAI-compatible gateway with the Portkey key. `portkey:openai-prod/gpt-5` and
    # `portkey:@openai-prod/gpt-5` are the same model.
    if "/" not in model:
        raise ValueError(f"{spec!r}: Portkey models look like 'portkey:<provider slug>/<model>'")
    slug_model = model if model.startswith("@") else f"@{model}"
    key = _require_env("PORTKEY_API_KEY", spec)
    lm = dspy.LM(
        f"openai/{slug_model}",
        api_base=os.environ.get("PORTKEY_BASE_URL", PORTKEY_BASE_URL),
        api_key=key,
        extra_headers={"x-portkey-api-key": key},
    )
    return _labeled(lm, "portkey", slug_model.split("/", 1)[1])  # the slug can name an org



PROVIDERS = {"openrouter": _openrouter, "portkey": _portkey}


def make_lm(spec: str) -> dspy.LM:
    """Build an LM from a model string.

    `provider:model` routes through a known gateway: `openrouter:openai/gpt-6-luna` or
    `portkey:<provider slug>/<model>`. Anything else is a plain litellm model string,
    passed through as is (`anthropic/claude-...`, `ollama_chat/llama3:8b`).
    """
    _load_env()
    prefix, sep, model = spec.partition(":")
    if sep and "/" not in prefix:
        if prefix not in PROVIDERS:
            raise ValueError(f"unknown LM provider {prefix!r} in {spec!r}; known: {', '.join(PROVIDERS)}. "
                             "Plain litellm model strings (no 'provider:') also work.")
        if not model:
            raise ValueError(f"{spec!r} names no model")
        return PROVIDERS[prefix](model, spec)
    return _labeled(dspy.LM(spec), "litellm", spec)


def _labeled(lm: dspy.LM, gateway: str, model: str) -> dspy.LM:
    lm._tp_label = (gateway, model)
    return lm


def _public_model(model: str) -> str:
    """Fine-tuned model ids (`ft:base:org::id`) name an org; keep only the base model."""
    parts = model.split("/")
    return "/".join(f"ft:{p.split(':')[1]}" if p.startswith("ft:") and p.count(":") > 1 else p for p in parts)


def model_label(lm: dspy.BaseLM) -> tuple[str, str]:
    """(gateway, model) for reporting, with no slugs, endpoints or fine-tune ids.

    LMs built by `make_lm` carry their label. Any other dspy.LM is "custom", keeping its
    model name only; other BaseLMs (fakes, wrappers) are reported as just "custom".
    """
    label = getattr(lm, "_tp_label", None)
    if label is None:
        label = ("custom", lm.model.rsplit("/", 1)[-1]) if isinstance(lm, dspy.LM) else ("custom", "custom")
    gateway, model = label
    return gateway, _public_model(model)


_lm_cache: dict[str, dspy.LM] = {}


def resolve_lm(lm: str | dspy.LM | None) -> dspy.LM:
    if lm is None:
        lm = _config.lm
    if lm is None:
        _load_env()
        lm = os.environ.get("TRACEPRESS_LM")
    if lm is None:
        raise RuntimeError(
            "No LM configured. Pick one with tp.configure(lm=...), an operator's lm=, or the "
            f'TRACEPRESS_LM environment variable, e.g. tp.configure(lm="{SUGGESTED_LM}").'
        )
    if isinstance(lm, dspy.BaseLM):
        return lm
    if lm not in _lm_cache:
        _lm_cache[lm] = make_lm(lm)
    return _lm_cache[lm]


def call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """USD for one call, from `Config.prices` or litellm's price table; None if unknown."""
    if not prompt_tokens and not completion_tokens:
        return 0.0
    price = _config.prices.get(model)
    if price is not None:
        return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1e6
    try:
        import litellm

        cost = sum(litellm.cost_per_token(model=model, prompt_tokens=prompt_tokens,
                                          completion_tokens=completion_tokens))
    except Exception:
        return None
    return cost or None  # litellm answers 0 for models it has no price for


@dataclass
class ModelStats:
    label: tuple[str, str]  # (gateway, model), see `model_label`
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = 0.0  # None once any paid call had no known price


@dataclass
class RunStats:
    calls: int = 0
    cached_calls: int = 0  # answered from DSPy's cache, so free
    errors: int = 0
    fallbacks: int = 0  # calls that needed a fallback adapter to parse
    timeouts: int = 0  # attempts abandoned after the timeout (then retried)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # for calls with a known price; see `unpriced_calls`
    unpriced_calls: int = 0  # paid calls to models with no known price
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    by_model: dict[str, ModelStats] = field(default_factory=dict)  # keyed by `lm.model`
    by_op: dict[str, int] = field(default_factory=dict)  # LM calls per operator, e.g. "lm_map"
    # One entry per call: (seconds, prompt_tokens, completion_tokens, ok). Seconds cover
    # the call itself, retries included, but not waiting for a concurrency slot.
    call_log: list[tuple[float, int, int, bool]] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    def latency(self) -> dict[str, float]:
        """Per-call latency percentiles, in seconds."""
        times = sorted(t for t, *_ in self.call_log)
        if not times:
            return {}
        pick = lambda q: times[min(len(times) - 1, int(q * len(times)))]
        return {"p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": times[-1]}

    def record(self, lm: dspy.BaseLM, op: str, seconds: float, prompt: int, completion: int,
               ok: bool, cached: bool) -> None:
        name = getattr(lm, "model", None) or type(lm).__name__
        model = self.by_model.get(name) or self.by_model.setdefault(name, ModelStats(model_label(lm)))
        model.calls += 1
        self.by_op[op] = self.by_op.get(op, 0) + 1
        self.call_log.append((seconds, prompt, completion, ok))
        if not ok:
            return
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        model.prompt_tokens += prompt
        model.completion_tokens += completion
        if cached:
            self.cached_calls += 1
            model.cached_calls += 1
            return
        cost = call_cost(name, prompt, completion)
        if cost is None:
            self.unpriced_calls += 1
            model.cost_usd = None
        else:
            self.cost_usd += cost
            if model.cost_usd is not None:
                model.cost_usd += cost

    def __repr__(self) -> str:
        cost = f"${self.cost_usd:.6f}" + (f" (+{self.unpriced_calls} unpriced calls)" if self.unpriced_calls else "")
        return (
            f"RunStats(calls={self.calls}, cached={self.cached_calls}, errors={self.errors}, "
            f"fallbacks={self.fallbacks}, timeouts={self.timeouts}, prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, cost={cost}, seconds={self.seconds:.1f})"
        )


class Executor:
    """One per run. Every lm_ operator sends its calls through here."""

    def __init__(self, max_concurrency: int | None = None):
        self.semaphore = asyncio.Semaphore(max_concurrency or _config.max_concurrency)
        self.stats = RunStats()

    async def call(self, module: dspy.Module, lm: dspy.LM, inputs: dict[str, Any],
                   timeout: float | None = None, op: str = "lm") -> dspy.Prediction:
        timeout = timeout or _config.request_timeout
        async with self.semaphore:
            self.stats.calls += 1
            start = time.monotonic()
            try:
                for attempt in range(_config.timeout_retries + 1):
                    try:
                        pred = await asyncio.wait_for(self._predict(module, lm, inputs), timeout)
                        break
                    except asyncio.TimeoutError:
                        self.stats.timeouts += 1
                        if attempt == _config.timeout_retries:
                            raise TimeoutError(f"LM call timed out {attempt + 1}x after {timeout:.0f}s") from None
            except Exception:
                self.stats.record(lm, op, time.monotonic() - start, 0, 0, ok=False, cached=False)
                raise
        # DSPy only records usage for requests that missed its cache.
        usage = pred.get_lm_usage() or {}
        prompt = sum(u.get("prompt_tokens") or 0 for u in usage.values())
        completion = sum(u.get("completion_tokens") or 0 for u in usage.values())
        self.stats.record(lm, op, time.monotonic() - start, prompt, completion, ok=True, cached=not usage)
        return pred

    async def _predict(self, module: dspy.Module, lm: dspy.LM, inputs: dict[str, Any]) -> dspy.Prediction:
        adapters = [_config.adapter, *_config.fallback_adapters]
        for i, adapter in enumerate(adapters):
            try:
                with dspy.context(lm=lm, adapter=adapter, track_usage=True):
                    return await module.acall(**inputs)
            except AdapterParseError:
                if i == len(adapters) - 1:
                    raise
                self.stats.fallbacks += 1
