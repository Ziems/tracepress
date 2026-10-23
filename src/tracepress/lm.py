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

DEFAULT_LM = "openrouter/openai/gpt-6-luna"


@dataclass
class Config:
    lm: str | dspy.LM = field(default_factory=lambda: os.environ.get("TRACEPRESS_LM", DEFAULT_LM))
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


_config = Config()


def configure(**kwargs: Any) -> Config:
    """Set global defaults, e.g. `tp.configure(lm="openrouter/openai/gpt-6-sol", max_concurrency=64)`."""
    for name, value in kwargs.items():
        if not hasattr(_config, name):
            raise TypeError(f"unknown setting {name!r}")
        setattr(_config, name, value)
    return _config


def config() -> Config:
    return _config


def _load_env() -> None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    env = Path.cwd() / ".env"
    if env.exists():
        from dotenv import load_dotenv

        load_dotenv(env)


_lm_cache: dict[str, dspy.LM] = {}


def resolve_lm(lm: str | dspy.LM | None) -> dspy.LM:
    lm = lm if lm is not None else _config.lm
    if isinstance(lm, dspy.BaseLM):
        return lm
    if lm not in _lm_cache:
        _load_env()
        _lm_cache[lm] = dspy.LM(lm)
    return _lm_cache[lm]


@dataclass
class RunStats:
    calls: int = 0
    errors: int = 0
    fallbacks: int = 0  # calls that needed a fallback adapter to parse
    timeouts: int = 0  # attempts abandoned after the timeout (then retried)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
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

    def __repr__(self) -> str:
        return (
            f"RunStats(calls={self.calls}, errors={self.errors}, fallbacks={self.fallbacks}, timeouts={self.timeouts}, "
            f"prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, seconds={self.seconds:.1f})"
        )


class Executor:
    """One per run. Every lm_ operator sends its calls through here."""

    def __init__(self, max_concurrency: int | None = None):
        self.semaphore = asyncio.Semaphore(max_concurrency or _config.max_concurrency)
        self.stats = RunStats()

    async def call(self, module: dspy.Module, lm: dspy.LM, inputs: dict[str, Any],
                   timeout: float | None = None) -> dspy.Prediction:
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
                self.stats.call_log.append((time.monotonic() - start, 0, 0, False))
                raise
        prompt = completion = 0
        for usage in (pred.get_lm_usage() or {}).values():
            prompt += usage.get("prompt_tokens") or 0
            completion += usage.get("completion_tokens") or 0
        self.stats.prompt_tokens += prompt
        self.stats.completion_tokens += completion
        self.stats.call_log.append((time.monotonic() - start, prompt, completion, True))
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
