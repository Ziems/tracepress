"""The shared LM executor: model config, concurrency limits, usage accounting.

Caching comes from DSPy's own request cache (on disk by default), which keys on the
model, the rendered prompt and the call settings. Re-running a pipeline therefore only
pays for calls whose inputs changed. In front of it, the prediction cache answers a call
from what its prompt is rendered from, before DSPy renders it (see `prediction_request`).
"""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import inspect
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import dspy
from dspy.utils.exceptions import AdapterParseError
from dspy.utils.usage_tracker import track_usage

log = logging.getLogger("tracepress")

# The model the examples and docs suggest. There is no default: the user always picks one.
SUGGESTED_LM = "openrouter:openai/gpt-6-luna"

PORTKEY_BASE_URL = "https://api.portkey.ai/v1"

# USD per million (input, output) tokens, for models DSPy's price table doesn't cover.
# Extend with `tp.configure(prices={...})`, keyed by the model name (`lm.model`).
PRICES = {
    "openrouter/openai/gpt-6-luna": (0.10, 0.50),
    # typesafe.ai lists $42 per billion input tokens; it lists no output price, so this assumes
    # the same. Jev's outputs are a few tokens per decision, so the assumption barely matters.
    "typesafe:jev-latest": (0.042, 0.042),
}


@dataclass
class Config:
    # A `provider:model` string (see `resolve_lm`), a plain DSPy model string, or a
    # dspy.LM. None means "read TRACEPRESS_LM", and fail if that isn't set either.
    lm: str | dspy.LM | None = None
    # How prompts and structured outputs are formatted. JSON parses more reliably than
    # DSPy's default text markers with the models we use.
    adapter: dspy.Adapter = field(default_factory=dspy.JSONAdapter)
    # Tried in order when a response can't be parsed. A different format is also a
    # different cache entry, so a cached bad response can't fail the same call forever.
    fallback_adapters: list[dspy.Adapter] = field(default_factory=lambda: [dspy.XMLAdapter()])
    max_concurrency: int = 128
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
    # Look a call's answer up before DSPy renders its prompt (see `prediction_request`). A hit
    # skips prompt rendering and parsing. Off, every call goes through DSPy and its cache.
    prediction_cache: bool = True


_config = Config()


def configure(**kwargs: Any) -> Config:
    """Set global defaults, e.g. `tp.configure(lm="openrouter:openai/gpt-6-luna", max_concurrency=256)`."""
    for name, value in kwargs.items():
        if not hasattr(_config, name):
            raise TypeError(f"unknown setting {name!r}")
        if name == "prices":
            value = {**_config.prices, **value}  # added to the prices so far (built-ins included), not a new table
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
    # For models with no known price (e.g. ones newer than DSPy's price table), ask
    # OpenRouter to report each call's cost. Only for those: the flag changes the request,
    # so adding it to priced models would miss every response already cached for them.
    priced = call_cost(f"openrouter/{model}", 1, 1)
    kwargs: dict[str, Any] = {} if priced else {"extra_body": {"usage": {"include": True}}}
    return _labeled(dspy.LM(f"openrouter/{model}", **kwargs), "openrouter", model)


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


def _typesafe(model: str, spec: str) -> dspy.LM:
    # TypeSafe's System One models (Jev) answer decision outputs (Noul, Score, Choice) with
    # probabilities instead of generating text. DSPy routes `typesafe:` models natively.
    _require_env("TYPESAFE_API_KEY", spec)
    return _labeled(dspy.LM(f"typesafe:{model}"), "typesafe", model)


PROVIDERS = {"openrouter": _openrouter, "portkey": _portkey, "typesafe": _typesafe}


def make_lm(spec: str) -> dspy.LM:
    """Build an LM from a model string.

    `provider:model` routes through a known gateway: `openrouter:openai/gpt-6-luna`,
    `portkey:<provider slug>/<model>`, or `typesafe:jev-latest` (decision outputs only).
    Anything else is a plain DSPy model string,
    passed through as is (`anthropic/claude-...`, `ollama_chat/llama3:8b`).
    """
    _load_env()
    prefix, sep, model = spec.partition(":")
    if sep and "/" not in prefix:
        if prefix not in PROVIDERS:
            raise ValueError(
                f"unknown LM provider {prefix!r} in {spec!r}; known: {', '.join(PROVIDERS)}. "
                "Plain DSPy model strings (no 'provider:') also work."
            )
        if not model:
            raise ValueError(f"{spec!r} names no model")
        return PROVIDERS[prefix](model, spec)
    return _labeled(dspy.LM(spec), "litellm", spec)


def _labeled(lm: dspy.LM, gateway: str, model: str) -> dspy.LM:
    vars(lm)["_tp_label"] = (gateway, model)
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
    if not isinstance(lm, str):  # a dspy.LM or another dspy.BaseLM
        return lm
    if lm not in _lm_cache:
        _lm_cache[lm] = make_lm(lm)
    return _lm_cache[lm]


def call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """USD for one call, from `Config.prices` or DSPy's price table; None if unknown."""
    if not prompt_tokens and not completion_tokens:
        return 0.0
    price = _config.prices.get(model) or _table_price(model)
    if price is None:
        return None
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1e6 or None


def _table_price(model: str) -> tuple[float, float] | None:
    """USD per million (input, output) tokens from DSPy's bundled price table, if listed."""
    try:
        from dspy.clients.model_metadata import model_info, rate

        match = re.match(r"([\w-]+)[/:](.+)", model)
        provider, name = match.groups() if match else ("openai", model)  # bare names are OpenAI's
        info = model_info(provider, name)
        prices = rate(info, "input_cost_per_token"), rate(info, "output_cost_per_token")
    except Exception:  # noqa: BLE001
        return None
    if prices[0] is None and prices[1] is None:
        return None
    return (prices[0] or 0.0) * 1e6, (prices[1] or 0.0) * 1e6


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
    cached_calls: int = 0  # answered from the prediction cache or DSPy's, so free
    errors: int = 0
    fallbacks: int = 0  # calls that needed a fallback adapter to parse
    timeouts: int = 0  # attempts abandoned after the timeout (then retried)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # for calls with a known price; see `unpriced_calls`
    unpriced_calls: int = 0  # paid calls to models with no known price
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    status: str = "running"  # then "completed", "failed" or "cancelled"; never changes after that
    error: BaseException | None = None  # what ended the run, unless it completed
    by_model: dict[str, ModelStats] = field(default_factory=dict)  # keyed by `lm.model`
    by_op: dict[str, int] = field(default_factory=dict)  # LM calls per operator, e.g. "lm_map"
    # One entry per call: (seconds, prompt_tokens, completion_tokens, ok). Seconds cover
    # the call itself, retries included, but not waiting for a concurrency slot.
    call_log: list[tuple[float, int, int, bool]] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    def conclude(self, error: BaseException | None) -> None:
        """Record how the run ended. `Exception`s failed it; anything else (Ctrl-C, a cancel) stopped it."""
        self.finished = time.monotonic()
        self.error = error
        self.status = "completed" if error is None else "failed" if isinstance(error, Exception) else "cancelled"

    def latency(self) -> dict[str, float]:
        """Per-call latency percentiles, in seconds."""
        times = sorted(t for t, *_ in self.call_log)
        if not times:
            return {}
        pick = lambda q: times[min(len(times) - 1, int(q * len(times)))]
        return {"p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": times[-1]}

    def record(
        self,
        lm: dspy.BaseLM,
        op: str,
        seconds: float,
        prompt: int,
        completion: int,
        ok: bool,
        cached: bool,
        reported_cost: float | None = None,
    ) -> None:
        """`reported_cost` is what the provider billed, when it says; it wins over the price table."""
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
        cost = reported_cost if reported_cost is not None else call_cost(name, prompt, completion)
        if cost is None:
            self.unpriced_calls += 1
            model.cost_usd = None
        else:
            self.cost_usd += cost
            if model.cost_usd is not None:
                model.cost_usd += cost

    def __repr__(self) -> str:
        cost = f"${self.cost_usd:.6f}" + (f" (+{self.unpriced_calls} unpriced calls)" if self.unpriced_calls else "")
        why = f" ({type(self.error).__name__})" if self.error is not None else ""
        status = "" if self.status == "completed" else f"status={self.status}{why}, "
        return (
            f"RunStats({status}calls={self.calls}, cached={self.cached_calls}, errors={self.errors}, "
            f"fallbacks={self.fallbacks}, timeouts={self.timeouts}, prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, cost={cost}, seconds={self.seconds:.1f})"
        )


# Bump to make every prediction-cache entry miss: after a change to how a call's inputs are
# built or its answer is read that the versions below don't show (an editable install keeps
# its version while its code changes).
PREDICTION_CACHE_VERSION = 1


@functools.cache
def _versions() -> tuple[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        own = version("tracepress")
    except PackageNotFoundError:
        own = "unknown"
    return own, dspy.__version__


_type_adapters: dict[Any, Any] = {}  # (annotation, its repr) -> pydantic TypeAdapter
_type_ids: dict[Any, Any] = {}  # (annotation, its repr) -> _type_id


def _memo_key(annotation: Any) -> Any:
    # Types compare equal regardless of order (Literal["a", "b"] == Literal["b", "a"]), so the
    # repr goes in too. An annotation that can't be hashed is known by its repr alone.
    try:
        key = (annotation, repr(annotation))
        hash(key)
        return key
    except TypeError:
        return repr(annotation)


def _type_adapter(annotation: Any) -> Any:
    import pydantic

    key = _memo_key(annotation)
    if key not in _type_adapters:
        _type_adapters[key] = pydantic.TypeAdapter(annotation)
    return _type_adapters[key]


def _type_id(annotation: Any) -> Any:
    # The repr keeps what the schema may not (Literal order); the schema holds what the repr
    # doesn't (a model's fields, an enum's members).
    key = _memo_key(annotation)
    if key not in _type_ids:
        try:
            _type_ids[key] = [repr(annotation), _type_adapter(annotation).json_schema()]
        except Exception:  # noqa: BLE001 - a type pydantic can't describe is keyed by its repr
            _type_ids[key] = repr(annotation)
    return _type_ids[key]


def _keyable(value: Any) -> Any:
    """`value` as JSON with its types spelled out, so inputs DSPy renders differently key differently.

    Plain JSON would make NaN, inf and None alike (all `null`), and tuples like lists.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ["<float>", repr(value)]
    # Every container is tagged, so a list can't pass for one (["<tuple>", [1]] isn't (1,)).
    if isinstance(value, (list, tuple)):
        return [f"<{type(value).__name__}>", [_keyable(v) for v in value]]
    if isinstance(value, dict):  # kept in order, with keys of any type: both reach the prompt
        return ["<dict>", [[_keyable(k), _keyable(v)] for k, v in value.items()]]
    if isinstance(value, (set, frozenset)):  # in an order that doesn't change between runs
        return [f"<{type(value).__name__}>", sorted((_keyable(v) for v in value), key=repr)]
    if hasattr(value, "model_dump"):
        return [f"<{type(value).__module__}.{type(value).__qualname__}>", _keyable(value.model_dump(mode="json"))]
    return [f"<{type(value).__module__}.{type(value).__qualname__}>", repr(value)]


@functools.cache
def _source_id(cls: type) -> str:
    """A hash of the code of a module class and the classes it inherits from, DSPy's aside.

    DSPy's own modules are covered by the DSPy version. A class whose source can't be read
    (defined in `python -c`, say) raises, so its calls aren't cached. Code a class calls
    outside itself, such as a module-level helper, isn't covered.
    """
    sources = [inspect.getsource(c) for c in cls.__mro__ if c is not object and c.__module__.split(".")[0] != "dspy"]
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()


def _module_settings(module: dspy.Module) -> dict[str, Any]:
    # A module's own plain settings (Refine's N and threshold, say). Predictors are keyed on
    # their own, and DSPy's bookkeeping (a random stage id, call history) changes every run.
    return {
        k: v
        for k, v in vars(module).items()
        if isinstance(v, (bool, int, float, str, type(None))) and k not in ("stage", "_compiled")
    }


def _signature_id(signature: Any) -> list:
    return [
        signature.instructions,
        [
            (name, _type_id(f.annotation), _keyable(f.json_schema_extra or {}), repr(f.default))
            for name, f in signature.fields.items()
        ],
    ]


def _adapter_id(adapter: dspy.Adapter) -> tuple:
    # Its class and plain settings (use_native_function_calling, ...): not callbacks or caches.
    settings = {k: v for k, v in vars(adapter).items() if isinstance(v, (bool, int, float, str, type(None)))}
    return type(adapter).__module__, type(adapter).__qualname__, settings


# Not part of what a call asks for: DSPy leaves these out of its own cache key too.
_UNKEYED_LM_KWARGS = ("api_key", "api_base", "base_url")
# Part of the call but can hold a credential (Portkey's key rides in a header): keyed by hash.
_HASHED_LM_KWARGS = ("extra_headers",)


def _lm_id(lm: dspy.LM) -> list:
    engine = lm.engine if isinstance(lm.engine, str) else f"{type(lm.engine).__module__}.{type(lm.engine).__qualname__}"
    return [
        f"{type(lm).__module__}.{type(lm).__qualname__}",
        engine,
        lm.model,
        lm.model_type,
        getattr(lm, "use_developer_role", False),
        _keyable(
            {
                k: hashlib.sha256(repr(_keyable(v)).encode()).hexdigest() if k in _HASHED_LM_KWARGS else v
                for k, v in lm.kwargs.items()
                if k not in _UNKEYED_LM_KWARGS
            }
        ),
    ]


def prediction_request(module: dspy.Module, lm: dspy.BaseLM, inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Everything that decides a call's answer, as a cache request; None if the call isn't cached.

    It holds what the prompt is rendered from, not the prompt: each predictor's signature,
    demos and settings, the module (its plain settings, and its classes' code for a module of
    your own), the LM and its settings, the adapters, the inputs, and the TracePress and DSPy
    versions, since a new version may render the same call differently. DSPy's own cache
    renders the prompt first.
    """
    if not _config.prediction_cache or not isinstance(lm, dspy.LM) or not lm.cache:
        return None
    try:
        return _prediction_request(module, lm, inputs)
    except Exception:  # something we can't key on: leave the call to DSPy
        log.debug("tracepress: not caching a call whose request can't be keyed", exc_info=True)
        return None


def _prediction_request(module: dspy.Module, lm: dspy.LM, inputs: dict[str, Any]) -> dict[str, Any]:
    adapters = [_config.adapter, *_config.fallback_adapters]
    return {
        "_fn_identifier": "tracepress.prediction",  # keeps these entries apart from DSPy's own
        "version": [PREDICTION_CACHE_VERSION, *_versions()],
        "module": [
            f"{type(module).__module__}.{type(module).__qualname__}",
            _source_id(type(module)),
            _keyable(_module_settings(module)),
        ],
        "predictors": [_predictor_id(name, p) for name, p in _predictors(module)],
        "lm": _lm_id(lm),
        "adapters": [_adapter_id(a) for a in adapters],
        "inputs": _keyable(inputs),
    }


def _predictor_id(name: str, predictor: Any) -> list:
    if predictor.lm is not None and not isinstance(predictor.lm, dspy.LM):
        raise TypeError("a predictor with an LM of its own that isn't a dspy.LM")  # not cached
    return [
        name,
        _signature_id(predictor.signature),
        _keyable([d.toDict() for d in predictor.demos]),
        _keyable(predictor.config),
        _keyable(getattr(predictor, "fields", None)),
        _lm_id(predictor.lm) if predictor.lm is not None else None,
    ]  # its own LM wins over the call's


# Where DSPy keeps its per-call bookkeeping: lists that grow with every call (of dicts, for
# the history). A module of your own may use the same names for predictors; see `_bookkeeping`.
_BOOKKEEPING = frozenset({"history", "traces", "train", "callbacks", "demos"})


def _predictors(module: dspy.Module) -> list[tuple[str, Any]]:
    """`module.named_predictors()`, without walking DSPy's per-call bookkeeping.

    DSPy looks through every attribute, including a module's call history, so asking it on
    every call costs more the longer a run goes. This walks the same way but skips DSPy's
    bookkeeping lists (see `_bookkeeping`).
    """
    from dspy.predict.parameter import Parameter

    def _bookkeeping(name: str, value: Any) -> bool:
        # One of DSPy's lists, judged by its ends so the check costs the same however long it
        # grows: a list of predictors or modules under one of these names is still walked.
        if name not in _BOOKKEEPING or not isinstance(value, list):
            return False
        return not value or not any(isinstance(v, (Parameter, dspy.Module)) for v in (value[0], value[-1]))

    def parameters(owner: Any) -> list[tuple[str, Any]]:  # as `BaseModule.named_parameters`
        seen: set[int] = set()
        found: list[tuple[str, Any]] = []

        def add(name: str, value: Any) -> None:
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    found.append((name, value))
            elif isinstance(value, dspy.Module) and not getattr(value, "_compiled", False):
                for sub, param in parameters(value):
                    add(f"{name}.{sub}", param)

        if isinstance(owner, Parameter):
            add("self", owner)
        for name, value in vars(owner).items():
            if _bookkeeping(name, value):
                continue
            if isinstance(value, (Parameter, dspy.Module)):
                add(name, value)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    add(f"{name}[{i}]", item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    add(f"{name}['{key}']", item)
        return found

    return [(name, p) for name, p in parameters(module) if isinstance(p, dspy.Predict)]


def _output_types(module: dspy.Module) -> dict[str, Any]:
    types: dict[str, Any] = {}
    for _, predictor in _predictors(module):
        types.update({name: f.annotation for name, f in predictor.signature.output_fields.items()})
    return types


def _plain(value: Any) -> bool:
    # Exact types: a subclass (a str Enum, say) is an object the disk cache won't load back.
    if type(value) in (type(None), bool, int, float, str):
        return True
    if type(value) is list:
        return all(_plain(v) for v in value)
    return type(value) is dict and all(type(k) is str and _plain(v) for k, v in value.items())


def _stored_outputs(module: dspy.Module, pred: dspy.Prediction) -> dict[str, Any] | None:
    """The answer as plain data, or None if it can't be stored exactly.

    Plain data, not the objects themselves: DSPy's disk cache won't load arbitrary classes in
    a new process, and a stored object would be shared with the caller who got it. A plain
    value is kept as it is; any other is dumped by its output type and kept only if it reads
    back equal, and of the same type.
    """
    types, outputs = _output_types(module), dict(pred.items())
    if not outputs or set(outputs) - set(types):
        return None  # a module whose answer isn't its predictors' outputs
    stored = {}
    for name, value in outputs.items():
        if _plain(value):
            stored[name] = ["plain", copy.deepcopy(value)]
            continue
        try:
            dumped = _type_adapter(types[name]).dump_python(value, mode="json", warnings=False)
            back = _type_adapter(types[name]).validate_python(dumped)
        except Exception:  # noqa: BLE001 - an answer that can't be stored just isn't
            return None
        if type(back) is not type(value) or back != value:
            return None
        stored[name] = ["typed", dumped]
    return stored


def _read_outputs(types: dict[str, Any], stored: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value if kind == "plain" else _type_adapter(types[name]).validate_python(value)
        for name, (kind, value) in stored.items()
    }


def _cached_prediction(module: dspy.Module, request: dict[str, Any]) -> dspy.Prediction | None:
    try:
        stored = dspy.cache.get(request)
        return None if stored is None else dspy.Prediction(**_read_outputs(_output_types(module), stored))
    except Exception:  # an entry that no longer reads, or a cache that failed: ask DSPy instead
        log.debug("tracepress: prediction cache lookup failed", exc_info=True)
        return None


class Executor:
    """One per run. Every lm_ operator sends its calls through here."""

    def __init__(self, max_concurrency: int | None = None):
        self.semaphore = asyncio.Semaphore(max_concurrency or _config.max_concurrency)
        self.stats = RunStats()
        self.stop: asyncio.Future | None = None  # set by the runtime; failing it stops the run

    async def call(
        self, module: dspy.Module, lm: dspy.LM, inputs: dict[str, Any], timeout: float | None = None, op: str = "lm"
    ) -> dspy.Prediction:
        timeout = timeout or _config.request_timeout
        start = time.monotonic()
        request = prediction_request(module, lm, inputs)
        if request is not None and (cached := _cached_prediction(module, request)) is not None:
            self.stats.calls += 1  # a hit makes no LM call, so it needn't wait for a slot
            self.stats.record(lm, op, time.monotonic() - start, 0, 0, ok=True, cached=True)
            return cached
        async with self.semaphore:
            self.stats.calls += 1
            start = time.monotonic()
            try:
                with track_usage() as tracker:
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
        # Every request the call made, including responses no adapter could parse. DSPy records
        # none for requests its cache answered.
        usage = tracker.get_total_tokens()
        prompt = sum(u.get("prompt_tokens") or 0 for u in usage.values())
        completion = sum(u.get("completion_tokens") or 0 for u in usage.values())
        costs = [u["cost"] for u in usage.values() if u.get("cost") is not None]  # e.g. OpenRouter's usage.cost
        self.stats.record(
            lm,
            op,
            time.monotonic() - start,
            prompt,
            completion,
            ok=True,
            cached=not usage,
            reported_cost=sum(costs) if costs else None,
        )
        if request is not None:
            self._remember(module, request, pred)
        return pred

    @staticmethod
    def _remember(module: dspy.Module, request: dict[str, Any], pred: dspy.Prediction) -> None:
        """Store a paid answer in the prediction cache; failing to never fails the call."""
        try:
            if (stored := _stored_outputs(module, pred)) is not None:
                dspy.cache.put(request, stored)
        except Exception:
            log.debug("tracepress: couldn't store an answer in the prediction cache", exc_info=True)

    async def _predict(self, module: dspy.Module, lm: dspy.LM, inputs: dict[str, Any]) -> dspy.Prediction:
        adapters = [_config.adapter, *_config.fallback_adapters]
        for i, adapter in enumerate(adapters):
            try:
                with dspy.context(lm=lm, adapter=adapter, track_usage=True):
                    return await module.acall(**inputs)
            except AdapterParseError as e:
                if i < len(adapters) - 1:
                    self.stats.fallbacks += 1
                    continue
                if (answer := bare_answer(e)) is None:
                    raise
                self.stats.fallbacks += 1
                return answer
        raise AssertionError("no adapter to try")


def bare_answer(error: AdapterParseError) -> dspy.Prediction | None:
    """The response as the answer, when the model wrote just the text of a lone text output.

    Models sometimes skip the format for a one-field answer, e.g. `-> digest`. Every adapter
    then fails on the same (cached) response, so the row would be skipped on every run.
    """
    outputs = cast(type[dspy.Signature], error.signature).output_fields
    text = (error.lm_response or "").strip()
    if len(outputs) != 1 or not text or text.startswith(("{", "[", "<")):
        return None
    ((name, field_info),) = outputs.items()
    return dspy.Prediction(**{name: text}) if field_info.annotation is str else None
