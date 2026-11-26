"""Anonymous usage telemetry: PostHog events built from an allowlist.

Events: `run_completed` (one per run), `optimize_completed` (one per `optimize()`; the runs
it makes inside are not reported on their own), `server_started` and `pipeline_registered`.

What is sent: library and Python versions, the OS name, run outcome and duration, call,
cache, token and cost counts, which gateways and models were used (no Portkey slugs,
endpoints or fine-tune ids), how many of each operator the plan has, the kinds of
sources, and which built-in pipelines ran. Nothing from the data, prompts, signatures,
instructions, column names, paths or error messages is ever sent.

Identity is a random install id in `~/.config/tracepress/telemetry.json`, not derived
from the machine; delete the file to reset it.

Off with `TRACEPRESS_TELEMETRY=0`, `DO_NOT_TRACK=1` or `tp.configure(telemetry=False)`,
and automatically in CI and under pytest. `TRACEPRESS_TELEMETRY=debug` prints the event
instead of sending it.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import sys
import threading
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import lm as lm_config

if TYPE_CHECKING:
    from .lm import RunStats
    from .plan import Node

# PostHog's project key is public by design: it can only write events.
POSTHOG_API_KEY = "phc_oj47o0vexqRFGmRy1u5Bm2aSx0CvhiHDygSUU6bAuix"
POSTHOG_HOST = "https://us.i.posthog.com"
EVENT = "run_completed"
TIMEOUT = 2.0

DOCS = "https://github.com/Ziems/tracepress/blob/main/docs/telemetry.md"
NOTICE = (
    "TracePress collects anonymous usage stats (versions, call/token/cost counts, model and "
    "operator names; never your data, prompts or paths). Turn it off with TRACEPRESS_TELEMETRY=0. "
    f"Details: {DOCS}"
)

# Plan node labels (see plan.py); anything else is reported as "other".
OPS = {"map", "filter", "select", "explode", "reduce", "lm_map", "lm_filter", "lm_extract", "lm_reduce", "lm_fold", "join",
       "edit"}  # edit: the node `Table.rerun` inserts
SOURCES = {
    "HarborAdapter": "harbor", "ATIFAdapter": "atif", "RowsAdapter": "rows", "ReplayAdapter": "replay",
    "ClaudeCodeAdapter": "claude_code", "CodexAdapter": "codex", "OTelAdapter": "otel", "OTLPReceiver": "otlp",
    "ServerSource": "server",
}
PUBLIC_ERROR_MODULES = ("builtins", "asyncio", "concurrent", "tracepress", "dspy", "litellm", "pydantic", "json")

_pending: list[threading.Thread] = []  # sends still in flight, for _flush to wait on at exit
_pending_lock = threading.Lock()
_quiet = 0
_quiet_lock = threading.Lock()


@contextmanager
def quiet():
    """Runs inside don't send their own `run_completed`: `optimize()` sends one event for the
    whole search instead of one per candidate prompt."""
    global _quiet
    with _quiet_lock:
        _quiet += 1
    try:
        yield
    finally:
        with _quiet_lock:
            _quiet -= 1


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def mode() -> str:
    """'off', 'debug' (print, don't send) or 'on'."""
    setting = (os.environ.get("TRACEPRESS_TELEMETRY") or "").strip().lower()
    if not lm_config.config().telemetry or _truthy(os.environ.get("DO_NOT_TRACK")):
        return "off"
    if setting == "debug":
        return "debug"
    if setting in {"0", "false", "off", "no"}:
        return "off"
    if _automated():
        return "off"
    return "on" if POSTHOG_API_KEY else "off"


def _automated() -> bool:
    """CI or a test run: usage from these isn't a person using TracePress."""
    return any(os.environ.get(name) for name in ("CI", "GITHUB_ACTIONS", "PYTEST_CURRENT_TEST"))


def _state_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "tracepress" / "telemetry.json"


def install_id() -> str | None:
    """The random install id, created (with a one-time notice) on first use; None if it can't be stored."""
    path = _state_path()
    try:
        return json.loads(path.read_text())["install_id"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    state = {"install_id": str(uuid.uuid4()), "notice_shown": True}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state))
    except OSError:
        return None  # without a stored id the notice would repeat every run, so don't send
    print(NOTICE, file=sys.stderr)
    return state["install_id"]


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


def _error_type(error: BaseException | None) -> str | None:
    if error is None:
        return None
    module = type(error).__module__.split(".")[0]
    return type(error).__name__ if module in PUBLIC_ERROR_MODULES else "other"


def _walk(nodes: list[Node]) -> list[Node]:
    seen: dict[int, Node] = {}
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen[id(node)] = node
        stack.extend([node.parent, *node.others])
    return list(seen.values())


def _source_kinds(op: Any) -> list[str]:
    kinds, adapter = [], getattr(op, "adapter", None)
    while adapter is not None:
        kinds.append(SOURCES.get(type(adapter).__name__, "custom"))
        adapter = getattr(adapter, "source", None) if type(adapter).__name__ == "ReplayAdapter" else None
    return kinds


def _emit_mode(op: Any) -> str | None:
    if not hasattr(op, "emit_every") or op.label not in {"lm_reduce", "lm_fold"}:
        return None
    return {None: "close", 1: "update"}.get(op.emit_every, "every")


def _count(names) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in names:
        out[name] = out.get(name, 0) + 1
    return out


def _common() -> dict[str, Any]:
    """Properties every event carries."""
    return {
        "tracepress_version": _version("tracepress"),
        "dspy_version": _version("dspy"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "os": platform.system(),
        "$lib": "tracepress",
        "$process_person_profile": False,
        "$geoip_disable": True,
    }


def _op_counts(plan: list[Node]) -> dict[str, int]:
    return _count(n.label if n.label in OPS else "other" for n in plan if n.parent is not None)


def payload(stats: RunStats, nodes: list[Node], operators: dict[int, Any] | None = None,
            error: BaseException | None = None, lineage: bool = False) -> dict[str, Any]:
    """The event properties for a run: exactly what `report` would send."""
    operators = operators or {}
    plan = _walk(nodes)
    built = [operators[id(n)] for n in plan if id(n) in operators]
    sources = sorted({k for op in built for k in _source_kinds(op)})
    models = sorted(
        ({"gateway": m.label[0], "model": m.label[1], "calls": m.calls, "cached_calls": m.cached_calls,
          "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens,
          "cost_usd": None if m.cost_usd is None else round(m.cost_usd, 6)} for m in stats.by_model.values()),
        key=lambda m: -m["calls"],
    )
    latency = stats.latency()
    return {
        **_common(),
        "ok": error is None,
        "error_type": _error_type(error),
        "duration_s": round(stats.seconds, 3),
        "calls": stats.calls,
        "cached_calls": stats.cached_calls,
        "errors": stats.errors,
        "fallbacks": stats.fallbacks,
        "timeouts": stats.timeouts,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "cost_usd": round(stats.cost_usd, 6),
        "unpriced_calls": stats.unpriced_calls,
        "latency_p50": latency.get("p50"),
        "latency_p90": latency.get("p90"),
        "models": models,
        "primary_model": f"{models[0]['gateway']}:{models[0]['model']}" if models else None,
        "llm_calls_by_op": {op: n for op, n in stats.by_op.items() if op in OPS},
        "ops": _op_counts(plan),
        "n_outputs": len(nodes),
        "live": any(getattr(op, "live", False) for op in built),
        "emit_modes": sorted({m for m in map(_emit_mode, built) if m}),
        "sources": sources,
        "pipelines": sorted({n.pipeline for n in plan if n.pipeline}),
        "lineage": lineage,
    }


def report(stats: RunStats, nodes: list[Node], operators: dict[int, Any] | None = None,
           error: BaseException | None = None, lineage: bool = False) -> None:
    """Send (or, in debug mode, print) the run's event. Never raises."""
    if _quiet:
        return
    _emit(EVENT, lambda: payload(stats, nodes, operators, error, lineage))


def report_optimize(*, optimizer: str, step_op: str, metric: str, n_labels: int, n_val: int, budget: int,
                    result: Any = None, seconds: float, error: BaseException | None = None) -> None:
    """One event for a whole `optimize()`: what was tuned and what it cost, never the prompts."""

    def properties() -> dict[str, Any]:
        out = {**_common(), "ok": error is None, "error_type": _error_type(error), "optimizer": optimizer,
               "step_op": step_op if step_op in OPS else "other", "metric": metric, "n_labels": n_labels,
               "n_val": n_val, "budget": budget, "duration_s": round(seconds, 3)}
        if result is not None:
            out.update({"prompts_tried": result.tried, "prompts_kept": result.candidates,
                        "lm_calls": result.lm_calls, "cost_usd": round(result.cost_usd, 6),
                        "val_before": round(result.val_before, 4), "val_after": round(result.val_after, 4)})
        return out

    _emit("optimize_completed", properties)


def report_server_started(*, local: bool, token: bool, retain: bool, default_lm: bool) -> None:
    """A TracePress server started: where it listens (local or not) and how it's set up, never the address."""
    _emit("server_started", lambda: {**_common(), "local": local, "token": token, "retain": retain,
                                     "default_lm": default_lm})


def report_pipeline_registered(node: Node, *, replaced: bool) -> None:
    """A client registered a pipeline on a server: its shape, counted when it starts, not when it ends."""
    _emit("pipeline_registered", lambda: {**_common(), "ops": _op_counts(_walk([node])), "replaced": replaced,
                                          "pipelines": sorted({n.pipeline for n in _walk([node]) if n.pipeline})})


def _emit(event: str, properties) -> None:
    """Build and send (or print, in debug mode) one event. Never raises."""
    try:
        current = mode()
        if current == "off":
            return
        props = properties()
        if current == "debug":
            print(f"[tracepress telemetry, not sent] {event} {json.dumps(props)}", file=sys.stderr)
            return
        distinct_id = install_id()
        if distinct_id is None:
            return
        body = {"api_key": POSTHOG_API_KEY, "event": event, "distinct_id": distinct_id, "properties": props}
        thread = threading.Thread(target=_send, args=(body,), daemon=True)
        with _pending_lock:
            _pending[:] = [t for t in _pending if t.is_alive()]  # finished sends would pile up for the process's life
            _pending.append(thread)
            thread.start()  # under the lock: a thread not started yet isn't alive, and would be pruned
    except Exception:
        pass


def _send(body: dict[str, Any]) -> None:
    try:
        host = os.environ.get("TRACEPRESS_TELEMETRY_HOST", POSTHOG_HOST).rstrip("/")
        request = urllib.request.Request(f"{host}/i/v0/e/", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(request, timeout=TIMEOUT).close()
    except Exception:
        pass


@atexit.register
def _flush() -> None:
    """Give in-flight events up to a second at exit; never hold up shutdown longer."""
    for thread in _pending:
        thread.join(timeout=1.0 / max(1, len(_pending)))
