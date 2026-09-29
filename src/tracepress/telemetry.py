"""Anonymous usage stats for operator calls and agent runs.

On by default. Opt out with ``TRACEPRESS_TELEMETRY=off`` (also ``0``, ``false``, ``no``,
``disabled``). Nothing here includes prompts, completions, paths, row contents, questions,
or exception messages. A send failure does not change the call that triggered it.

Events are Pydantic models. A property that is not on the model is removed before send.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

POSTHOG_PROJECT_API_KEY = "phc_oj47o0vexqRFGmRy1u5Bm2aSx0CvhiHDygSUU6bAuix"
POSTHOG_HOST = "https://us.i.posthog.com"

_OFF = frozenset({"0", "false", "no", "off", "disabled"})
_INVOKED_BY = (
    ("CURSOR_AGENT", "cursor"),
    ("CLAUDECODE", "claude-code"),
    ("CODEX_SANDBOX", "codex"),
    ("GEMINI_CLI", "gemini"),
)
_AGENT_TOOLS = (
    "list_pipelines",
    "list_artifacts",
    "describe_artifact",
    "read_rows",
    "python",
    "run_op",
    "run_pipeline",
)

_cached_install_id: str | None = None
_tracking: ContextVar[bool] = ContextVar("tracepress_telemetry_tracking", default=False)
_note: ContextVar[dict[str, Any] | None] = ContextVar("tracepress_telemetry_note", default=None)
in_agent: ContextVar[bool] = ContextVar("tracepress_telemetry_in_agent", default=False)


# --------------------------------------------------------------------- events


class _Event(BaseModel):
    """Properties shared by every event. The fields are the allowlist."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_: str = Field(alias="schema")
    app_version: str
    install_source: Literal["package", "editable", "source", "vcs"]
    python_version: str
    os: str
    arch: str
    ci: bool
    invoked_by: Literal["cursor", "claude-code", "codex", "gemini"] | None = None
    process_person_profile: Literal[False] = Field(False, alias="$process_person_profile")
    geoip_disable: Literal[True] = Field(True, alias="$geoip_disable")


class LmOp(_Event):
    op: Literal["map", "filter", "agg", "topk", "dedup", "join"]
    status: Literal["completed", "errored", "interrupted"]
    duration_seconds: float
    exception_type: str | None = None
    provider: str | None = None
    model: str | None = None
    calls: int | None = None
    cached_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    n_input_rows: int | None = None
    n_output_rows: int | None = None
    in_workspace: bool
    in_agent: bool
    run_kind: Literal["pipeline", "agent_run", "save"] | None = None
    n_artifacts: int | None = None
    n_pipelines: int | None = None


class AgentRun(_Event):
    status: Literal["completed", "errored", "interrupted"]
    duration_seconds: float
    exception_type: str | None = None
    provider: str | None = None
    model: str | None = None
    calls: int | None = None
    cached_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    op_calls: int | None = None
    n_turns: int | None = None
    n_tool_calls: int | None = None
    n_list_pipelines: int | None = None
    n_list_artifacts: int | None = None
    n_describe_artifact: int | None = None
    n_read_rows: int | None = None
    n_python: int | None = None
    n_run_op: int | None = None
    n_run_pipeline: int | None = None
    hit_step_limit: bool | None = None
    n_artifacts: int | None = None
    n_pipelines: int | None = None


_EVENTS: dict[str, type[_Event]] = {"lm_op": LmOp, "agent_run": AgentRun}


def _allowlist(model: type[BaseModel]) -> frozenset[str]:
    return frozenset(field.alias or name for name, field in model.model_fields.items())


def _event_properties(event: str, props: dict[str, Any]) -> dict[str, Any] | None:
    """The allowlisted properties for ``event``, or None if the event is not registered."""
    model = _EVENTS.get(event)
    if model is None:
        return None
    allowed = _allowlist(model)
    clean = {key: value for key, value in props.items() if key in allowed and value is not None}
    return model.model_validate(clean).model_dump(by_alias=True, exclude_none=True)


# ----------------------------------------------------------------------- send


def _enabled() -> bool:
    raw = os.environ.get("TRACEPRESS_TELEMETRY", "")
    if raw.strip().lower() in _OFF:
        return False
    return POSTHOG_PROJECT_API_KEY.startswith("phc_")


def _home() -> Path:
    override = os.environ.get("TRACEPRESS_HOME")
    if override and override.strip():
        return Path(override).expanduser()
    return Path.home() / ".tracepress"


def install_id() -> str:
    """A random id for this install, stored beside other local state. Not derived from the user."""
    global _cached_install_id
    if _cached_install_id:
        return _cached_install_id
    path = _home() / "telemetry_id"
    try:
        existing = path.read_text().strip()
    except OSError:
        existing = ""
    if existing:
        _cached_install_id = existing
        return existing
    new = uuid.uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as fh:
            fh.write(new)
    except FileExistsError:
        try:
            existing = path.read_text().strip()
        except OSError:
            existing = ""
        _cached_install_id = existing or new
        return _cached_install_id
    except OSError:
        log.debug("could not store telemetry install id", exc_info=True)
        _cached_install_id = new
        return new
    _cached_install_id = new
    return new


def _app_version() -> str:
    try:
        return importlib.metadata.version("tracepress")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _install_source() -> str:
    try:
        dist = importlib.metadata.distribution("tracepress")
    except importlib.metadata.PackageNotFoundError:
        return "source"
    raw = dist.read_text("direct_url.json")
    if not raw:
        return "package"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "package"
    if data.get("vcs_info"):
        return "vcs"
    if (data.get("dir_info") or {}).get("editable"):
        return "editable"
    if str(data.get("url") or "").startswith("file:"):
        return "source"
    return "package"


def _invoked_by() -> str | None:
    for key, label in _INVOKED_BY:
        if key in os.environ:
            return label
    return None


def _context(event: str) -> dict[str, Any]:
    props: dict[str, Any] = {
        "schema": f"{event}.v1",
        "app_version": _app_version(),
        "install_source": _install_source(),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "ci": "CI" in os.environ,
        "$process_person_profile": False,
        "$geoip_disable": True,
    }
    who = _invoked_by()
    if who is not None:
        props["invoked_by"] = who
    return props


def _status(exc: BaseException | None) -> str:
    if exc is None:
        return "completed"
    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
        return "interrupted"
    if isinstance(exc, SystemExit) and exc.code == 130:
        return "interrupted"
    return "errored"


def _provider(model: str | None) -> str | None:
    if not model:
        return None
    if ":" in model:
        return model.split(":", 1)[0]
    if "/" in model:
        return model.split("/", 1)[0]
    return None


def _model_props(model: str | None) -> dict[str, Any]:
    if not model:
        return {}
    props: dict[str, Any] = {"model": model}
    provider = _provider(model)
    if provider:
        props["provider"] = provider
    return props


def _usage_props(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "calls": usage.calls,
        "cached_calls": usage.cached_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(usage.cost_usd, 6),
    }


_CHILD = """
import json, sys, urllib.request
url = sys.argv[1]
body = sys.stdin.buffer.read()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=10).read()
"""


def _spawn(body: dict[str, Any]) -> None:
    try:
        url = POSTHOG_HOST.rstrip("/") + "/i/v0/e/"
        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen([sys.executable, "-c", _CHILD, url], **kwargs)
        if proc.stdin is not None:
            proc.stdin.write(json.dumps(body).encode())
            proc.stdin.close()
    except Exception:
        log.debug("telemetry send failed", exc_info=True)


def _dispatch(body: dict[str, Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _spawn(body)
        return
    loop.run_in_executor(None, _spawn, body)


def _emit(event: str, props: dict[str, Any]) -> None:
    properties = _event_properties(event, props)
    if properties is None:
        return
    _dispatch(
        {
            "api_key": POSTHOG_PROJECT_API_KEY,
            "event": event,
            "distinct_id": install_id(),
            "properties": properties,
        }
    )


def _workspace_props() -> dict[str, Any]:
    from .run import current_run

    props: dict[str, Any] = {"in_workspace": False, "in_agent": in_agent.get()}
    run = current_run()
    if run is None:
        return props
    label = str(run.manifest.get("pipeline") or "")
    if label == "agent_run":
        kind = "agent_run"
    elif label.startswith("save-"):
        kind = "save"
    else:
        kind = "pipeline"
    props.update(
        {
            "in_workspace": True,
            "run_kind": kind,
            "n_artifacts": len(run.workspace.artifact_records()),
            "n_pipelines": len(run.workspace.pipelines),
        }
    )
    return props


def capture_lm_op(op: str, duration: float, exc: BaseException | None, note: dict[str, Any] | None) -> None:
    """Send one ``lm_op`` event. Never raises."""
    try:
        if not _enabled():
            return
        props: dict[str, Any] = {
            **_context("lm_op"),
            "op": op,
            "status": _status(exc),
            "duration_seconds": round(duration, 3),
            **_workspace_props(),
        }
        if exc is not None:
            props["exception_type"] = type(exc).__name__
        if note:
            props.update(note)
        if props.get("model") and not props.get("provider"):
            provider = _provider(str(props["model"]))
            if provider:
                props["provider"] = provider
        _emit("lm_op", props)
    except Exception:
        log.debug("telemetry capture failed", exc_info=True)


def capture_agent_run(
    ws: Any,
    duration: float,
    exc: BaseException | None,
    *,
    model: str | None,
    usage: Any,
    op_calls: int,
    n_turns: int,
    steps: list[dict[str, Any]] | None,
    hit_step_limit: bool,
) -> None:
    """Send one ``agent_run`` event. Never raises."""
    try:
        if not _enabled():
            return
        counts = {name: 0 for name in _AGENT_TOOLS}
        n_tool_calls = 0
        for step in steps or []:
            if step.get("type") != "tool":
                continue
            n_tool_calls += 1
            name = step.get("name")
            if isinstance(name, str) and name in counts:
                counts[name] += 1
        props: dict[str, Any] = {
            **_context("agent_run"),
            "status": _status(exc),
            "duration_seconds": round(duration, 3),
            **_model_props(model),
            **_usage_props(usage),
            "op_calls": op_calls,
            "n_turns": n_turns,
            "n_tool_calls": n_tool_calls,
            "hit_step_limit": hit_step_limit,
            "n_artifacts": len(ws.artifact_records()),
            "n_pipelines": len(ws.pipelines),
            **{f"n_{name}": count for name, count in counts.items()},
        }
        if exc is not None:
            props["exception_type"] = type(exc).__name__
        _emit("agent_run", props)
    except Exception:
        log.debug("telemetry capture failed", exc_info=True)


# --------------------------------------------------------------------- hooks


def note_lm_op(*, model: str | None, usage: Any, n_input_rows: int, n_output_rows: int) -> None:
    """Remember model, usage, and row counts for the operator wrapper. No content."""
    if not _tracking.get() or not _enabled():
        return
    noted: dict[str, Any] = {"n_input_rows": n_input_rows, "n_output_rows": n_output_rows}
    noted.update(_model_props(model))
    noted.update(_usage_props(usage))
    _note.set(noted)


def track_op(op: str) -> Callable[[Callable], Callable]:
    """Time one operator and send ``lm_op`` when it returns or raises."""

    def decorate(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracking = _tracking.set(True)
            note = _note.set(None)
            started = time.monotonic()
            exc: BaseException | None = None
            try:
                return fn(*args, **kwargs)
            except BaseException as caught:
                exc = caught
                raise
            finally:
                try:
                    capture_lm_op(op, time.monotonic() - started, exc, _note.get())
                finally:
                    _note.reset(note)
                    _tracking.reset(tracking)

        return wrapper

    return decorate
