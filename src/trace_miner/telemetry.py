from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import platform
import subprocess
import sys
from contextvars import ContextVar
from pathlib import Path
from importlib.metadata import PackageNotFoundError, distribution, version
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from trace_miner.home import home_dir
from trace_miner.model import RunStats


logger = logging.getLogger(__name__)

TELEMETRY_ENV = "TRACE_MINER_TELEMETRY"
POSTHOG_PROJECT_API_KEY = "phc_oj47o0vexqRFGmRy1u5Bm2aSx0CvhiHDygSUU6bAuix"
POSTHOG_HOST = "https://us.i.posthog.com"

PIPELINE_INIT = "pipeline_init"
PIPELINE_RUN = "pipeline_run"
PIPELINE_SERVED = "pipeline_served"
PIPELINE_SERVE_START = "pipeline_serve_start"
PIPELINE_SERVING = "pipeline_serving"

# Cumulative serve snapshots. The process may never exit, so these are sent while it runs.
SERVE_SNAPSHOT_SECONDS = 300

_EXECUTION_EVENTS = {
    "run": PIPELINE_RUN,
    "serve": PIPELINE_SERVED,
}

_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}
_POSTHOG_CONTROL_KEYS = {"$process_person_profile", "$geoip_disable"}
_INVOKED_BY = (
    ("CURSOR_AGENT", "cursor"),
    ("CLAUDECODE", "claude-code"),
    ("CODEX_SANDBOX", "codex"),
    ("GEMINI_CLI", "gemini"),
)

# Labels a follow-runtime run. `serve` sets this to `serve` before it starts.
run_command_name: ContextVar[str] = ContextVar("trace_miner_run_command", default="serve")

_install_id: str | None = None
_command_run_ids: list[str] = []
_command_flags: list[str] = []


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(alias="schema")
    app_version: str
    install_source: str
    python_version: str
    os_name: str = Field(alias="os")
    arch: str
    ci: bool
    invoked_by: str | None = None

    def properties(self) -> dict[str, Any]:
        dumped = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        dumped["$process_person_profile"] = False
        dumped["$geoip_disable"] = True
        return dumped


class PipelineInitV1(TelemetryEvent):
    command_flags: list[str]
    status: str
    duration_seconds: float
    exit_code: int
    exception_type: str | None = None


class PipelineExecutionV1(TelemetryEvent):
    command_flags: list[str]
    status: str
    duration_seconds: float
    pipeline_count: int
    records_read: int
    emitted: int
    failed: int
    stage_count: int
    maps: int
    filters: int
    stages: int
    alerts: int
    expands: int
    reduces: int
    depth: int
    width: int
    lm_calls: int
    lm_prompt_tokens: int
    lm_completion_tokens: int
    lm_cost_usd: float
    run_id: str | None = None
    exception_type: str | None = None
    lm_provider: str | None = None
    lm_model: str | None = None
    exit_code: int | None = None
    snapshot: int | None = None


def _property_keys(model: type[TelemetryEvent]) -> set[str]:
    keys = {field.alias or name for name, field in model.model_fields.items()}
    return keys | _POSTHOG_CONTROL_KEYS


_EVENT_PROPERTY_KEYS = {
    PIPELINE_INIT: _property_keys(PipelineInitV1),
    PIPELINE_RUN: _property_keys(PipelineExecutionV1),
    PIPELINE_SERVED: _property_keys(PipelineExecutionV1),
    PIPELINE_SERVE_START: _property_keys(PipelineExecutionV1),
    PIPELINE_SERVING: _property_keys(PipelineExecutionV1),
}

_SENDER_SOURCE = """\
import json
import sys
import urllib.request

payload = json.load(sys.stdin)
request = urllib.request.Request(
    payload["url"],
    data=json.dumps(payload["body"]).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=10) as response:
    response.read()
"""


def telemetry_disabled() -> bool:
    return os.environ.get(TELEMETRY_ENV, "").strip().lower() in _DISABLED_VALUES


def _sending_enabled() -> bool:
    return not telemetry_disabled() and POSTHOG_PROJECT_API_KEY.startswith("phc_")


def reset_command_run_ids() -> None:
    _command_run_ids.clear()
    _command_flags.clear()


def set_command_flags(flags: list[str]) -> None:
    _command_flags.clear()
    _command_flags.extend(flags)


def record_command_run_id(run_id: str) -> None:
    if not _sending_enabled():
        return
    _command_run_ids.append(run_id)


def command_telemetry_from_argv(
    argv: list[str],
    options_for: Callable[[str], set[str]],
) -> tuple[str, str, list[str], bool]:
    """Return command, command path, registered flag names, and whether the config was bare."""
    commands = {"init", "run", "serve", "stats"}
    if not argv or argv[0] in {"-h", "--help"} or argv[0] not in commands:
        flags = [argv[0]] if argv and argv[0] in {"-h", "--help"} else []
        return "", "", flags, False
    command = argv[0]
    tokens = argv[1:]
    bare = False
    options = options_for(command)
    flags: list[str] = []
    for token in tokens:
        if token == "--":
            break
        name = token.split("=", 1)[0]
        if name in options:
            flags.append(name)
    return command, command, flags, bare


def capture_cli_finished(
    *,
    command: str,
    command_flags: list[str],
    duration_seconds: float,
    exit_code: int,
    exception: BaseException | None,
) -> None:
    """Capture a CLI invocation that did not start a pipeline runtime."""
    if not _sending_enabled():
        return
    if "-h" in command_flags or "--help" in command_flags:
        return
    try:
        if command == "init":
            status = _command_status(exit_code, exception)
            event = PipelineInitV1(
                schema_="pipeline_init.v1",
                command_flags=command_flags,
                status=status,
                duration_seconds=duration_seconds,
                exit_code=exit_code,
                exception_type=_exception_type(exception, status),
                **_shared_context(),
            )
            _capture(PIPELINE_INIT, event.properties())
            return
        if command in _EXECUTION_EVENTS and not _command_run_ids:
            _capture_execution(
                command=command,
                run_id=None,
                command_flags=list(command_flags),
                pipeline_count=0,
                lm_provider=None,
                lm_model=None,
                records_read=0,
                emitted=0,
                failed=0,
                depth=0,
                width=0,
                lm_calls=0,
                lm_prompt_tokens=0,
                lm_completion_tokens=0,
                lm_cost_usd=0.0,
                duration_seconds=duration_seconds,
                exit_code=exit_code,
                exception=exception,
                kinds=[],
            )
    except Exception:
        logger.debug("Unable to capture command telemetry", exc_info=True)


def capture_pipeline_finished(
    *,
    command: str,
    run_id: str,
    pipeline_count: int,
    lm_provider: str,
    lm_model: str,
    records_read: int,
    emitted: int,
    failed: int,
    lm_calls: int,
    lm_prompt_tokens: int,
    lm_completion_tokens: int,
    lm_cost_usd: float,
    duration_seconds: float,
    exception: BaseException | None,
    kinds: list[str],
    depth: int,
    width: int,
) -> None:
    if not _sending_enabled():
        return
    try:
        status = _run_status(exception)
        _capture_execution(
            command=command,
            run_id=run_id,
            command_flags=list(_command_flags),
            pipeline_count=pipeline_count,
            lm_provider=lm_provider,
            lm_model=lm_model,
            records_read=records_read,
            emitted=emitted,
            failed=failed,
            lm_calls=lm_calls,
            lm_prompt_tokens=lm_prompt_tokens,
            lm_completion_tokens=lm_completion_tokens,
            lm_cost_usd=lm_cost_usd,
            duration_seconds=duration_seconds,
            exit_code=_exit_code_for_status(status),
            exception=exception,
            kinds=kinds,
            depth=depth,
            width=width,
        )
    except Exception:
        logger.debug("Unable to capture pipeline telemetry", exc_info=True)


def capture_pipeline_finished_async(
    *,
    command: str,
    run_id: str,
    pipeline_count: int,
    lm_provider: str,
    lm_model: str,
    stats: RunStats,
    exception: BaseException | None,
    kinds: list[str],
    depth: int,
    width: int,
) -> None:
    """Schedule capture and return. Never raises."""
    if not _sending_enabled():
        return
    try:
        asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                capture_pipeline_finished,
                command=command,
                run_id=run_id,
                pipeline_count=pipeline_count,
                lm_provider=lm_provider,
                lm_model=lm_model,
                records_read=stats.records_read,
                emitted=stats.emitted,
                failed=stats.failed,
                lm_calls=stats.lm_calls,
                lm_prompt_tokens=stats.lm_prompt_tokens,
                lm_completion_tokens=stats.lm_completion_tokens,
                lm_cost_usd=stats.lm_cost_usd,
                duration_seconds=stats.elapsed_s,
                exception=exception,
                kinds=list(kinds),
                depth=depth,
                width=width,
            ),
        )
    except Exception:
        logger.debug("Unable to schedule telemetry", exc_info=True)


def capture_pipeline_serving(
    *,
    run_id: str,
    pipeline_count: int,
    lm_provider: str,
    lm_model: str,
    records_read: int,
    emitted: int,
    failed: int,
    lm_calls: int,
    lm_prompt_tokens: int,
    lm_completion_tokens: int,
    lm_cost_usd: float,
    duration_seconds: float,
    kinds: list[str],
    depth: int,
    width: int,
    snapshot: int,
) -> None:
    if not _sending_enabled():
        return
    try:
        _capture_execution(
            command="serve",
            event_name=PIPELINE_SERVE_START if snapshot == 1 else PIPELINE_SERVING,
            status="running",
            run_id=run_id,
            command_flags=list(_command_flags),
            pipeline_count=pipeline_count,
            lm_provider=lm_provider,
            lm_model=lm_model,
            records_read=records_read,
            emitted=emitted,
            failed=failed,
            lm_calls=lm_calls,
            lm_prompt_tokens=lm_prompt_tokens,
            lm_completion_tokens=lm_completion_tokens,
            lm_cost_usd=lm_cost_usd,
            duration_seconds=duration_seconds,
            exit_code=None,
            exception=None,
            kinds=kinds,
            depth=depth,
            width=width,
            snapshot=snapshot,
        )
    except Exception:
        logger.debug("Unable to capture serving telemetry", exc_info=True)


def capture_pipeline_serving_async(
    *,
    run_id: str,
    pipeline_count: int,
    lm_provider: str,
    lm_model: str,
    stats: RunStats,
    kinds: list[str],
    depth: int,
    width: int,
    snapshot: int,
) -> None:
    """Schedule a cumulative serve snapshot and return. Never raises."""
    if not _sending_enabled():
        return
    try:
        asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                capture_pipeline_serving,
                run_id=run_id,
                pipeline_count=pipeline_count,
                lm_provider=lm_provider,
                lm_model=lm_model,
                records_read=stats.records_read,
                emitted=stats.emitted,
                failed=stats.failed,
                lm_calls=stats.lm_calls,
                lm_prompt_tokens=stats.lm_prompt_tokens,
                lm_completion_tokens=stats.lm_completion_tokens,
                lm_cost_usd=stats.lm_cost_usd,
                duration_seconds=stats.elapsed_s,
                kinds=list(kinds),
                depth=depth,
                width=width,
                snapshot=snapshot,
            ),
        )
    except Exception:
        logger.debug("Unable to schedule serving telemetry", exc_info=True)


def _capture_execution(
    *,
    command: str,
    run_id: str | None,
    command_flags: list[str],
    pipeline_count: int,
    lm_provider: str | None,
    lm_model: str | None,
    records_read: int,
    emitted: int,
    failed: int,
    lm_calls: int,
    lm_prompt_tokens: int,
    lm_completion_tokens: int,
    lm_cost_usd: float,
    duration_seconds: float,
    exit_code: int | None,
    exception: BaseException | None,
    kinds: list[str],
    depth: int = 0,
    width: int = 0,
    event_name: str | None = None,
    status: str | None = None,
    snapshot: int | None = None,
) -> None:
    resolved_event = event_name or _EXECUTION_EVENTS.get(command)
    if resolved_event is None:
        return
    if status is None:
        status = _run_status(exception) if run_id is not None else _command_status(exit_code or 1, exception)
    event = PipelineExecutionV1(
        schema_=f"{resolved_event}.v1",
        command_flags=command_flags,
        status=status,
        duration_seconds=duration_seconds,
        exit_code=exit_code,
        snapshot=snapshot,
        exception_type=_exception_type(exception, status),
        run_id=run_id,
        pipeline_count=pipeline_count,
        lm_provider=lm_provider,
        lm_model=lm_model,
        records_read=records_read,
        emitted=emitted,
        failed=failed,
        stage_count=len(kinds),
        depth=depth,
        width=width,
        lm_calls=lm_calls,
        lm_prompt_tokens=lm_prompt_tokens,
        lm_completion_tokens=lm_completion_tokens,
        lm_cost_usd=lm_cost_usd,
        **_kind_counts(kinds),
        **_shared_context(),
    )
    _capture(resolved_event, event.properties())


_KIND_FIELDS = {
    "map": "maps",
    "filter": "filters",
    "stage": "stages",
    "alert": "alerts",
    "expand": "expands",
    "reduce": "reduces",
}


def _kind_counts(kinds: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in _KIND_FIELDS.values()}
    for kind in kinds:
        field = _KIND_FIELDS.get(kind)
        if field is not None:
            counts[field] += 1
    return counts


def _exit_code_for_status(status: str) -> int:
    if status == "completed":
        return 0
    if status == "interrupted":
        return 130
    return 1


def _command_status(exit_code: int, exception: BaseException | None) -> str:
    if isinstance(exception, KeyboardInterrupt) or exit_code == 130:
        return "interrupted"
    if exit_code == 0:
        return "completed"
    return "errored"


def _run_status(exception: BaseException | None) -> str:
    if exception is None:
        return "completed"
    if isinstance(exception, (KeyboardInterrupt, asyncio.CancelledError)):
        return "interrupted"
    return "errored"


def _exception_type(exception: BaseException | None, status: str) -> str | None:
    if exception is None or status == "completed":
        return None
    return type(exception).__name__


def _shared_context() -> dict[str, Any]:
    context: dict[str, Any] = {
        "app_version": _app_version(),
        "install_source": _install_source(),
        "python_version": platform.python_version(),
        "os_name": platform.system(),
        "arch": platform.machine(),
        "ci": "CI" in os.environ,
    }
    invoked_by = _invoked_by()
    if invoked_by is not None:
        context["invoked_by"] = invoked_by
    return context


def _app_version() -> str:
    try:
        return version("trace-miner")
    except PackageNotFoundError:
        return "0.0.0"


def _install_source() -> str:
    try:
        dist = distribution("trace-miner")
    except PackageNotFoundError:
        return "source"
    raw = dist.read_text("direct_url.json")
    if not raw:
        return "package"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "package"
    if not isinstance(payload, dict):
        return "package"
    if payload.get("vcs_info"):
        return "vcs"
    dir_info = payload.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable"):
        return "editable"
    url = payload.get("url")
    if isinstance(url, str) and url.startswith("file:"):
        return "source"
    return "package"


def _invoked_by() -> str | None:
    for name, label in _INVOKED_BY:
        if name in os.environ:
            return label
    return None


def _capture(event_name: str, properties: dict[str, Any]) -> None:
    try:
        if not _sending_enabled():
            return
        event = _allowlist_before_send({"event": event_name, "properties": properties})
        if event is None:
            return
        _spawn_sender(
            {
                "url": f"{POSTHOG_HOST.rstrip('/')}/i/v0/e/",
                "body": {
                    "api_key": POSTHOG_PROJECT_API_KEY,
                    "event": event_name,
                    "distinct_id": _get_install_id(),
                    "properties": event["properties"],
                },
            }
        )
    except Exception:
        logger.debug("Unable to send telemetry event", exc_info=True)


def _allowlist_before_send(event: dict[str, Any]) -> dict[str, Any] | None:
    property_keys = _EVENT_PROPERTY_KEYS.get(event.get("event"))
    if property_keys is None:
        return None
    properties = event.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    event["properties"] = {key: value for key, value in properties.items() if key in property_keys}
    return event


def _spawn_sender(payload: dict[str, Any]) -> None:
    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen([sys.executable, "-c", _SENDER_SOURCE], **popen_kwargs)
    if process.stdin is None:
        return
    process.stdin.write(json.dumps(payload).encode())
    process.stdin.close()


def _get_install_id() -> str:
    global _install_id
    if _install_id:
        return _install_id
    _install_id = _load_install_id(home_dir() / "telemetry_id")
    return _install_id


def _load_install_id(path: Path) -> str:
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass

    install_id = uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            file.write(f"{install_id}\n")
    except FileExistsError:
        try:
            existing = path.read_text(encoding="utf-8").strip()
            if existing:
                install_id = existing
        except OSError:
            pass
    except OSError:
        logger.debug("Unable to persist telemetry install id", exc_info=True)
    return install_id
