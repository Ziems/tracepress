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

COMMAND_FINISHED = "trace_miner.command_finished"
RUN_FINISHED = "trace_miner.run_finished"

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


class CommandFinishedV1(TelemetryEvent):
    command: str
    command_path: str
    command_flags: list[str]
    bare: bool
    status: str
    duration_seconds: float
    exit_code: int
    exception_type: str | None = None
    run_id: str | None = None


class RunFinishedV1(TelemetryEvent):
    run_id: str
    command: str
    status: str
    duration_seconds: float
    exception_type: str | None = None
    pipeline_count: int
    lm_provider: str
    lm_model: str
    records_read: int
    emitted: int
    failed: int
    lm_calls: int
    lm_prompt_tokens: int
    lm_completion_tokens: int
    lm_cost_usd: float


def _property_keys(model: type[TelemetryEvent]) -> set[str]:
    keys = {field.alias or name for name, field in model.model_fields.items()}
    return keys | _POSTHOG_CONTROL_KEYS


_EVENT_PROPERTY_KEYS = {
    COMMAND_FINISHED: _property_keys(CommandFinishedV1),
    RUN_FINISHED: _property_keys(RunFinishedV1),
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


def capture_command_finished(
    *,
    command: str,
    command_path: str,
    command_flags: list[str],
    bare: bool,
    duration_seconds: float,
    exit_code: int,
    exception: BaseException | None,
) -> None:
    if not _sending_enabled():
        return
    try:
        status = _command_status(exit_code, exception)
        event = CommandFinishedV1(
            schema_="trace_miner.command_finished.v1",
            command=command,
            command_path=command_path,
            command_flags=command_flags,
            bare=bare,
            status=status,
            duration_seconds=duration_seconds,
            exit_code=exit_code,
            exception_type=_exception_type(exception, status),
            run_id=_command_run_ids[-1] if _command_run_ids else None,
            **_shared_context(),
        )
        _capture(COMMAND_FINISHED, event.properties())
    except Exception:
        logger.debug("Unable to capture command telemetry", exc_info=True)


def capture_run_finished(
    *,
    run_id: str,
    command: str,
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
) -> None:
    if not _sending_enabled():
        return
    try:
        status = _run_status(exception)
        event = RunFinishedV1(
            schema_="trace_miner.run_finished.v1",
            run_id=run_id,
            command=command,
            status=status,
            duration_seconds=duration_seconds,
            exception_type=_exception_type(exception, status),
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
            **_shared_context(),
        )
        _capture(RUN_FINISHED, event.properties())
    except Exception:
        logger.debug("Unable to capture run telemetry", exc_info=True)


def capture_run_finished_async(
    *,
    run_id: str,
    command: str,
    pipeline_count: int,
    lm_provider: str,
    lm_model: str,
    stats: RunStats,
    exception: BaseException | None,
) -> None:
    """Schedule capture and return. Never raises."""
    if not _sending_enabled():
        return
    try:
        asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(
                capture_run_finished,
                run_id=run_id,
                command=command,
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
            ),
        )
    except Exception:
        logger.debug("Unable to schedule telemetry", exc_info=True)


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
