from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import trace_miner.telemetry as telemetry_module
from trace_miner.cli import _option_strings
from trace_miner.home import home_dir
from trace_miner.telemetry import COMMAND_FINISHED, RUN_FINISHED, command_telemetry_from_argv


@pytest.fixture(autouse=True)
def reset_telemetry_state() -> Iterator[None]:
    telemetry_module._install_id = None
    telemetry_module.reset_command_run_ids()
    yield
    telemetry_module._install_id = None
    telemetry_module.reset_command_run_ids()


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACE_MINER_TELEMETRY", "1")
    monkeypatch.setattr(telemetry_module, "POSTHOG_PROJECT_API_KEY", "phc_test")


def _command_capture() -> None:
    telemetry_module.capture_command_finished(
        command="stats",
        command_path="stats",
        command_flags=[],
        bare=False,
        duration_seconds=0.1,
        exit_code=0,
        exception=None,
    )


def _run_capture(*, exception: BaseException | None = None) -> None:
    telemetry_module.capture_run_finished(
        run_id="abc",
        command="run",
        pipeline_count=1,
        lm_provider="openrouter",
        lm_model="demo-model",
        records_read=2,
        emitted=1,
        failed=0,
        lm_calls=3,
        lm_prompt_tokens=10,
        lm_completion_tokens=4,
        lm_cost_usd=1.25,
        duration_seconds=1.5,
        exception=exception,
    )


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", "OFF"])
def test_opt_out_skips_build_and_send(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TRACE_MINER_TELEMETRY", value)
    monkeypatch.setattr(telemetry_module, "POSTHOG_PROJECT_API_KEY", "phc_test")
    built: list[str] = []
    sent: list[dict[str, object]] = []

    def refuse_build(*args: object, **kwargs: object) -> None:
        built.append("built")
        raise RuntimeError("built")

    monkeypatch.setattr(telemetry_module, "CommandFinishedV1", refuse_build)
    monkeypatch.setattr(telemetry_module, "RunFinishedV1", refuse_build)
    monkeypatch.setattr(telemetry_module, "_spawn_sender", sent.append)
    _command_capture()
    _run_capture()
    assert built == []
    assert sent == []


def test_flag_parser_keeps_registered_names() -> None:
    options = {
        "run": {"--progress", "--no-progress", "-h", "--help"},
        "serve": {"--poll-interval", "--from-end", "--progress", "-h", "--help"},
    }
    command, path, flags, bare = command_telemetry_from_argv(
        ["serve", "pipe.yaml", "--poll-interval=2", "positional", "--from-end", "--", "--progress"],
        lambda name: options[name],
    )
    assert (command, path, bare) == ("serve", "serve", False)
    assert flags == ["--poll-interval", "--from-end"]

    command, path, flags, bare = command_telemetry_from_argv(
        ["pipeline.yaml", "--progress", "--bogus", "--", "--no-progress"],
        lambda name: options[name],
    )
    assert (command, path, bare) == ("run", "run", True)
    assert flags == ["--progress"]


def test_registered_cli_flags_drop_values() -> None:
    command, path, flags, bare = command_telemetry_from_argv(
        ["serve", "alerts.yaml", "--host=0.0.0.0", "--port", "8765", "--from-end", "--", "--progress"],
        _option_strings,
    )
    assert (command, path, bare) == ("serve", "serve", False)
    assert flags == ["--host", "--port", "--from-end"]


def test_allowlist_strips_unknown_properties() -> None:
    event = {
        "event": COMMAND_FINISHED,
        "properties": {
            "schema": "trace_miner.command_finished.v1",
            "command": "run",
            "prompt": "secret prompt",
            "completion": "secret completion",
            "$geoip_disable": True,
        },
    }
    result = telemetry_module._allowlist_before_send(event)
    assert result is not None
    assert result["properties"]["command"] == "run"
    assert result["properties"]["$geoip_disable"] is True
    assert "prompt" not in result["properties"]
    assert "completion" not in result["properties"]


def test_unknown_event_produces_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    spawned: list[dict[str, object]] = []
    monkeypatch.setattr(telemetry_module, "_spawn_sender", spawned.append)
    telemetry_module._capture("trace_miner.unknown", {"prompt": "secret prompt", "schema": "nope"})
    assert spawned == []


def test_install_id_is_stable(isolate_trace_miner_home: Path) -> None:
    first = telemetry_module._get_install_id()
    telemetry_module._install_id = None
    second = telemetry_module._get_install_id()
    assert first == second
    assert (home_dir() / "telemetry_id").read_text(encoding="utf-8").strip() == first
    assert isolate_trace_miner_home.is_dir()


def test_install_id_race_adopts_winner(monkeypatch: pytest.MonkeyPatch, isolate_trace_miner_home: Path) -> None:
    path = home_dir() / "telemetry_id"
    real_open = Path.open

    def opening(self: Path, mode: str = "r", *args: object, **kwargs: object):
        if self.resolve() == path.resolve() and "x" in str(mode):
            with real_open(self, "w", encoding="utf-8") as handle:
                handle.write("winner\n")
            raise FileExistsError
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    assert telemetry_module._get_install_id() == "winner"
    assert path.read_text(encoding="utf-8").strip() == "winner"


def test_capture_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)

    def explode(payload: dict[str, object]) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(telemetry_module, "_spawn_sender", explode)
    _command_capture()
    _run_capture(exception=RuntimeError("boom"))

    def explode_context() -> dict[str, object]:
        raise RuntimeError("context")

    monkeypatch.setattr(telemetry_module, "_shared_context", explode_context)
    _command_capture()


def test_sender_posts_allowlisted_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    received: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            received["path"] = self.path
            received["body"] = json.loads(self.rfile.read(length))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        monkeypatch.setattr(telemetry_module, "POSTHOG_HOST", f"http://{host}:{port}")
        _run_capture()
        deadline = time.monotonic() + 5
        while "body" not in received and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        server.shutdown()
        server.server_close()

    assert received["path"] == "/i/v0/e/"
    body = received["body"]
    assert isinstance(body, dict)
    assert body["api_key"] == "phc_test"
    assert body["event"] == RUN_FINISHED
    assert body["distinct_id"] == (home_dir() / "telemetry_id").read_text(encoding="utf-8").strip()
    properties = body["properties"]
    assert isinstance(properties, dict)
    assert properties["$geoip_disable"] is True
    assert properties["$process_person_profile"] is False
    assert properties["lm_calls"] == 3
    assert properties["lm_prompt_tokens"] == 10
    assert properties["lm_completion_tokens"] == 4
    assert properties["lm_cost_usd"] == 1.25
    assert set(properties) <= telemetry_module._EVENT_PROPERTY_KEYS[RUN_FINISHED]
    assert "prompt" not in properties
    assert "completion" not in properties
    assert "secret" not in json.dumps(body)
