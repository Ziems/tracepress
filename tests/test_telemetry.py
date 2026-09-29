from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

import tracepress
from tracepress import telemetry
from tracepress.workspace import Run

from .conftest import ScriptedEngine, tool_call


@pytest.fixture
def fresh_id():
    previous = telemetry._cached_install_id
    telemetry._cached_install_id = None
    yield
    telemetry._cached_install_id = previous


def _collect(monkeypatch) -> list[dict]:
    monkeypatch.setenv("TRACEPRESS_TELEMETRY", "on")
    sent: list[dict] = []
    monkeypatch.setattr(telemetry, "_dispatch", lambda body: sent.append(body))
    return sent


def _shared() -> dict:
    return {
        "schema": "lm_op.v1",
        "app_version": "0.1.0",
        "install_source": "source",
        "python_version": "3.13.0",
        "os": "Darwin",
        "arch": "arm64",
        "ci": False,
        "$process_person_profile": False,
        "$geoip_disable": True,
        "op": "map",
        "status": "completed",
        "duration_seconds": 0.1,
        "in_workspace": False,
        "in_agent": False,
    }


def test_allowlist_strips_content():
    props = telemetry._event_properties(
        "lm_op",
        {**_shared(), "instruction": "secret instruction", "question": "secret question", "path": "/Users/secret/ws"},
    )
    assert props is not None
    assert props["op"] == "map"
    assert props["schema"] == "lm_op.v1"
    assert props["$process_person_profile"] is False
    assert props["$geoip_disable"] is True
    for leaked in ("instruction", "question", "path", "secret"):
        assert leaked not in props
        assert leaked not in json.dumps(props)


def test_unknown_event_is_dropped():
    assert telemetry._event_properties("prompt", {**_shared(), "instruction": "secret"}) is None


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", " OFF "])
def test_opt_out_does_not_build_or_send(value, tmp_path, monkeypatch, fresh_id):
    monkeypatch.setenv("TRACEPRESS_TELEMETRY", value)
    monkeypatch.setenv("TRACEPRESS_HOME", str(tmp_path))
    built: list = []
    monkeypatch.setattr(telemetry.LmOp, "model_validate", classmethod(lambda cls, *args, **kwargs: built.append(args)))
    monkeypatch.setattr(telemetry, "_dispatch", lambda body: built.append(body))
    telemetry.capture_lm_op("map", 0.1, None, {"instruction": "secret", "n_input_rows": 1, "n_output_rows": 1})
    assert built == []
    assert not (tmp_path / "telemetry_id").exists()


def test_unset_leaves_telemetry_on(monkeypatch):
    monkeypatch.delenv("TRACEPRESS_TELEMETRY", raising=False)
    assert telemetry._enabled() is True


def test_install_id_is_reused(tmp_path, monkeypatch, fresh_id):
    monkeypatch.setenv("TRACEPRESS_HOME", str(tmp_path))
    first = telemetry.install_id()
    telemetry._cached_install_id = None
    assert telemetry.install_id() == first
    assert (tmp_path / "telemetry_id").read_text() == first
    assert len(first) == 32


def test_install_id_keeps_the_other_process_value(tmp_path, monkeypatch, fresh_id):
    monkeypatch.setenv("TRACEPRESS_HOME", str(tmp_path))
    path = tmp_path / "telemetry_id"
    real_open = Path.open

    def open_side(self, mode="r", *args, **kwargs):
        if self == path and "x" in mode:
            path.write_text("from-other")
            raise FileExistsError
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_side)
    assert telemetry.install_id() == "from-other"


def test_install_id_stays_in_memory_when_write_fails(tmp_path, monkeypatch, fresh_id):
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("x")
    monkeypatch.setenv("TRACEPRESS_HOME", str(blocked / "child"))
    first = telemetry.install_id()
    assert telemetry.install_id() == first
    assert len(first) == 32
    assert not (blocked / "child").exists()


def test_lm_map_event_has_counts_and_no_content(make_lm, monkeypatch):
    events = _collect(monkeypatch)
    lm, _ = make_lm(lambda r: {"loud": "LOUD"})
    out = pd.DataFrame({"text": ["secret cell", "b"]}).lm_map("text -> loud", "secret instruction", lm=lm)
    assert list(out["loud"]) == ["LOUD", "LOUD"]
    assert len(events) == 1
    props = events[0]["properties"]
    assert events[0]["event"] == "lm_op"
    assert props["op"] == "map"
    assert props["status"] == "completed"
    assert props["model"] == "fake/model"
    assert props["provider"] == "fake"
    assert props["n_input_rows"] == 2
    assert props["n_output_rows"] == 2
    assert props["calls"] == 2
    assert props["in_workspace"] is False
    assert props["in_agent"] is False
    assert props["schema"] == "lm_op.v1"
    assert "exception_type" not in props
    blob = json.dumps(events)
    assert "secret instruction" not in blob
    assert "secret cell" not in blob


def test_failed_op_is_errored_and_still_raises(make_lm, monkeypatch):
    events = _collect(monkeypatch)
    lm, engine = make_lm(lambda r: "{}")
    with pytest.raises(KeyError, match="nope"):
        pd.DataFrame({"a": [1]}).lm_map("nope -> x", "secret instruction", lm=lm)
    assert engine.requests == []
    props = events[0]["properties"]
    assert props["op"] == "map"
    assert props["status"] == "errored"
    assert props["exception_type"] == "KeyError"
    assert "n_input_rows" not in props
    assert "secret instruction" not in json.dumps(events)
    assert "not columns" not in json.dumps(events)


def test_join_error_is_reported(monkeypatch):
    events = _collect(monkeypatch)
    with pytest.raises(NotImplementedError):
        pd.DataFrame({"a": [1]}).lm_join(pd.DataFrame({"a": [1]}), "a", "secret condition")
    props = events[0]["properties"]
    assert props["op"] == "join"
    assert props["status"] == "errored"
    assert props["exception_type"] == "NotImplementedError"
    assert "secret condition" not in json.dumps(events)


def test_interrupt_status(monkeypatch):
    events = _collect(monkeypatch)

    @telemetry.track_op("dedup")
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        boom()
    props = events[0]["properties"]
    assert props["status"] == "interrupted"
    assert props["exception_type"] == "KeyboardInterrupt"
    assert props["op"] == "dedup"


def test_workspace_context_omits_names(tmp_path, make_lm, monkeypatch):
    events = _collect(monkeypatch)
    lm, _ = make_lm(lambda r: {"loud": "A"})
    ws = tracepress.Workspace(tmp_path / "secret-workspace")

    @ws.pipeline(description="secret pipeline description")
    def named_secret_pipeline(rows):
        return rows.lm_map("text -> loud", "secret instruction", lm=lm, save_as="secret_artifact")

    named_secret_pipeline(pd.DataFrame({"text": ["cell-value"]}))
    props = events[0]["properties"]
    assert props["in_workspace"] is True
    assert props["in_agent"] is False
    assert props["run_kind"] == "pipeline"
    assert props["n_pipelines"] == 1
    assert props["n_artifacts"] == 2
    assert props["n_input_rows"] == 1
    blob = json.dumps(events)
    for leaked in (
        "named_secret_pipeline",
        "secret pipeline",
        "secret instruction",
        "secret_artifact",
        "secret-workspace",
        "cell-value",
    ):
        assert leaked not in blob

    events.clear()
    with Run(ws, "save-secret-name"):
        pd.DataFrame({"text": ["other-cell"]}).lm_map("text -> loud", "another instruction", lm=lm)
    assert events[0]["properties"]["run_kind"] == "save"
    assert "secret-name" not in json.dumps(events)
    assert "other-cell" not in json.dumps(events)


def test_agent_run_and_nested_op(tmp_path, make_lm, monkeypatch):
    events = _collect(monkeypatch)
    lm, _ = make_lm(lambda r: {"loud": "A"})
    tracepress.configure(lm=lm)
    ws = tracepress.Workspace(tmp_path / "ws")
    ws.save("rows", pd.DataFrame({"text": ["hidden cell"]}))
    agent = tracepress.Model(
        "fake/agent",
        engine=ScriptedEngine(
            [
                tool_call("list_artifacts"),
                tool_call("run_op", op="map", artifact="rows", signature="text -> loud", instruction="secret instruction"),
                "secret answer text",
            ]
        ),
        cache=False,
        progress=False,
    )
    answer = ws.ask("secret question text", model=agent, verbose=False)
    assert answer.text == "secret answer text"
    assert [event["event"] for event in events] == ["lm_op", "agent_run"]

    op = events[0]["properties"]
    assert op["op"] == "map"
    assert op["in_agent"] is True
    assert op["in_workspace"] is True
    assert op["run_kind"] == "ask"
    assert op["n_input_rows"] == 1

    run = events[1]["properties"]
    assert run["status"] == "completed"
    assert run["model"] == "fake/agent"
    assert run["provider"] == "fake"
    assert run["n_turns"] == 3
    assert run["n_tool_calls"] == 2
    assert run["n_list_artifacts"] == 1
    assert run["n_run_op"] == 1
    assert run["n_list_pipelines"] == 0
    assert run["n_python"] == 0
    assert run["hit_step_limit"] is False
    assert run["op_calls"] == 1
    assert run["n_pipelines"] == 0
    assert run["n_artifacts"] == 2
    assert run["schema"] == "agent_run.v1"
    blob = json.dumps(events)
    for leaked in ("secret question", "secret answer", "secret instruction", "hidden cell"):
        assert leaked not in blob

    events.clear()
    pd.DataFrame({"text": ["z"]}).lm_map("text -> loud", "after the ask", lm=lm)
    assert events[0]["properties"]["in_agent"] is False


def test_failed_ask_is_reported_and_reraised(tmp_path, monkeypatch):
    events = _collect(monkeypatch)

    class Boom:
        def complete(self, request):
            raise RuntimeError("secret boom message")

    ws = tracepress.Workspace(tmp_path / "ws")
    agent = tracepress.Model("fake/agent", engine=Boom(), cache=False, progress=False)
    with pytest.raises(Exception, match="secret boom message") as raised:
        ws.ask("secret question text", model=agent, verbose=False)
    assert type(raised.value).__name__ == "LMUnexpectedError"
    assert [event["event"] for event in events] == ["agent_run"]
    props = events[0]["properties"]
    assert props["status"] == "errored"
    assert props["exception_type"] == "LMUnexpectedError"
    assert props["n_turns"] == 1
    blob = json.dumps(events)
    assert "secret boom message" not in blob
    assert "secret question" not in blob


def test_context_reads_presence_not_values(monkeypatch):
    events = _collect(monkeypatch)
    monkeypatch.setenv("CI", "secret-ci-value")
    monkeypatch.setenv("CURSOR_AGENT", "secret-agent-value")
    monkeypatch.setenv("CLAUDECODE", "secret-claude-value")
    telemetry.capture_lm_op(
        "filter",
        0.2,
        None,
        {"model": "anthropic/claude", "n_input_rows": 4, "n_output_rows": 1, "calls": 4, "cached_calls": 1, "input_tokens": 8, "output_tokens": 2, "cost_usd": 0.01},
    )
    props = events[0]["properties"]
    assert props["ci"] is True
    assert props["invoked_by"] == "cursor"
    assert props["provider"] == "anthropic"
    assert props["cached_calls"] == 1
    blob = json.dumps(events)
    assert "secret-ci-value" not in blob
    assert "secret-agent-value" not in blob
    assert "secret-claude-value" not in blob


def test_running_loop_spawns_via_executor(monkeypatch):
    monkeypatch.setenv("TRACEPRESS_TELEMETRY", "on")
    spawned: list = []
    monkeypatch.setattr(telemetry, "_spawn", lambda body: spawned.append(body))

    async def go():
        loop = asyncio.get_running_loop()
        calls: list = []
        original = loop.run_in_executor

        def spy(executor, func, *args):
            calls.append(func)
            return original(executor, func, *args)

        monkeypatch.setattr(loop, "run_in_executor", spy)
        telemetry.capture_lm_op("topk", 0.0, None, {"n_input_rows": 1, "n_output_rows": 1})
        await asyncio.sleep(0.2)
        assert calls == [telemetry._spawn]
        assert len(spawned) == 1
        assert spawned[0]["event"] == "lm_op"

    asyncio.run(go())


def test_child_posts_the_event(monkeypatch):
    monkeypatch.setenv("TRACEPRESS_TELEMETRY", "on")
    received: list[tuple[str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            received.append((self.path, json.loads(self.rfile.read(length))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setattr(telemetry, "POSTHOG_HOST", f"http://{host}:{port}")
    try:
        telemetry.capture_lm_op(
            "agg",
            0.01,
            None,
            {"model": "openai/gpt-test", "n_input_rows": 2, "n_output_rows": 1, "calls": 1, "cached_calls": 0, "input_tokens": 3, "output_tokens": 1, "cost_usd": 0.0},
        )
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.05)
    finally:
        server.shutdown()
        server.server_close()
    assert received, "child did not POST"
    path, body = received[0]
    assert path == "/i/v0/e/"
    assert body["event"] == "lm_op"
    assert body["api_key"].startswith("phc_")
    assert body["distinct_id"]
    assert body["properties"]["op"] == "agg"
    assert body["properties"]["model"] == "openai/gpt-test"
    assert "instruction" not in body["properties"]
