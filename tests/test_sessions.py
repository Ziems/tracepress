import json

import pytest

import tracepress as tp
from tracepress.adapters.atif import IMAGE_PLACEHOLDER


def _write(path, lines, torn=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(line) for line in lines) + "\n"
    path.write_text(text + ('{"type": "assist' if torn else ""))


def _cc(kind, ts, content, **extra):
    base = {"type": kind, "timestamp": f"2026-01-01T00:00:{ts:02d}Z", "cwd": "/work/app", "sessionId": "s1",
            "gitBranch": "main", "version": "2.1.0", "entrypoint": "cli", "isSidechain": False, "uuid": f"u{ts}"}
    return {**base, **extra, "message": content}


def _assistant(ts, mid, blocks, **extra):
    usage = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 1}
    return _cc("assistant", ts, {"id": mid, "role": "assistant", "model": "claude-opus-5", "content": blocks,
                                 "usage": usage}, **extra)


def _result(ts, call_id, content, is_error=False, **extra):
    return _cc("user", ts, {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}]}, **extra)


@pytest.fixture
def claude_dir(tmp_path):
    projects = tmp_path / "projects"
    session = projects / "-work-app" / "s1.jsonl"
    _write(session, [
        {"type": "custom-title", "customTitle": "Fix the build", "sessionId": "s1"},
        _cc("user", 1, {"role": "user", "content": "Please fix the build."}),
        {"type": "attachment", "timestamp": "2026-01-01T00:00:01Z", "attachment": {"type": "environment"}},
        _assistant(2, "m1", [{"type": "thinking", "thinking": "Check the logs first.", "signature": "x"}]),
        _assistant(3, "m1", [{"type": "text", "text": "Running the build."}]),
        _assistant(4, "m1", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "make"}}]),
        _result(5, "t1", "error: missing header", is_error=True),
        _assistant(6, "m1", [{"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "a.png"}}]),
        _result(7, "t2", [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}]),
        _cc("user", 8, {"role": "user", "content": "<local-command-stdout>ok</local-command-stdout>"}, isMeta=True),
        _assistant(9, "m2", [{"type": "tool_use", "id": "t3", "name": "Agent", "input": {"prompt": "look"}}]),
        _result(12, "t3", [{"type": "text", "text": "Found it."}]),
        _assistant(13, "m3", [{"type": "text", "text": "Fixed."}]),
    ], torn=True)
    _write(projects / "-work-app" / "s1" / "subagents" / "agent-abc.jsonl", [
        _cc("user", 10, {"role": "user", "content": "look"}, isSidechain=True, agentId="abc"),
        _assistant(11, "m9", [{"type": "text", "text": "The header is missing."}], isSidechain=True, agentId="abc"),
    ])
    other = [_cc("user", 1, {"role": "user", "content": "Hi"}, cwd="/work/other", sessionId="s2"),
             _assistant(2, "m1", [{"type": "text", "text": "Hello."}], cwd="/work/other", sessionId="s2")]
    _write(projects / "-work-other" / "s2.jsonl", other)
    _write(projects / "-work-other" / "s3.jsonl", [{"type": "summary", "summary": "nothing here"}])
    return projects


def test_claude_code_sessions(claude_dir):
    sessions = tp.read.claude_code(claude_dir).table("sessions").collect()
    assert [r.key for r in sessions] == [
        (("project", "/work/app"), ("session", "s1")), (("project", "/work/other"), ("session", "s2"))]
    s = sessions[0]
    assert (s.title, s.prompt, s.model, s.git_branch) == ("Fix the build", "Please fix the build.", "claude-opus-5", "main")
    assert (s.n_steps, s.n_subagents, s.n_tool_errors) == (7, 1, 1)
    # Usage is repeated on each line of a message; m1, m2, m3 and the subagent's m9 aren't double counted.
    assert (s.input_tokens, s.output_tokens) == (30, 15)
    assert s.duration_s == 12.0


def test_claude_code_steps(claude_dir):
    src = tp.read.claude_code(claude_dir, project="app")
    steps = src.table("steps").collect()
    assert [(s.source, s.agent_id) for s in steps] == [
        ("user", None), ("agent", None), ("system", None), ("agent", None),
        ("user", "abc"), ("agent", "abc"), ("agent", None)]
    turn = steps[1]
    assert turn.reasoning == "Check the logs first." and turn.message == "Running the build."
    assert [(c["name"], c["result"], c["is_error"]) for c in turn.tool_calls] == [
        ("Bash", "error: missing header", True), ("Read", IMAGE_PLACEHOLDER, False)]
    assert steps[3].tool_calls[0]["result"] == "Found it."
    assert steps[4].is_sidechain and not steps[3].is_sidechain

    calls = src.table("tool_calls").collect()
    assert [(c.key[-2:], c.name) for c in calls] == [
        ((("step", 2), ("call", 0)), "Bash"), ((("step", 2), ("call", 1)), "Read"), ((("step", 4), ("call", 0)), "Agent")]


def test_claude_code_options(claude_dir):
    assert len(tp.read.claude_code(claude_dir / "-work-app" / "s1.jsonl").files) == 1
    steps = tp.read.claude_code(claude_dir, project="app", subagents=False).table("steps").collect()
    assert not any(s.is_sidechain for s in steps)
    assert len(tp.read.claude_code(claude_dir, limit=1).files) == 1
    with pytest.raises(FileNotFoundError):
        tp.read.claude_code(claude_dir, project="nope")


def _cx(ts, kind, payload):
    return {"timestamp": f"2026-01-01T00:00:{ts:02d}Z", "type": kind, "payload": payload}


def _item(ts, **payload):
    return _cx(ts, "response_item", payload)


@pytest.fixture
def codex_dir(tmp_path):
    root = tmp_path / "sessions" / "2026" / "01" / "01"
    _write(root / "rollout-2026-01-01T00-00-00-11111111-1111-1111-1111-111111111111.jsonl", [
        _cx(0, "session_meta", {"id": "c1", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/work/app",
                                "cli_version": "0.50.0", "originator": "codex_cli_rs",
                                "git": {"branch": "main", "commit_hash": "abc"}}),
        _cx(0, "turn_context", {"cwd": "/work/app", "model": "gpt-6-codex", "effort": "high"}),
        _item(1, type="message", role="user", content=[
            {"type": "input_text", "text": "<environment_context><cwd>/work/app</cwd></environment_context>"}]),
        _item(1, type="message", role="user", content=[{"type": "input_text", "text": "Run the tests."}]),
        _cx(1, "event_msg", {"type": "user_message", "message": "Run the tests."}),
        _item(2, type="reasoning", summary=[{"type": "summary_text", "text": "Run pytest."}], encrypted_content="x"),
        _item(2, type="function_call", name="shell", arguments=json.dumps({"command": ["pytest"]}), call_id="f1"),
        _item(3, type="function_call_output", call_id="f1", output="Exit code: 1\nOutput:\n1 failed"),
        _item(4, type="reasoning", summary=[{"type": "summary_text", "text": "Patch it."}]),
        _item(4, type="message", role="assistant", content=[{"type": "output_text", "text": "One test fails; patching."}]),
        _item(4, type="custom_tool_call", name="apply_patch", input="*** Begin Patch", call_id="f2"),
        _item(5, type="custom_tool_call_output", call_id="f2", output=json.dumps({"output": "Done", "metadata": {"exit_code": 0}})),
        _item(6, type="message", role="assistant", content=[{"type": "output_text", "text": "Fixed."}]),
        _cx(6, "event_msg", {"type": "token_count", "info": {"total_token_usage": {
            "input_tokens": 500, "cached_input_tokens": 400, "output_tokens": 50, "reasoning_output_tokens": 20}}}),
    ], torn=True)
    # The older format: a bare meta line and bare items.
    _write(root / "rollout-2025-06-01T00-00-00-22222222-2222-2222-2222-222222222222.jsonl", [
        {"id": "c0", "timestamp": "2025-06-01T00:00:00Z", "instructions": None},
        {"record_type": "state"},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<environment_context><cwd>/work/old</cwd></environment_context>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi."}]},
    ])
    return tmp_path / "sessions"


def test_codex_sessions(codex_dir):
    sessions = tp.read.codex(codex_dir).table("sessions").collect()
    assert [r.key for r in sessions] == [
        (("project", "/work/app"), ("session", "c1")), (("project", "/work/old"), ("session", "c0"))]
    s = sessions[0]
    assert (s.prompt, s.model, s.reasoning_effort, s.git_branch, s.cli_version) == (
        "Run the tests.", "gpt-6-codex", "high", "main", "0.50.0")
    assert (s.input_tokens, s.output_tokens, s.reasoning_tokens, s.n_steps, s.n_tool_errors) == (500, 50, 20, 5, 1)
    assert sessions[1].prompt == "Hello"


def test_codex_steps(codex_dir):
    src = tp.read.codex(codex_dir, project="/work/app")
    steps = src.table("steps").collect()
    assert [s.source for s in steps] == ["system", "user", "agent", "agent", "agent"]
    first, second, last = steps[2], steps[3], steps[4]
    assert first.reasoning == "Run pytest." and first.message is None and first.model_name == "gpt-6-codex"
    assert first.tool_calls == [{"id": "f1", "name": "shell", "arguments": {"command": ["pytest"]},
                                 "result": "Exit code: 1\nOutput:\n1 failed", "is_error": True}]
    assert second.message == "One test fails; patching." and second.reasoning == "Patch it."
    assert second.tool_calls[0]["arguments"] == {"input": "*** Begin Patch"}
    assert (second.tool_calls[0]["result"], second.tool_calls[0]["is_error"]) == ("Done", False)
    assert last.message == "Fixed." and last.tool_calls == []
    assert len(src.table("tool_calls").collect()) == 2


@pytest.mark.parametrize("reader,table", [("claude_code", "steps"), ("codex", "steps"), ("codex", "tool_calls")])
def test_session_replay_matches_batch(reader, table, claude_dir, codex_dir):
    src = getattr(tp.read, reader)(claude_dir if reader == "claude_code" else codex_dir)
    count = lambda t: [(r.key, r.values) for r in t.group_by("session").reduce(lambda rs: {"n": len(rs)}).collect()]
    batch = count(src.table(table))
    for seed in range(3):
        live = tp.replay(src, seed=seed, shuffle_within=4, duplicates=0.3)
        assert count(live.table(table)) == batch
