import asyncio
import gzip
import json
import struct

import pytest

import tracepress as tp
from tracepress.adapters.otlp import decode_traces_request

T0 = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z in ns
SEC = 1_000_000_000


def _value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, list):
        return {"arrayValue": {"values": [_value(x) for x in v]}}
    return {"stringValue": v}


def span(trace, sid, name, start, end, parent=None, error=None, **attrs):
    return {"traceId": trace, "spanId": sid, "parentSpanId": parent or "", "name": name, "kind": 1,
            "startTimeUnixNano": str(T0 + start * SEC), "endTimeUnixNano": str(T0 + end * SEC),
            "attributes": [{"key": k.replace("__", "."), "value": _value(v)} for k, v in attrs.items()],
            "status": {"code": 2, "message": error} if error else {}}


def request(spans, service="my-agent"):
    return {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": _value(service)}]},
                               "scopeSpans": [{"scope": {"name": "test"}, "spans": spans}]}]}


def semconv_trace(trace="a" * 32):
    """GenAI semantic conventions: an agent span, two chat calls and a tool call."""
    m = lambda *ms: json.dumps(list(ms))
    return [
        span(trace, "0000000000000001", "invoke_agent helper", 0, 10, **{
            "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "helper"}),
        span(trace, "0000000000000002", "chat gpt-6", 1, 3, parent="0000000000000001", **{
            "gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-6",
            "gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 20,
            "gen_ai.input.messages": m({"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
                                       {"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]}),
            "gen_ai.output.messages": m({"role": "assistant", "parts": [
                {"type": "reasoning", "content": "Need the weather tool."},
                {"type": "tool_call", "id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}]})}),
        span(trace, "0000000000000003", "execute_tool get_weather", 4, 5, parent="0000000000000001", **{
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_weather",
            "gen_ai.tool.call.id": "call_1", "gen_ai.tool.call.result": "18C, sunny"}),
        span(trace, "0000000000000004", "chat gpt-6", 6, 9, parent="0000000000000001", **{
            "gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-6",
            "gen_ai.usage.input_tokens": 150, "gen_ai.usage.output_tokens": 10,
            "gen_ai.input.messages": m(
                {"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]},
                {"role": "assistant", "parts": [{"type": "tool_call", "id": "call_1", "name": "get_weather"}]},
                {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call_1", "response": "18C, sunny"}]}),
            "gen_ai.output.messages": m({"role": "assistant", "parts": [{"type": "text", "content": "18C and sunny."}]})}),
    ]


def openinference_trace(trace="b" * 32):
    """OpenInference: flattened llm.input_messages.N.* attributes, a failing tool."""
    return [
        span(trace, "00000000000000a1", "Agent", 0, 10, **{
            "openinference.span.kind": "AGENT", "input.value": "Summarize x.txt", "output.value": "Could not read it."}),
        span(trace, "00000000000000a2", "ChatCompletion", 1, 2, parent="00000000000000a1", **{
            "openinference.span.kind": "LLM", "llm.model_name": "claude-opus-5",
            "llm.token_count.prompt": 50, "llm.token_count.completion": 5,
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Summarize x.txt",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "t1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "read_file",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"path": "x.txt"}'}),
        span(trace, "00000000000000a3", "read_file", 3, 4, parent="00000000000000a1", error="FileNotFoundError", **{
            "openinference.span.kind": "TOOL", "tool.name": "read_file", "input.value": '{"path": "x.txt"}'}),
        # No LLM call asked for this tool, so it's a step of its own.
        span(trace, "00000000000000a4", "log", 5, 6, parent="00000000000000a1", **{
            "openinference.span.kind": "TOOL", "tool.name": "log", "output.value": "logged"}),
        span(trace, "00000000000000a5", "ChatCompletion", 7, 8, parent="00000000000000a1", **{
            "openinference.span.kind": "LLM", "llm.model_name": "claude-opus-5",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Summarize x.txt",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Could not read it."}),
    ]


def openllmetry_trace(trace="c" * 32):
    return [span(trace, "00000000000000c1", "openai.chat", 0, 1, **{
        "llm.request.type": "chat", "gen_ai.request.model": "gpt-6",
        "gen_ai.prompt.0.role": "user", "gen_ai.prompt.0.content": "Hi",
        "gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "Hello!"})]


@pytest.fixture
def otel_file(tmp_path):
    path = tmp_path / "traces.jsonl"
    sem, oi = semconv_trace(), openinference_trace()
    # Spans of one trace split over export batches, one span exported twice, and a torn last line.
    lines = [request(sem[1:3]), request(oi), request(sem[2:] + sem[:1]), request(openllmetry_trace(), "other")]
    path.write_text("\n".join(json.dumps(x) for x in lines) + '\n{"resourceSp')
    return path


def test_traces_table(otel_file):
    rows = tp.read.otel(otel_file).table("traces").collect()
    by_id = {r.trace: r for r in rows}
    sem, oi = by_id["a" * 32], by_id["b" * 32]
    assert (sem.name, sem.service_name, sem.model, sem.status) == ("invoke_agent helper", "my-agent", "gpt-6", "ok")
    assert (sem.n_spans, sem.n_llm_calls, sem.n_tool_calls, sem.input_tokens, sem.output_tokens) == (4, 2, 1, 250, 30)
    assert sem.duration_s == 10.0 and sem.started_at == "2026-01-01T00:00:00Z"
    assert (oi.status, oi.error, oi.n_errors, oi.input, oi.output) == (
        "error", "FileNotFoundError", 1, "Summarize x.txt", "Could not read it.")
    assert by_id["c" * 32].service_name == "other"


def test_spans_table(otel_file):
    spans = [s for s in tp.read.otel(otel_file).table("spans").collect() if s.trace == "a" * 32]
    assert [(s.span, s.depth, s.span_type) for s in spans] == [(0, 0, "agent"), (1, 1, "llm"), (2, 1, "tool"), (3, 1, "llm")]
    assert spans[1].parent_span_id == "0000000000000001" and spans[1].duration_ms == 2000
    assert spans[1].attributes["gen_ai.request.model"] == "gpt-6"


def test_semconv_steps(otel_file):
    src = tp.read.otel(otel_file)
    steps = [s for s in src.table("steps").collect() if s.trace == "a" * 32]
    assert [s.source for s in steps] == ["user", "agent", "agent"]
    assert steps[0].message == "Weather in Paris?"  # the system prompt isn't a step
    ask = steps[1]
    assert (ask.reasoning, ask.model_name, ask.agent_name, ask.service_name) == (
        "Need the weather tool.", "gpt-6", "helper", "my-agent")
    assert ask.tool_calls == [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"},
                               "result": "18C, sunny", "is_error": False}]
    assert steps[2].message == "18C and sunny." and steps[2].metrics == {"input_tokens": 150, "output_tokens": 10}


def test_openinference_steps(otel_file):
    steps = [s for s in tp.read.otel(otel_file).table("steps").collect() if s.trace == "b" * 32]
    assert [s.source for s in steps] == ["user", "agent", "agent", "agent"]
    call = steps[1].tool_calls[0]  # the tool span names no call id, so it's matched by name
    assert (call["name"], call["arguments"], call["result"], call["is_error"]) == (
        "read_file", {"path": "x.txt"}, "FileNotFoundError", True)
    assert steps[2].tool_calls[0]["name"] == "log" and steps[2].tool_calls[0]["result"] == "logged"
    assert steps[3].message == "Could not read it."


def test_openllmetry_and_tool_calls(otel_file):
    src = tp.read.otel(otel_file)
    steps = [s for s in src.table("steps").collect() if s.trace == "c" * 32]
    assert [(s.source, s.message) for s in steps] == [("user", "Hi"), ("agent", "Hello!")]
    calls = src.table("tool_calls").collect()
    assert [(c.trace[0], c.name) for c in calls] == [("a", "get_weather"), ("b", "read_file"), ("b", "log")]


def test_console_exporter_format(tmp_path):
    doc = {"name": "chat", "context": {"trace_id": "0x" + "d" * 32, "span_id": "0x" + "1" * 16},
           "kind": "SpanKind.CLIENT", "parent_id": None, "start_time": "2026-01-01T00:00:00.000000Z",
           "end_time": "2026-01-01T00:00:01.000000Z", "status": {"status_code": "UNSET"},
           "attributes": {"gen_ai.operation.name": "chat", "gen_ai.output.messages": json.dumps(
               [{"role": "assistant", "parts": [{"type": "text", "content": "Hi there"}]}])},
           "events": [], "resource": {"attributes": {"service.name": "console"}}}
    (tmp_path / "console.json").write_text(json.dumps(doc, indent=4) + "\n" + json.dumps(doc, indent=4))
    src = tp.read.otel(tmp_path)
    spans = src.table("spans").collect()
    assert len(spans) == 1 and spans[0].kind == "client" and spans[0].service_name == "console"
    assert [s.message for s in src.table("steps").collect()] == ["Hi there"]


@pytest.mark.parametrize("table", ["steps", "tool_calls", "spans"])
def test_otel_replay_matches_batch(otel_file, table):
    src = tp.read.otel(otel_file)
    count = lambda t: [(r.key, r.values) for r in t.group_by("trace").reduce(lambda rs: {"n": len(rs)}).collect()]
    batch = count(src.table(table))
    for seed in range(3):
        live = tp.replay(src, seed=seed, shuffle_within=4, duplicates=0.3)
        assert count(live.table(table)) == batch


def test_missing_otel_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        tp.read.otel(tmp_path / "nothing*.json")


# ---------------------------------------------------------------------------- protobuf and the live receiver


def _varint(n):
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field, payload):
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _fixed64(field, n):
    return _varint(field << 3 | 1) + struct.pack("<Q", n)


def _pb_value(v):
    if isinstance(v, bool):
        return _varint(2 << 3) + _varint(int(v))
    if isinstance(v, int):
        return _varint(3 << 3) + _varint(v & (1 << 64) - 1)
    if isinstance(v, float):
        return _fixed64(4, struct.unpack("<Q", struct.pack("<d", v))[0])
    return _ld(1, v.encode())


def _pb_kv(k, v):
    return _ld(1, k.encode()) + _ld(2, _pb_value(v))


def _pb_span(s):
    out = _ld(1, bytes.fromhex(s["traceId"])) + _ld(2, bytes.fromhex(s["spanId"]))
    if s["parentSpanId"]:
        out += _ld(4, bytes.fromhex(s["parentSpanId"]))
    out += _ld(5, s["name"].encode()) + _varint(6 << 3) + _varint(s["kind"])
    out += _fixed64(7, int(s["startTimeUnixNano"])) + _fixed64(8, int(s["endTimeUnixNano"]))
    for kv in s["attributes"]:
        value = next(iter(kv["value"].values()))
        value = int(value) if "intValue" in kv["value"] else value
        out += _ld(9, _pb_kv(kv["key"], value))
    if s["status"]:
        out += _ld(15, _ld(2, s["status"]["message"].encode()) + _varint(3 << 3) + _varint(s["status"]["code"]))
    return out


def pb_request(spans, service="my-agent"):
    resource = _ld(1, _pb_kv("service.name", service))
    scope = b"".join(_ld(2, _pb_span(s)) for s in spans)
    return _ld(1, _ld(1, resource) + _ld(2, scope))


def test_protobuf_decodes_like_json():
    spans = semconv_trace() + openinference_trace()
    decoded = decode_traces_request(pb_request(spans) + _ld(1, b""))  # plus an empty ResourceSpans
    from tracepress.adapters.otel import spans_from_otlp
    as_json = list(spans_from_otlp(request(spans)))
    assert list(spans_from_otlp(decoded)) == as_json
    assert any(s.attributes.get("gen_ai.usage.input_tokens") == 100 for s in as_json)


async def _post(port, body, content_type, encoding=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    headers = f"POST /v1/traces HTTP/1.1\r\nHost: x\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\n"
    if encoding:
        headers += f"Content-Encoding: {encoding}\r\n"
    writer.write((headers + "Connection: close\r\n\r\n").encode() + body)
    await writer.drain()
    status = (await reader.readline()).split()[1]
    writer.close()
    return int(status)


def test_live_otlp_receiver():
    asyncio.run(_live_otlp_receiver())


async def _live_otlp_receiver():
    live = tp.watch.otlp(port=0, settle=0.1)
    steps = live.table("steps")
    traces = live.table("traces")
    seen, done = [], asyncio.Event()

    async def watch(table, name):
        async for event in table.events():
            seen.append((name, event))
            if name == "traces" and isinstance(event, tp.Closed):
                done.set()

    tasks = [asyncio.create_task(watch(steps, "steps")), asyncio.create_task(watch(traces, "traces"))]
    while live.server is None:
        await asyncio.sleep(0.01)
    sem = semconv_trace()
    # Live spans arrive in the order they end: chat, tool, chat, then the agent (root) span.
    assert await _post(live.port, pb_request(sem[1:2]), "application/x-protobuf") == 200
    assert await _post(live.port, gzip.compress(pb_request(sem[2:3])), "application/x-protobuf", "gzip") == 200
    assert await _post(live.port, json.dumps(request(sem[3:] + sem[:1])).encode(), "application/json") == 200
    assert await _post(live.port, b"\xff\xff", "application/x-protobuf") == 400
    await asyncio.wait_for(done.wait(), 5)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    step_rows = [e.row for name, e in seen if name == "steps" and isinstance(e, tp.RowAdded)]
    assert [(r.key[-1], r.source) for r in step_rows] == [(("step", 1), "user"), (("step", 2), "agent"), (("step", 3), "agent")]
    assert step_rows[1].tool_calls[0]["result"] == "18C, sunny"
    trace_rows = [e.row for name, e in seen if name == "traces" and isinstance(e, tp.RowAdded)]
    assert [(r.trace, r.n_spans, r.name) for r in trace_rows] == [("a" * 32, 4, "invoke_agent helper")]
    assert live.server is None  # stopped once nothing listens
