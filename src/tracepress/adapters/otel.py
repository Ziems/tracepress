"""OpenTelemetry traces of LLM apps and agents, from OTLP JSON files.

Levels: trace > span, and trace > step (> call).

Spans form a tree of any depth, so the `spans` table keeps it flat: one row per span with
its `parent_span_id` and `depth`, keyed by start order. The `steps` table reads the LLM and
tool spans as agent steps in the shared step convention (`source`, `message`, `reasoning`,
`tool_calls`), so pipelines written for other agent traces work unchanged. It understands the
OpenTelemetry GenAI conventions (`gen_ai.*`), OpenInference (`llm.*`, `tool.*`, used by
Phoenix, Arize and many instrumentors) and OpenLLMetry's flattened `gen_ai.prompt.N.*`.

Files can be what the collector's file exporter writes (one OTLP JSON request per line), a
single OTLP JSON document, or the Python SDK's `ConsoleSpanExporter` output.
"""

from __future__ import annotations

import asyncio
import glob
import json
from collections import Counter
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..rows import Closed, Event, Key, Row, RowAdded
from ..schema import Schema
from .atif import STEP_COLUMNS, TOOL_CALL_COLUMNS
from .base import Adapter

TRACE_COLUMNS = {
    "name": str,  # the root span's name
    "service_name": str,
    "model": str,
    "input": str,  # the root span's input / output, when instrumented (OpenInference)
    "output": str,
    "status": str,  # "error" if any span failed, else "ok"
    "error": str,  # the first failed span's message
    "started_at": str,
    "finished_at": str,
    "duration_s": float,
    "n_spans": int,
    "n_llm_calls": int,
    "n_tool_calls": int,
    "n_errors": int,
    "input_tokens": int,
    "output_tokens": int,
}

SPAN_COLUMNS = {
    "span_id": str,
    "parent_span_id": str,
    "depth": int,
    "name": str,
    "span_type": str,  # "llm" | "tool" | "agent" | "other"
    "kind": str,  # OpenTelemetry span kind: "internal", "client", ...
    "service_name": str,
    "model": str,
    "started_at": str,
    "timestamp": str,  # when the span ended (and was exported)
    "duration_ms": float,
    "status": str,  # "ok" | "error" | "unset"
    "error": str,
    "input_tokens": int,
    "output_tokens": int,
    "attributes": dict,
    "events": list,
}

OTEL_STEP_COLUMNS = {**STEP_COLUMNS, "service_name": str, "agent_name": str, "span_id": str, "error": str}
OTEL_CALL_COLUMNS = {**TOOL_CALL_COLUMNS, "is_error": bool, "service_name": str}

LLM_OPERATIONS = {"chat", "text_completion", "generate_content", "completion", "responses"}
KINDS = {0: "unspecified", 1: "internal", 2: "server", 3: "client", 4: "producer", 5: "consumer"}
STATUSES = {0: "unset", 1: "ok", 2: "error"}


# ---------------------------------------------------------------------------- spans


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    start_ns: int
    end_ns: int
    attributes: dict[str, Any]
    events: list[dict]
    status: str
    status_message: str | None
    resource: dict[str, Any] = field(default_factory=dict)

    @property
    def service_name(self) -> str | None:
        return self.resource.get("service.name")

    @property
    def started_at(self) -> str | None:
        return iso(self.start_ns)

    @property
    def finished_at(self) -> str | None:
        return iso(self.end_ns)


def iso(ns: int | None) -> str | None:
    if not ns:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _ns(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, str) and value:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1e9)
    return 0


def _id(value: Any) -> str | None:
    if not value:
        return None
    value = str(value).lower().removeprefix("0x")
    return None if set(value) <= {"0"} else value


def any_value(value: dict) -> Any:
    """Decode an OTLP AnyValue (JSON encoding)."""
    if not isinstance(value, dict):
        return value
    for name in ("stringValue", "string_value"):
        if name in value:
            return value[name]
    for name in ("boolValue", "bool_value"):
        if name in value:
            return value[name]
    for name in ("intValue", "int_value"):
        if name in value:
            return int(value[name])
    for name in ("doubleValue", "double_value"):
        if name in value:
            return float(value[name])
    for name in ("arrayValue", "array_value"):
        if name in value:
            return [any_value(v) for v in (value[name] or {}).get("values", [])]
    for name in ("kvlistValue", "kvlist_value"):
        if name in value:
            return attributes((value[name] or {}).get("values", []))
    for name in ("bytesValue", "bytes_value"):
        if name in value:
            return value[name]
    return None


def attributes(items: Any) -> dict[str, Any]:
    if isinstance(items, dict):  # already decoded, e.g. ConsoleSpanExporter
        return dict(items)
    return {kv.get("key"): any_value(kv.get("value")) for kv in items or [] if isinstance(kv, dict)}


def _enum(value: Any, names: dict[int, str], prefix: str) -> str:
    if isinstance(value, int):
        return names.get(value, str(value))
    text = str(value or "").lower()
    for p in (prefix, "spankind.", "statuscode.", "status_code_", "span_kind_"):
        text = text.removeprefix(p)
    return text or names[0]


def spans_from_otlp(data: dict) -> Iterator[Span]:
    """Spans in one OTLP JSON request: `{"resourceSpans": [...]}`."""
    for rs in data.get("resourceSpans") or data.get("resource_spans") or []:
        resource = attributes((rs.get("resource") or {}).get("attributes"))
        for ss in rs.get("scopeSpans") or rs.get("scope_spans") or rs.get("instrumentationLibrarySpans") or []:
            for s in ss.get("spans") or []:
                status = s.get("status") or {}
                yield Span(
                    trace_id=_id(s.get("traceId") or s.get("trace_id")),
                    span_id=_id(s.get("spanId") or s.get("span_id")),
                    parent_id=_id(s.get("parentSpanId") or s.get("parent_span_id")),
                    name=s.get("name") or "",
                    kind=_enum(s.get("kind"), KINDS, "span_kind_"),
                    start_ns=_ns(s.get("startTimeUnixNano") or s.get("start_time_unix_nano")),
                    end_ns=_ns(s.get("endTimeUnixNano") or s.get("end_time_unix_nano")),
                    attributes=attributes(s.get("attributes")),
                    events=[{"name": e.get("name"), "timestamp": iso(_ns(e.get("timeUnixNano") or e.get("time_unix_nano"))),
                             "attributes": attributes(e.get("attributes"))} for e in s.get("events") or []],
                    status=_enum(status.get("code"), STATUSES, "status_code_"),
                    status_message=status.get("message") or None,
                    resource=resource,
                )


def span_from_console(s: dict) -> Span:
    """One span as the Python SDK's `ConsoleSpanExporter` (`span.to_json()`) prints it."""
    context = s.get("context") or {}
    status = s.get("status") or {}
    return Span(
        trace_id=_id(context.get("trace_id")),
        span_id=_id(context.get("span_id")),
        parent_id=_id(s.get("parent_id")),
        name=s.get("name") or "",
        kind=_enum(s.get("kind"), KINDS, "spankind."),
        start_ns=_ns(s.get("start_time")),
        end_ns=_ns(s.get("end_time")),
        attributes=dict(s.get("attributes") or {}),
        events=[{"name": e.get("name"), "timestamp": e.get("timestamp"), "attributes": dict(e.get("attributes") or {})}
                for e in s.get("events") or []],
        status=_enum(status.get("status_code"), STATUSES, "statuscode."),
        status_message=status.get("description") or None,
        resource=dict((s.get("resource") or {}).get("attributes") or {}),
    )


def json_documents(text: str) -> Iterator[Any]:
    """JSON values in a file: JSON Lines, one document, or pretty-printed documents back to back."""
    decoder, i, n = json.JSONDecoder(), 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            value, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            # A torn line (a file still being written); skip to the next line.
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        yield value


def spans_from_document(doc: Any) -> Iterator[Span]:
    if isinstance(doc, list):
        for item in doc:
            yield from spans_from_document(item)
    elif isinstance(doc, dict):
        if "resourceSpans" in doc or "resource_spans" in doc:
            yield from spans_from_otlp(doc)
        elif "context" in doc and "name" in doc:
            yield span_from_console(doc)


# ---------------------------------------------------------------------------- reading GenAI spans


def _get(attrs: dict, *names: str) -> Any:
    for name in names:
        value = attrs.get(name)
        if value not in (None, ""):
            return value
    return None


def span_type(span: Span) -> str:
    a = span.attributes
    operation = str(a.get("gen_ai.operation.name") or "").lower()
    oi = str(a.get("openinference.span.kind") or a.get("traceloop.span.kind") or "").lower()
    if operation == "execute_tool" or oi == "tool":
        return "tool"
    if operation in LLM_OPERATIONS or oi == "llm" or a.get("llm.request.type") in ("chat", "completion") \
            or any(k.startswith(("gen_ai.prompt.", "llm.input_messages.")) for k in a) or "gen_ai.input.messages" in a:
        return "llm"
    if operation in ("invoke_agent", "create_agent") or oi == "agent":
        return "agent"
    return "other"


def _model(span: Span) -> str | None:
    return _get(span.attributes, "gen_ai.response.model", "gen_ai.request.model", "llm.model_name")


def _tokens(span: Span) -> tuple[int | None, int | None]:
    a = span.attributes
    return (_get(a, "gen_ai.usage.input_tokens", "llm.token_count.prompt", "gen_ai.usage.prompt_tokens"),
            _get(a, "gen_ai.usage.output_tokens", "llm.token_count.completion", "gen_ai.usage.completion_tokens"))


def unflatten(attrs: dict, prefix: str) -> Any:
    """`{"p.0.role": "user", "p.0.content": "hi"}` -> `[{"role": "user", "content": "hi"}]`."""
    root: dict = {}
    dot = prefix + "."
    for key, value in attrs.items():
        if not key.startswith(dot):
            continue
        node, parts = root, key[len(dot):].split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            node[parts[-1]] = value

    def listify(node):
        if not isinstance(node, dict):
            return node
        node = {k: listify(v) for k, v in node.items()}
        if node and all(k.isdigit() for k in node):
            return [node[k] for k in sorted(node, key=int)]
        return node

    return listify(root) if root else None


def _json(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(t for t in (_text(v) for v in value) if t) or None
    if isinstance(value, dict):
        if value.get("type") in ("image", "image_url", "input_image"):
            return "[image omitted]"
        for name in ("text", "content", "message_content"):
            if name in value:
                return _text(value[name])
    return json.dumps(value)


def _arguments(value: Any) -> Any:
    value = _json(value)
    return value if isinstance(value, dict) or value is None else {"input": value}


def _message(role: Any, text: Any = None, reasoning: Any = None, calls: Iterable[dict] = (),
             tool_call_id: Any = None) -> dict:
    return {"role": str(role or "").lower(), "text": _text(text), "reasoning": _text(reasoning),
            "tool_calls": [c for c in calls if c.get("name") or c.get("id")], "tool_call_id": tool_call_id}


def _semconv_messages(value: Any) -> list[dict]:
    """`gen_ai.input.messages` / `gen_ai.output.messages`: [{role, parts: [{type, ...}]}]."""
    out = []
    for m in _json(value) or []:
        if not isinstance(m, dict):
            continue
        texts, thoughts, calls = [], [], []
        for part in m.get("parts") or []:
            kind = part.get("type")
            if kind == "text":
                texts.append(part.get("content"))
            elif kind == "reasoning":
                thoughts.append(part.get("content"))
            elif kind == "tool_call":
                calls.append({"id": part.get("id"), "name": part.get("name"), "arguments": _arguments(part.get("arguments"))})
            elif kind == "tool_call_response":
                out.append(_message("tool", _text(part.get("response", part.get("result"))), tool_call_id=part.get("id")))
        if texts or thoughts or calls or not m.get("parts"):
            out.append(_message(m.get("role"), texts or m.get("content"), thoughts, calls))
    return out


def _openinference_messages(items: Any) -> list[dict]:
    out = []
    for item in items or []:
        m = (item or {}).get("message") or {}
        contents = [c.get("message_content") or {} for c in m.get("contents") or []]
        texts = [c.get("text") for c in contents if c.get("type", "text") == "text"] or [m.get("content")]
        calls = [{"id": tc.get("id"), "name": (tc.get("function") or {}).get("name"),
                  "arguments": _arguments((tc.get("function") or {}).get("arguments"))}
                 for tc in ((c or {}).get("tool_call") or {} for c in m.get("tool_calls") or [])]
        out.append(_message(m.get("role"), texts, None, calls, m.get("tool_call_id")))
    return out


def _openllmetry_messages(items: Any) -> list[dict]:
    out = []
    for m in items or []:
        m = m or {}
        calls = [{"id": c.get("id"), "name": c.get("name"), "arguments": _arguments(c.get("arguments"))}
                 for c in m.get("tool_calls") or [] if isinstance(c, dict)]
        out.append(_message(m.get("role") or "assistant", _json(m.get("content")), m.get("reasoning"), calls,
                            m.get("tool_call_id")))
    return out


EVENT_ROLES = {"gen_ai.system.message": "system", "gen_ai.user.message": "user",
               "gen_ai.assistant.message": "assistant", "gen_ai.tool.message": "tool"}


def _event_messages(span: Span) -> tuple[list[dict], list[dict]]:
    """The older GenAI conventions put messages in span events."""
    inputs, outputs = [], []
    for e in span.events:
        a = e.get("attributes") or {}
        body = _json(a.get("gen_ai.event.content") or a.get("event_body")) or a
        if not isinstance(body, dict):
            body = {"content": body}
        if e.get("name") == "gen_ai.choice":
            body = _json(body.get("message")) or body
            outputs.append(_openllmetry_messages([body])[0])
        elif e.get("name") in EVENT_ROLES:
            inputs.append(_openllmetry_messages([{**body, "role": EVENT_ROLES[e["name"]]}])[0])
    return inputs, outputs


def llm_messages(span: Span) -> tuple[list[dict], list[dict]]:
    """(input messages, output messages) of an LLM span, in whichever convention it uses."""
    a = span.attributes
    if "gen_ai.input.messages" in a or "gen_ai.output.messages" in a:
        return _semconv_messages(a.get("gen_ai.input.messages")), _semconv_messages(a.get("gen_ai.output.messages"))
    if any(k.startswith(("llm.input_messages.", "llm.output_messages.")) for k in a):
        return (_openinference_messages(unflatten(a, "llm.input_messages")),
                _openinference_messages(unflatten(a, "llm.output_messages")))
    if any(k.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for k in a):
        return _openllmetry_messages(unflatten(a, "gen_ai.prompt")), _openllmetry_messages(unflatten(a, "gen_ai.completion"))
    inputs, outputs = _event_messages(span)
    if inputs or outputs:
        return inputs, outputs
    # Nothing structured: fall back to the raw input and output values.
    raw_in, raw_out = _get(a, "input.value", "traceloop.entity.input"), _get(a, "output.value", "traceloop.entity.output")
    return ([_message("user", raw_in)] if raw_in else []), ([_message("assistant", raw_out)] if raw_out else [])


def tool_call(span: Span) -> dict:
    a = span.attributes
    return {
        "id": _get(a, "gen_ai.tool.call.id", "tool_call.id", "tool.call.id"),
        "name": _get(a, "gen_ai.tool.name", "tool.name", "traceloop.entity.name") or span.name,
        "arguments": _arguments(_get(a, "gen_ai.tool.call.arguments", "tool.parameters", "input.value",
                                     "traceloop.entity.input")),
        "result": _text(_json(_get(a, "gen_ai.tool.call.result", "output.value", "traceloop.entity.output"))),
        "is_error": span.status == "error",
    }


def agent_name(span: Span) -> str | None:
    return _get(span.attributes, "gen_ai.agent.name", "agent.name") or \
        (_get(span.attributes, "traceloop.entity.name") or span.name if span_type(span) == "agent" else None)


# ---------------------------------------------------------------------------- spans -> steps


class TraceSteps:
    """Builds one trace's steps as its spans arrive, in start order (a file) or end order (live).

    An LLM span becomes an agent step: its output's text, reasoning and tool calls, preceded by
    a user step for any new user message in its input. A tool span fills in the result of the
    call that requested it, or becomes a step of its own if no LLM span asked for it. A step is
    released once all its calls have results, a later LLM call of the same agent arrives, or
    the trace finishes, and steps are released in order.
    """

    def __init__(self, service_name: str | None = None):
        self.service_name = service_name
        self.steps: list[dict] = []
        self.released = 0
        self.calls: dict[str, dict] = {}
        self.said: set[str] = set()
        self.agents: dict[str, str] = {}  # span id -> agent name, for agent spans seen so far
        self.parents: dict[str, str | None] = {}

    def _agent(self, span: Span) -> str | None:
        own = _get(span.attributes, "gen_ai.agent.name", "agent.name")
        if own:
            return own
        parent, hops = span.parent_id, 0
        while parent and hops < 100:
            if parent in self.agents:
                return self.agents[parent]
            parent, hops = self.parents.get(parent), hops + 1
        return None

    def _step(self, source: str, span: Span, **values) -> dict:
        step = {"source": source, "message": None, "reasoning": None, "tool_calls": [], "observation": None,
                "model_name": None, "timestamp": span.started_at, "metrics": None, "is_sidechain": False,
                "service_name": self.service_name, "agent_name": self._agent(span), "span_id": span.span_id,
                "error": None, "_scope": span.parent_id, "_open": False, **values}
        self.steps.append(step)
        return step

    def add(self, span: Span) -> list[dict]:
        self.parents[span.span_id] = span.parent_id
        kind = span_type(span)
        if kind == "agent":
            self.agents[span.span_id] = agent_name(span)
        elif kind == "llm":
            self._llm(span)
        elif kind == "tool":
            call = tool_call(span)
            known = self._requested(call)
            if known is not None:
                known["result"], known["is_error"], known["_done"] = call["result"], call["is_error"], True
                if known["result"] is None and span.status == "error":
                    known["result"] = span.status_message
            else:
                step = self._step("agent", span, error=span.status_message if span.status == "error" else None)
                step["tool_calls"].append(call)
        return self.release()

    def _requested(self, call: dict) -> dict | None:
        """The unanswered call an LLM asked for that this tool span ran: by id, else by name."""
        if call["id"]:
            known = self.calls.get(call["id"])
            return known if known is not None and not known["_done"] else None
        return next((c for c in self.calls.values() if c["name"] == call["name"] and not c["_done"]), None)

    def _llm(self, span: Span) -> None:
        inputs, outputs = llm_messages(span)
        for step in self.steps[self.released:]:
            if step["_scope"] == span.parent_id:
                step["_open"] = False  # the agent moved on to its next call
        last_assistant = max((i for i, m in enumerate(inputs) if m["role"] == "assistant"), default=-1)
        for m in inputs[last_assistant + 1:]:
            if m["role"] == "tool":
                call = self.calls.get(m["tool_call_id"])
                if call is not None and not call["_done"]:
                    call["result"], call["_done"] = m["text"], True
            elif m["role"] == "user" and m["text"] and m["text"] not in self.said:
                self.said.add(m["text"])
                self._step("user", span, message=m["text"], agent_name=None)
        answer = [m for m in outputs if m["role"] in ("assistant", "model", "")] or outputs
        tokens_in, tokens_out = _tokens(span)
        step = self._step(
            "agent", span,
            message="\n\n".join(m["text"] for m in answer if m["text"]) or None,
            reasoning="\n\n".join(m["reasoning"] for m in answer if m["reasoning"]) or None,
            model_name=_model(span),
            metrics={"input_tokens": tokens_in, "output_tokens": tokens_out} if tokens_in or tokens_out else None,
            error=span.status_message if span.status == "error" else None,
        )
        for m in answer:
            for c in m["tool_calls"]:
                call = {**c, "result": None, "is_error": None, "_done": False}
                step["tool_calls"].append(call)
                self.calls[call["id"] or f"{span.span_id}/{len(self.calls)}"] = call
        step["_open"] = bool(step["tool_calls"])

    def release(self, finished: bool = False) -> list[dict]:
        """Steps that are now complete, in order."""
        out = []
        while self.released < len(self.steps):
            step = self.steps[self.released]
            waiting = any(not c.get("_done", True) for c in step["tool_calls"])
            if not finished and step["_open"] and waiting:
                break
            calls = [{k: v for k, v in c.items() if k != "_done"} for c in step["tool_calls"]]
            out.append({**{k: v for k, v in step.items() if not k.startswith("_")}, "tool_calls": calls})
            self.released += 1
        return out

    def finish(self) -> list[dict]:
        return self.release(finished=True)


def trace_rows(trace_key: Key, spans: list[Span]) -> tuple[Row, list[Row]]:
    """The trace row and its span rows, from all of a trace's spans."""
    spans = sorted(spans, key=lambda s: (s.start_ns, s.end_ns))
    ids = {s.span_id for s in spans}
    parent = {s.span_id: s.parent_id for s in spans}

    def depth(s: Span) -> int:
        d, p = 0, s.parent_id
        while p in ids and d < 1000:
            d, p = d + 1, parent[p]
        return d

    span_rows, types = [], Counter()
    for i, s in enumerate(spans):
        kind = span_type(s)
        types[kind] += 1
        tokens_in, tokens_out = _tokens(s)
        span_rows.append(Row((*trace_key, ("span", i)), {
            "span_id": s.span_id, "parent_span_id": s.parent_id, "depth": depth(s), "name": s.name,
            "span_type": kind, "kind": s.kind, "service_name": s.service_name, "model": _model(s),
            "started_at": s.started_at, "timestamp": s.finished_at,
            "duration_ms": (s.end_ns - s.start_ns) / 1e6 if s.end_ns and s.start_ns else None,
            "status": s.status, "error": s.status_message if s.status == "error" else None,
            "input_tokens": tokens_in, "output_tokens": tokens_out,
            "attributes": s.attributes, "events": s.events,
        }))
    roots = [s for s in spans if s.parent_id not in ids]
    root = roots[0] if roots else None
    llm = [s for s in spans if span_type(s) == "llm"]
    errors = [s for s in spans if s.status == "error"]
    models = Counter(m for m in map(_model, llm) if m)
    start = min((s.start_ns for s in spans if s.start_ns), default=0)
    end = max((s.end_ns for s in spans), default=0)
    total = lambda i: sum((_tokens(s)[i] or 0) for s in llm) or None
    trace = Row(trace_key, {
        "name": root.name if root else None,
        "service_name": root.service_name if root else None,
        "model": models.most_common(1)[0][0] if models else None,
        "input": _text(_get(root.attributes, "input.value")) if root else None,
        "output": _text(_get(root.attributes, "output.value")) if root else None,
        "status": "error" if errors else "ok",
        "error": errors[0].status_message if errors else None,
        "started_at": iso(start), "finished_at": iso(end),
        "duration_s": (end - start) / 1e9 if start and end else None,
        "n_spans": len(spans), "n_llm_calls": len(llm), "n_tool_calls": types["tool"], "n_errors": len(errors),
        "input_tokens": total(0), "output_tokens": total(1),
    })
    return trace, span_rows


def step_events(trace_key: Key, first: int, steps: list[dict], table: str) -> Iterator[Event]:
    """Step or tool-call events for steps numbered from `first`."""
    for i, step in enumerate(steps, start=first):
        key = (*trace_key, ("step", i))
        if table == "steps":
            yield RowAdded(Row(key, step))
            continue
        for j, call in enumerate(step["tool_calls"]):
            yield RowAdded(Row((*key, ("call", j)), {
                "name": call["name"], "arguments": call["arguments"], "result": call["result"],
                "is_error": call["is_error"], "timestamp": step["timestamp"], "service_name": step["service_name"],
            }))
        yield Closed(key)


def otel_tables() -> dict[str, Schema]:
    return {
        "traces": Schema(("trace",), dict(TRACE_COLUMNS)),
        "spans": Schema(("trace", "span"), dict(SPAN_COLUMNS)),
        "steps": Schema(("trace", "step"), dict(OTEL_STEP_COLUMNS)),
        "tool_calls": Schema(("trace", "step", "call"), dict(OTEL_CALL_COLUMNS)),
    }


def trace_events(trace_id: str, spans: list[Span], table: str) -> Iterator[Event]:
    """Every event for one finished trace, then its close."""
    key = (("trace", trace_id),)
    if table in ("traces", "spans"):
        trace, span_rows = trace_rows(key, spans)
        for row in [trace] if table == "traces" else span_rows:
            yield RowAdded(row)
    else:
        ordered = sorted(spans, key=lambda s: (s.start_ns, s.end_ns))
        builder = TraceSteps(next((s.service_name for s in ordered if s.service_name), None))
        steps = []
        for span in ordered:
            steps += builder.add(span)
        steps += builder.finish()
        yield from step_events(key, 1, steps, table)
    yield Closed(key)


# ---------------------------------------------------------------------------- the file adapter


def load_spans(paths: list[Path]) -> dict[str, list[Span]]:
    """Every span in the files, by trace, with duplicates (re-exported spans) dropped."""
    traces: dict[str, dict[str, Span]] = {}
    for path in paths:
        for doc in json_documents(path.read_text(encoding="utf-8", errors="replace")):
            for span in spans_from_document(doc):
                if span.trace_id and span.span_id:
                    traces.setdefault(span.trace_id, {})[span.span_id] = span
    return {t: list(spans.values()) for t, spans in traces.items()}


class OTelAdapter(Adapter):
    """OpenTelemetry traces. Levels: trace > span, and trace > step (> call)."""

    levels = ("trace", "step")

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self._traces: dict[str, list[Span]] | None = None

    def tables(self) -> dict[str, Schema]:
        return otel_tables()

    async def traces(self) -> dict[str, list[Span]]:
        # A trace's spans can be spread over many export batches (and files), so read them all.
        if self._traces is None:
            self._traces = await asyncio.to_thread(load_spans, self.paths)
        return self._traces

    async def events(self, table: str) -> AsyncIterator[Event]:
        traces = await self.traces()
        start = lambda t: min((s.start_ns for s in traces[t]), default=0)
        for trace_id in sorted(traces, key=lambda t: (start(t), t)):
            for event in trace_events(trace_id, traces[trace_id], table):
                yield event


def read_otel(path: str | Path | Iterable[str | Path]) -> OTelAdapter:
    """OTLP JSON files: a file, a directory (searched for .json and .jsonl), a glob, or a list."""
    if isinstance(path, (str, Path)):
        p = Path(path).expanduser()
        if p.is_dir():
            paths = [*p.rglob("*.json"), *p.rglob("*.jsonl")]
        elif p.exists():
            paths = [p]
        else:
            paths = [Path(x) for x in glob.glob(str(path), recursive=True)]
    else:
        paths = [Path(x) for x in path]
    if not paths:
        raise FileNotFoundError(f"no OpenTelemetry files found at {path}")
    return OTelAdapter(sorted(paths))
