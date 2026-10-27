"""A live OpenTelemetry source: an OTLP/HTTP receiver that agents export their traces to.

Point any OTLP exporter at it (`OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`) and it
yields the same tables as `tp.read.otel`. Both protobuf (the SDKs' default) and JSON
requests are accepted, gzipped or not. The protobuf decoding is done here, so no extra
dependency is needed.

Steps stream as they complete. Spans and the trace row arrive when the trace is done: once
its root span has arrived and nothing new has come for `settle` seconds, or after
`idle_timeout` seconds of silence. Spans arriving for a trace already closed are dropped.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import struct
import time
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..rows import Closed, Event
from ..schema import Schema
from .base import Adapter
from .otel import Span, TraceSteps, otel_tables, spans_from_otlp, step_events, trace_events

# ---------------------------------------------------------------------------- protobuf


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """(field number, value) pairs of one protobuf message. Length-delimited values are bytes."""
    i, n = 0, len(buf)
    while i < n:
        tag, i = _varint(buf, i)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = struct.unpack_from("<Q", buf, i)[0], i + 8
        elif wire == 2:
            size, i = _varint(buf, i)
            value, i = buf[i:i + size], i + size
        elif wire == 5:
            value, i = struct.unpack_from("<I", buf, i)[0], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, value


def _str(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _any_value(buf: bytes) -> dict:
    for number, value in _fields(buf):
        if number == 1:
            return {"stringValue": _str(value)}
        if number == 2:
            return {"boolValue": bool(value)}
        if number == 3:
            return {"intValue": value - (1 << 64) if value >= 1 << 63 else value}
        if number == 4:
            return {"doubleValue": struct.unpack("<d", struct.pack("<Q", value))[0]}
        if number == 5:
            return {"arrayValue": {"values": [_any_value(v) for n, v in _fields(value) if n == 1]}}
        if number == 6:
            return {"kvlistValue": {"values": [_key_value(v) for n, v in _fields(value) if n == 1]}}
        if number == 7:
            return {"bytesValue": value.hex()}
    return {}


def _key_value(buf: bytes) -> dict:
    out: dict = {"key": "", "value": {}}
    for number, value in _fields(buf):
        if number == 1:
            out["key"] = _str(value)
        elif number == 2:
            out["value"] = _any_value(value)
    return out


def _event(buf: bytes) -> dict:
    out: dict = {"attributes": []}
    for number, value in _fields(buf):
        if number == 1:
            out["timeUnixNano"] = value
        elif number == 2:
            out["name"] = _str(value)
        elif number == 3:
            out["attributes"].append(_key_value(value))
    return out


def _span(buf: bytes) -> dict:
    out: dict = {"attributes": [], "events": [], "status": {}}
    for number, value in _fields(buf):
        if number == 1:
            out["traceId"] = value.hex()
        elif number == 2:
            out["spanId"] = value.hex()
        elif number == 4:
            out["parentSpanId"] = value.hex()
        elif number == 5:
            out["name"] = _str(value)
        elif number == 6:
            out["kind"] = value
        elif number == 7:
            out["startTimeUnixNano"] = value
        elif number == 8:
            out["endTimeUnixNano"] = value
        elif number == 9:
            out["attributes"].append(_key_value(value))
        elif number == 11:
            out["events"].append(_event(value))
        elif number == 15:
            for n, v in _fields(value):
                if n == 2:
                    out["status"]["message"] = _str(v)
                elif n == 3:
                    out["status"]["code"] = v
    return out


def decode_traces_request(buf: bytes) -> dict:
    """An `ExportTraceServiceRequest` (protobuf) as its OTLP JSON equivalent."""
    resource_spans = []
    for number, rs in _fields(buf):
        if number != 1:
            continue
        entry: dict = {"resource": {"attributes": []}, "scopeSpans": []}
        for n, value in _fields(rs):
            if n == 1:
                entry["resource"]["attributes"] = [_key_value(v) for m, v in _fields(value) if m == 1]
            elif n == 2:
                entry["scopeSpans"].append({"spans": [_span(v) for m, v in _fields(value) if m == 2]})
        resource_spans.append(entry)
    return {"resourceSpans": resource_spans}


# ---------------------------------------------------------------------------- HTTP


async def _serve_http(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, on_spans) -> None:
    """A minimal HTTP/1.1 server for `POST /v1/traces`, with keep-alive."""
    try:
        while True:
            request = await reader.readline()
            if not request:
                return
            method, path, *_ = request.decode("latin-1").split()
            headers = {}
            while (line := await reader.readline()) not in (b"\r\n", b"\n", b""):
                name, _, value = line.decode("latin-1").partition(":")
                headers[name.strip().lower()] = value.strip()
            body = await reader.readexactly(int(headers.get("content-length") or 0))
            status, content_type, reply = 200, headers.get("content-type", "application/x-protobuf"), b""
            if method != "POST" or path.split("?")[0].rstrip("/") != "/v1/traces":
                status, reply = 404, b"not found"
            else:
                try:
                    encoding = headers.get("content-encoding", "")
                    if encoding == "gzip":
                        body = gzip.decompress(body)
                    elif encoding == "deflate":
                        body = zlib.decompress(body)
                    if "json" in content_type:
                        data, reply = json.loads(body or b"{}"), b"{}"
                    else:
                        data = decode_traces_request(body)
                    on_spans(list(spans_from_otlp(data)))
                except Exception as e:  # a bad request must not stop the receiver
                    status, reply = 400, str(e).encode()
            reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}[status]
            writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n"
                         f"Content-Length: {len(reply)}\r\n\r\n".encode() + reply)
            await writer.drain()
            if headers.get("connection", "").lower() == "close":
                return
    except (asyncio.IncompleteReadError, ConnectionError, ValueError):
        return
    finally:
        writer.close()


# ---------------------------------------------------------------------------- the live adapter


@dataclass
class _Trace:
    spans: dict[str, Span] = field(default_factory=dict)
    builder: TraceSteps | None = None
    n_steps: int = 0
    root_seen: bool = False
    last_seen: float = 0.0


class OTLPReceiver(Adapter):
    """Live OpenTelemetry traces. Levels: trace > span, and trace > step (> call)."""

    levels = ("trace", "step")
    bounded = False

    def __init__(self, host: str, port: int, settle: float, idle_timeout: float):
        self.host, self.port = host, port
        self.settle, self.idle_timeout = settle, idle_timeout
        self.subscribers: list[tuple[str, asyncio.Queue]] = []
        self.open: dict[str, _Trace] = {}
        self.closed: set[str] = set()
        self.server: asyncio.AbstractServer | None = None
        self._sweeper: asyncio.Task | None = None

    def tables(self) -> dict[str, Schema]:
        return otel_tables()

    async def start(self) -> None:
        if self.server is None:
            self.server = await asyncio.start_server(
                lambda r, w: _serve_http(r, w, self.receive), self.host, self.port)
            self.port = self.server.sockets[0].getsockname()[1]  # the real port, when 0 was asked for
            self._sweeper = asyncio.create_task(self._sweep())

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.server = self._sweeper = None

    def _send(self, events_for) -> None:
        for table, queue in self.subscribers:
            for event in events_for(table):
                queue.put_nowait(event)

    def receive(self, spans: list[Span]) -> None:
        now = time.monotonic()
        for span in spans:
            if not span.trace_id or not span.span_id or span.trace_id in self.closed:
                continue
            trace = self.open.setdefault(span.trace_id, _Trace())
            if span.span_id in trace.spans:
                continue  # re-exported
            trace.spans[span.span_id] = span
            trace.last_seen = now
            trace.root_seen = trace.root_seen or span.parent_id is None
            if trace.builder is None:
                trace.builder = TraceSteps(span.service_name)
            self._release(span.trace_id, trace, trace.builder.add(span))

    def _release(self, trace_id: str, trace: _Trace, steps: list[dict]) -> None:
        if not steps:
            return
        first, trace.n_steps = trace.n_steps + 1, trace.n_steps + len(steps)
        key = (("trace", trace_id),)
        self._send(lambda table: step_events(key, first, steps, table)
                   if table in ("steps", "tool_calls") else ())

    def close_trace(self, trace_id: str) -> None:
        trace = self.open.pop(trace_id)
        self.closed.add(trace_id)
        self._release(trace_id, trace, trace.builder.finish())
        spans = list(trace.spans.values())
        key = (("trace", trace_id),)
        self._send(lambda table: trace_events(trace_id, spans, table)
                   if table in ("traces", "spans") else [Closed(key)])

    async def _sweep(self) -> None:
        while True:
            await asyncio.sleep(min(self.settle, 0.5))
            now = time.monotonic()
            for trace_id, trace in list(self.open.items()):
                quiet = now - trace.last_seen
                if (trace.root_seen and quiet >= self.settle) or quiet >= self.idle_timeout:
                    self.close_trace(trace_id)

    async def events(self, table: str) -> AsyncIterator[Event]:
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append((table, queue))
        try:
            await self.start()
            while True:
                yield await queue.get()
        finally:
            self.subscribers.remove((table, queue))
            if not self.subscribers:
                await self.stop()


def otlp(port: int = 4318, host: str = "127.0.0.1", settle: float = 2.0, idle_timeout: float = 300.0) -> OTLPReceiver:
    """Receive traces live over OTLP/HTTP on `host:port` (path `/v1/traces`).

    It listens once a pipeline over it starts. A trace is done `settle` seconds after its
    root span arrives, or after `idle_timeout` seconds with no new spans.
    """
    return OTLPReceiver(host, port, settle, idle_timeout)
