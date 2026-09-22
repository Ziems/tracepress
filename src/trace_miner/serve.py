from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from trace_miner.config import load_config
from trace_miner.errors import PipelineExecutionError, TraceMinerError
from trace_miner.model import Config, RunResult
from trace_miner.runtime import published_outputs, root_inputs, run_stream_async


_MAX_BODY = 8 * 1024 * 1024


@dataclass
class _Event:
    event_id: str
    record: Any


@dataclass
class _Subscriber:
    name: str
    field: str
    value: str
    seen: dict[str, int]
    queue: asyncio.Queue[_Event | None]


class _Hub:
    def __init__(self, outputs: dict[str, tuple[Path, ...]]):
        self.outputs = outputs
        self.cursors: dict[str, dict[str, int]] = {}
        for name, paths in outputs.items():
            self.cursors[name] = {str(path): path.stat().st_size if path.exists() else 0 for path in paths}
        self.subscribers: list[_Subscriber] = []
        self.closed = False

    def publish(self, name: str, path: Path, record: Any, next_offset: int) -> None:
        key = str(path)
        if self.closed or name not in self.cursors:
            return
        self.cursors[name][key] = max(self.cursors[name].get(key, 0), next_offset)
        for subscriber in list(self.subscribers):
            self._offer(subscriber, key, record, next_offset)

    def subscribe(self, name: str, field: str, value: str, last_event_id: str | None) -> _Subscriber:
        if name not in self.outputs:
            raise _RequestError(404, f"Unknown output `{name}`.")
        if last_event_id:
            try:
                seen = _decode_cursor(last_event_id)
            except ValueError as exc:
                raise _RequestError(400, "Invalid Last-Event-ID.") from exc
        else:
            seen = dict(self.cursors[name])
        subscriber = _Subscriber(name, field, value, seen, asyncio.Queue())
        self.subscribers.append(subscriber)
        if last_event_id:
            self._replay(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)

    def close(self) -> None:
        self.closed = True
        for subscriber in list(self.subscribers):
            subscriber.queue.put_nowait(None)

    def _replay(self, subscriber: _Subscriber) -> None:
        for path in self.outputs[subscriber.name]:
            start = subscriber.seen.get(str(path), 0)
            for _offset, next_offset, record in _read_jsonl_lines(path, start):
                self._offer(subscriber, str(path), record, next_offset)

    def _offer(self, subscriber: _Subscriber, path: str, record: Any, next_offset: int) -> None:
        if next_offset <= subscriber.seen.get(path, 0):
            return
        subscriber.seen[path] = next_offset
        if not _matches(record, subscriber.field, subscriber.value):
            return
        subscriber.queue.put_nowait(_Event(_encode_cursor(subscriber.seen), record))


class _RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class RunningServe:
    def __init__(self) -> None:
        self.url = ""
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._finished = threading.Event()
        self.error: BaseException | None = None
        self.result: RunResult | None = None

    def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._thread is not None and self._thread.is_alive():
            raise TraceMinerError("Timed out stopping trace-miner serve.")
        if self.error is not None:
            raise self.error


def serve(
    config_or_path: Config | str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    checkpoint_path: str | Path | None = None,
    from_start: bool = False,
    from_end: bool = False,
    reset_checkpoint: bool = False,
    poll_interval: float = 1.0,
    progress: bool | None = None,
    progress_stream: Any = None,
) -> RunResult:
    running = start_serve(
        config_or_path,
        host=host,
        port=port,
        checkpoint_path=checkpoint_path,
        from_start=from_start,
        from_end=from_end,
        reset_checkpoint=reset_checkpoint,
        poll_interval=poll_interval,
        progress=progress,
        progress_stream=progress_stream,
    )
    try:
        running._finished.wait()
    except KeyboardInterrupt:
        running.close()
        raise
    running.close()
    if not isinstance(running.result, RunResult):
        raise PipelineExecutionError("serve stopped before producing a result.")
    return running.result


def start_serve(
    config_or_path: Config | str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    checkpoint_path: str | Path | None = None,
    from_start: bool = False,
    from_end: bool = False,
    reset_checkpoint: bool = False,
    poll_interval: float = 1.0,
    progress: bool | None = None,
    progress_stream: Any = None,
) -> RunningServe:
    config = load_config(config_or_path) if not isinstance(config_or_path, Config) else config_or_path
    roots = root_inputs(config)
    if len(roots) != 1:
        raise PipelineExecutionError("`trace-miner serve` requires exactly one root input.")
    running = RunningServe()

    def run() -> None:
        try:
            running.result = asyncio.run(
                _serve_on_loop(
                    running,
                    config,
                    host=host,
                    port=port,
                    checkpoint_path=checkpoint_path,
                    from_start=from_start,
                    from_end=from_end,
                    reset_checkpoint=reset_checkpoint,
                    poll_interval=poll_interval,
                    progress=progress,
                    progress_stream=progress_stream,
                )
            )
        except BaseException as exc:
            running.error = exc
        finally:
            running._ready.set()
            running._finished.set()

    thread = threading.Thread(target=run, name="trace-miner-serve")
    running._thread = thread
    thread.start()
    if not running._ready.wait(timeout=10):
        raise TraceMinerError("Timed out starting trace-miner serve.")
    if running.error is not None:
        thread.join(timeout=5)
        raise running.error
    return running


async def _serve_on_loop(
    running: RunningServe,
    config: Config,
    *,
    host: str,
    port: int,
    checkpoint_path: str | Path | None,
    from_start: bool,
    from_end: bool,
    reset_checkpoint: bool,
    poll_interval: float,
    progress: bool | None,
    progress_stream: Any,
) -> RunResult:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    wake = asyncio.Event()
    running._loop = loop
    running._stop = stop
    hub = _Hub(published_outputs(config))
    app = _build_app(hub, config, root_inputs(config)[0], wake)
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="error", access_log=False, lifespan="off")
    )
    server_task = asyncio.create_task(server.serve())
    try:
        await _wait_until_started(server, server_task)
        bound_host, bound_port = _bound_address(server)
        running.url = _display_url(bound_host, bound_port)
        return await run_stream_async(
            config,
            checkpoint_path=checkpoint_path,
            from_start=from_start,
            from_end=from_end,
            reset_checkpoint=reset_checkpoint,
            poll_interval=poll_interval,
            progress=progress,
            progress_stream=progress_stream,
            wake_event=wake,
            stop_event=stop,
            on_write=hub.publish,
            on_ready=running._ready.set,
        )
    finally:
        hub.close()
        server.should_exit = True
        await server_task


def _build_app(
    hub: _Hub,
    config: Config,
    root_input: Path,
    wake: asyncio.Event,
) -> Starlette:
    async def records(request: Request) -> Response:
        body = await request.body()
        if len(body) > _MAX_BODY:
            return PlainTextResponse("Record is too large.\n", status_code=400)
        try:
            record = json.loads(body)
        except json.JSONDecodeError as exc:
            return PlainTextResponse(f"Invalid JSON: {exc}\n", status_code=400)
        if not isinstance(record, dict):
            return PlainTextResponse("Record must be a JSON object.\n", status_code=400)
        _append_record(root_input, record)
        wake.set()
        return Response(status_code=202)

    async def events(request: Request) -> Response:
        try:
            field, value, output = _event_filter(config, dict(request.query_params))
            subscriber = hub.subscribe(output, field, value, request.headers.get("last-event-id"))
        except _RequestError as exc:
            return PlainTextResponse(exc.message + "\n", status_code=exc.status)

        async def stream():
            try:
                while True:
                    item = await subscriber.queue.get()
                    if item is None:
                        break
                    payload = json.dumps(item.record, separators=(",", ":"))
                    yield f"id: {item.event_id}\ndata: {payload}\n\n".encode()
            finally:
                hub.unsubscribe(subscriber)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return Starlette(
        routes=[
            Route("/v1/records", records, methods=["POST"]),
            Route("/v1/events", events, methods=["GET"]),
        ]
    )


async def _wait_until_started(server: uvicorn.Server, server_task: asyncio.Task[None]) -> None:
    for _ in range(1000):
        if server.started:
            return
        if server_task.done():
            exc = server_task.exception()
            if exc is not None:
                raise exc
            raise TraceMinerError("HTTP server exited before it started.")
        await asyncio.sleep(0.01)
    raise TraceMinerError("Timed out starting the HTTP server.")


def _bound_address(server: uvicorn.Server) -> tuple[str, int]:
    sockets = [sock for listener in server.servers or [] for sock in (listener.sockets or [])]
    if not sockets:
        raise TraceMinerError("HTTP server did not bind a port.")
    host, port = sockets[0].getsockname()[:2]
    return str(host), int(port)


def _display_url(host: str, port: int) -> str:
    shown = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{shown}:{port}"


def _event_filter(config: Config, query: dict[str, str]) -> tuple[str, str, str]:
    output = query.get("output", "")
    if not output:
        raise _RequestError(400, "Missing `output`.")
    if output not in published_outputs(config):
        raise _RequestError(404, f"Unknown output `{output}`.")
    if config.id_field:
        field = config.id_field
        if "field" in query and query["field"] != field:
            raise _RequestError(400, f"`field` must be `{field}`.")
        if not query.get(field):
            raise _RequestError(400, f"Missing `{field}`.")
        return field, query[field], output
    field = query.get("field", "")
    if not field:
        raise _RequestError(400, "Missing `field`.")
    if not query.get(field):
        raise _RequestError(400, f"Missing `{field}`.")
    return field, query[field], output


def _matches(record: Any, field: str, value: str) -> bool:
    if not isinstance(record, dict) or field not in record:
        return False
    return str(record[field]) == value


def _append_record(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _read_jsonl_lines(path: Path, offset: int) -> list[tuple[int, int, Any]]:
    if not path.exists():
        return []
    lines: list[tuple[int, int, Any]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            line_offset = handle.tell()
            line = handle.readline()
            if not line or not line.endswith(b"\n"):
                break
            next_offset = handle.tell()
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            lines.append((line_offset, next_offset, record))
    return lines


def _encode_cursor(cursors: dict[str, int]) -> str:
    raw = json.dumps(cursors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(event_id: str) -> dict[str, int]:
    pad = "=" * (-len(event_id) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(event_id + pad))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("bad cursor") from exc
    if not isinstance(data, dict):
        raise ValueError("bad cursor")
    return {str(key): int(value) for key, value in data.items()}

