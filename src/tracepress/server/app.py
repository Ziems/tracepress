"""The TracePress server: one live stream of traces, and pipelines that run on it continuously.

    tracepress serve --port 7700

Clients push events into the stream, register pipelines (pickled plans) that run on it, and
listen to any pipeline's output. Registering runs the pickled code on this machine, so the
server listens on localhost by default and needs a token for anything else.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import pickle
import threading
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from aiohttp import web

from .. import lm as lm_config
from .. import runtime, telemetry
from ..adapters.base import Adapter
from ..rows import Event, RowAdded, RowUpdated, sort_key
from ..schema import Schema
from .protocol import decode, encode

log = logging.getLogger("tracepress")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


class Hub:
    """The stream: every table's events so far (up to `retain` each), fanned out to subscribers."""

    def __init__(self, retain: int | None = None):
        self.retain = retain
        self.schemas: dict[str, Schema] = {}
        self.history: dict[str, deque] = {}
        self.subscribers: dict[str, set[asyncio.Queue]] = {}

    def declare(self, schemas: dict[str, Schema]) -> None:
        for name, schema in schemas.items():
            self.schemas[name] = schema
            self.history.setdefault(name, deque(maxlen=self.retain))
            self.subscribers.setdefault(name, set())

    def publish(self, table: str, events: list[Event]) -> None:
        if table not in self.schemas:
            raise KeyError(f"no table {table!r} has been declared; stream from an adapter, or push with its schema")
        self.history[table].extend(events)
        for queue in self.subscribers[table]:
            for event in events:
                queue.put_nowait(event)


class ServerSource(Adapter):
    """The hub as a live source for registered pipelines: history first, then new events."""

    bounded = False

    def __init__(self, hub: Hub):
        self.hub = hub
        self.levels = ()

    def tables(self) -> dict[str, Schema]:
        return dict(self.hub.schemas)

    async def events(self, table: str) -> AsyncIterator[Event]:
        queue: asyncio.Queue = asyncio.Queue()
        backlog = list(self.hub.history[table])  # taken with the subscription, so nothing falls between
        self.hub.subscribers[table].add(queue)
        try:
            for event in backlog:
                yield event
            while True:
                yield await queue.get()
        finally:
            self.hub.subscribers[table].discard(queue)


def rebind(node, source: ServerSource, memo: dict | None = None):
    """The plan with every source built on a client's `client.source` reading from `source` instead."""
    from ..runtime import SourceOp
    from .client import RemoteSource

    memo = {} if memo is None else memo
    if id(node) in memo:
        return memo[id(node)]
    if node.parent is None:
        op = node.make()
        if isinstance(op, SourceOp) and isinstance(op.adapter, RemoteSource):
            table = op.table
            out = replace(node, make=lambda: SourceOp(source, table))
        else:
            out = node  # e.g. a lookup table shipped with the pipeline
    else:
        out = replace(
            node, parent=rebind(node.parent, source, memo), others=tuple(rebind(o, source, memo) for o in node.others)
        )
    memo[id(node)] = out
    return out


class Pipeline:
    """One registered pipeline: its run, its newest rows, and who's listening."""

    def __init__(self, name: str, node):
        self.name, self.node = name, node
        self.rows: dict = {}
        self.listeners: set[asyncio.Queue] = set()
        self.status, self.error = "running", None
        self.task: asyncio.Task | None = None

    def send(self, message: dict[str, Any] | None) -> None:
        for queue in self.listeners:
            queue.put_nowait(message)

    async def run(self) -> None:
        try:
            async for event in runtime.stream_events(self.node):
                if isinstance(event, (RowAdded, RowUpdated)):
                    self.rows[event.row.key] = event.row
                self.send(encode(event))
            self.status = "finished"
        except asyncio.CancelledError:
            self.status = "stopped"
            raise
        except Exception as e:  # a pipeline's failure is its own; the server keeps going
            log.exception("pipeline %r failed", self.name)
            self.status, self.error = "failed", repr(e)
            self.send({"kind": "error", "error": repr(e)})
        finally:
            self.send(None)  # ends every listener's stream

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "error": self.error, "rows": len(self.rows)}


def make_app(token: str | None = None, retain: int | None = None) -> web.Application:
    hub = Hub(retain)
    source = ServerSource(hub)
    pipelines: dict[str, Pipeline] = {}

    @web.middleware
    async def no_browsers(request, handler):
        # Browsers send Origin on every cross-site POST and websocket; clients never do. A web page
        # open in the user's browser must not reach /register, which runs what it's sent.
        if "Origin" in request.headers:
            raise web.HTTPForbidden(text="requests from web pages aren't accepted")
        return await handler(request)

    @web.middleware
    async def auth(request, handler):
        given = request.headers.get("Authorization", "")
        if token is not None and not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
            raise web.HTTPUnauthorized(text="missing or wrong token")
        return await handler(request)

    async def unpickle(request) -> Any:
        if request.content_type != "application/octet-stream":
            raise web.HTTPUnsupportedMediaType(text="send pickles as application/octet-stream")
        return pickle.loads(await request.read())

    async def health(request):
        return web.json_response({"ok": True})

    async def put_tables(request):
        hub.declare(await unpickle(request))
        return web.json_response({"tables": sorted(hub.schemas)})

    async def get_tables(request):
        return web.Response(body=pickle.dumps(hub.schemas), content_type="application/octet-stream")

    async def push(request):
        body = await request.json()
        by_table: dict[str, list[Event]] = {}
        for item in body["events"]:
            by_table.setdefault(item["table"], []).append(decode(item))
        try:
            for table, events in by_table.items():
                hub.publish(table, events)
        except KeyError as e:
            raise web.HTTPBadRequest(text=str(e.args[0]))
        return web.json_response({"events": sum(map(len, by_table.values()))})

    async def stop(name: str) -> None:
        await end(pipelines.pop(name, None))

    async def end(p: Pipeline | None) -> None:
        if p is not None and p.task is not None:
            p.task.cancel()
            await asyncio.gather(p.task, return_exceptions=True)

    async def register(request):
        spec = await unpickle(request)  # runs the client's code: see the module docstring
        name = spec["name"]
        p = Pipeline(name, rebind(spec["node"], source))
        p.task = asyncio.ensure_future(p.run())
        # Swapped in before any await, so a registration that overlaps this one replaces this one.
        old, pipelines[name] = pipelines.get(name), p
        telemetry.report_pipeline_registered(p.node, replaced=old is not None)
        await end(old)  # registering a name again replaces it
        return web.json_response(p.info())

    async def unregister(request):
        name = request.match_info["name"]
        if name not in pipelines:
            raise web.HTTPNotFound(text=f"no pipeline named {name!r}")
        await stop(name)
        return web.json_response({"stopped": name})

    async def list_pipelines(request):
        return web.json_response([p.info() for p in pipelines.values()])

    async def listen(request):
        name = request.match_info["name"]
        p = pipelines.get(name)
        if p is None:
            raise web.HTTPNotFound(text=f"no pipeline named {name!r}")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        queue: asyncio.Queue = asyncio.Queue()
        if request.query.get("from_start") in ("1", "true"):
            for key in sorted(p.rows, key=sort_key):
                row = p.rows[key]
                queue.put_nowait(encode(RowUpdated(row) if row.version > 1 else RowAdded(row)))
        p.listeners.add(queue)
        if p.task is not None and p.task.done():  # it already told its listeners how it ended
            if p.error is not None:
                queue.put_nowait({"kind": "error", "error": p.error})
            queue.put_nowait(None)

        async def send():
            while (message := await queue.get()) is not None:
                await ws.send_json(message)
            await ws.close()  # ends the read loop below

        sending = asyncio.ensure_future(send())
        try:
            async for _ in ws:  # clients send nothing; reading is what answers their close
                pass
        finally:
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)  # a send to a gone client just ends it
            p.listeners.discard(queue)
            await ws.close()
        return ws

    async def shutdown(app):
        for name in list(pipelines):
            await stop(name)

    app = web.Application(middlewares=[no_browsers, auth], client_max_size=256 * 1024 * 1024)
    app.add_routes(
        [
            web.get("/health", health),
            web.post("/tables", put_tables),
            web.get("/tables", get_tables),
            web.post("/push", push),
            web.post("/register", register),
            web.get("/pipelines", list_pipelines),
            web.delete("/pipelines/{name}", unregister),
            web.get("/listen/{name}", listen),
        ]
    )
    app.on_shutdown.append(shutdown)
    return app


def check_host(host: str, token: str | None) -> None:
    if host not in LOCAL_HOSTS and not token:
        raise ValueError(f"serving on {host!r} lets anyone who can reach it run code here; pass a token")


def _report_started(host: str, token: str | None, retain: int | None) -> None:
    telemetry.report_server_started(
        local=host in LOCAL_HOSTS,
        token=token is not None,
        retain=retain is not None,
        default_lm=lm_config.config().lm is not None or bool(os.environ.get("TRACEPRESS_LM")),
    )


def serve(host: str = "127.0.0.1", port: int = 7700, token: str | None = None, retain: int | None = None) -> None:
    """Run the server until interrupted."""
    check_host(host, token)
    _report_started(host, token, retain)
    web.run_app(make_app(token, retain), host=host, port=port, print=lambda msg: log.warning(msg))


class Running:
    """A server running in a background thread, e.g. for tests or a notebook. `stop()` ends it."""

    def __init__(self, host: str, port: int, token: str | None, retain: int | None):
        check_host(host, token)
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._runner: web.AppRunner | None = None
        self.address = ""
        self._thread = threading.Thread(target=self._main, args=(host, port, token, retain), daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("the server didn't start")
        _report_started(host, token, retain)

    def _main(self, host, port, token, retain):
        asyncio.set_event_loop(self._loop)
        self._runner = runner = web.AppRunner(make_app(token, retain))
        self._loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, host, port)
        self._loop.run_until_complete(site.start())
        port = runner.addresses[0][1]
        self.address = f"{host}:{port}"
        self._ready.set()
        self._loop.run_forever()

    def stop(self) -> None:
        if not self._thread.is_alive():
            return  # already stopped: its loop would never run the cleanup
        if self._runner is not None:
            asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


def serve_in_thread(
    host: str = "127.0.0.1", port: int = 0, token: str | None = None, retain: int | None = None
) -> Running:
    """Start a server in a background thread; `port=0` picks a free port (see `.address`)."""
    return Running(host, port, token, retain)


__all__ = ["make_app", "serve", "serve_in_thread"]
