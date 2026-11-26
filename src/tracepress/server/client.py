"""Talk to a TracePress server: `client = tp.connect("localhost:7700")`.

Any client can stream traces in, register pipelines, or listen to them, and they don't have to
be the same client, or on the same machine.
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..adapters.base import Adapter
from ..rows import Closed, Event, Key, Row, RowAdded, RowUpdated
from ..schema import Schema
from .protocol import decode, encode

if TYPE_CHECKING:
    from ..plan import Table


class RemoteSource(Adapter):
    """The server's stream, for building pipelines to register: `live = client.source`.

    It has the server's tables but no rows here; a pipeline built on it runs on the server.
    """

    bounded = False

    def __init__(self, schemas: dict[str, Schema]):
        self.schemas = schemas
        self.levels = next(iter(schemas.values())).levels if schemas else ()

    def tables(self) -> dict[str, Schema]:
        return dict(self.schemas)

    async def events(self, table: str) -> AsyncIterator[Event]:
        raise RuntimeError("a pipeline on client.source runs on the server: client.register(name=..., pipeline=...)")
        yield  # pragma: no cover


@dataclass(frozen=True)
class ServerEvent:
    """One event from a registered pipeline: a new or updated row, or a closed group."""

    kind: str  # "added", "updated" or "closed"
    row: Row | None = None
    prefix: Key | None = None


class Streaming:
    """A source being streamed to the server in the background. `wait()` or `stop()` it."""

    def __init__(self, client: Client, adapter: Adapter, tables: list[str], batch: int, every: float):
        self.client, self.adapter, self.tables = client, adapter, tables
        self.batch, self.every = batch, every
        self.sent = 0
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()

    def _main(self) -> None:
        try:
            asyncio.run(self._pump())
        except BaseException as e:  # noqa: BLE001 - reported by wait()
            self.error = e

    async def _pump(self) -> None:
        loop = asyncio.get_running_loop()
        pending: list[dict] = []

        async def flush() -> None:
            nonlocal pending
            if pending:
                batch, pending = pending, []
                await loop.run_in_executor(None, self.client._push_encoded, batch)
                self.sent += len(batch)

        async def read(table: str) -> None:
            async for event in self.adapter.events(table):
                if self._stop.is_set():
                    return
                pending.append(encode(event, table))
                if len(pending) >= self.batch:
                    await flush()

        async def tick() -> None:
            while True:
                await asyncio.sleep(self.every)
                await flush()

        ticker = asyncio.ensure_future(tick())
        try:
            await asyncio.gather(*(read(t) for t in self.tables))
        finally:
            ticker.cancel()
            await flush()

    def wait(self, timeout: float | None = None) -> int:
        """Block until a bounded source has been streamed in full; returns the events sent.

        Raises `TimeoutError` if the stream is still going after `timeout` seconds.
        """
        self._thread.join(timeout)
        if self.error is not None:
            raise self.error
        if self._thread.is_alive():
            raise TimeoutError(f"still streaming after {timeout}s ({self.sent} events sent so far)")
        return self.sent

    def stop(self) -> None:
        self._stop.set()


def _quote(name: str) -> str:
    """A pipeline name as one URL path segment: names may hold `/`, `?`, spaces and the like."""
    return urllib.parse.quote(name, safe="")


class Client:
    def __init__(self, address: str, token: str | None = None, timeout: float = 30):
        self.base = address if "://" in address else f"http://{address}"
        self.token, self.timeout = token, timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(
        self, method: str, path: str, data: bytes | None = None, content_type: str = "application/json"
    ) -> bytes:
        headers = {**self._headers(), **({"Content-Type": content_type} if data is not None else {})}
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"server said {e.code}: {e.read().decode(errors='replace')}") from None

    # --- The stream -------------------------------------------------------------------

    def declare(self, schemas: dict[str, Schema]) -> None:
        """Tell the server which tables the stream has (`stream()` does this for you)."""
        self._request("POST", "/tables", pickle.dumps(schemas), "application/octet-stream")

    @property
    def source(self) -> RemoteSource:
        """The server's stream, to build pipelines on: `live = client.source; live.steps.lm_map(...)`."""
        schemas = pickle.loads(self._request("GET", "/tables"))
        if not schemas:
            raise RuntimeError("the server's stream has no tables yet; stream or declare some first")
        return RemoteSource(schemas)

    def stream(
        self, adapter: Adapter, tables: Iterable[str] | None = None, batch: int = 200, every: float = 0.2
    ) -> Streaming:
        """Stream an adapter's events to the server in the background, e.g. `tp.watch.otlp()`.

        Returns at once. For a bounded source (a stored job, `tp.replay(...)`), `.wait()` blocks
        until it has all been sent; `.stop()` stops early.
        """
        schemas = adapter.tables()
        names = list(tables) if tables is not None else list(schemas)
        self.declare({t: schemas[t] for t in names})
        return Streaming(self, adapter, names, batch, every)

    def push(self, items: Iterable[Row | Event], table: str) -> int:
        """Send rows (or events such as `tp.Closed(prefix)`) into one of the stream's tables."""
        events = [RowAdded(i) if isinstance(i, Row) else i for i in items]
        return self._push_encoded([encode(e, table) for e in events])

    def _push_encoded(self, events: list[dict]) -> int:
        body = json.dumps({"events": events}, default=str).encode()
        return json.loads(self._request("POST", "/push", body))["events"]

    # --- Pipelines --------------------------------------------------------------------

    def register(self, name: str, pipeline: Table) -> dict[str, Any]:
        """Run `pipeline` on the server continuously, under `name`. Registering a name again replaces it.

        The pipeline is pickled with its code, so the server runs it as written. Functions it
        uses that live in your own modules must be importable on the server too.
        """
        import cloudpickle

        blob = cloudpickle.dumps({"name": name, "node": pipeline._node})
        return json.loads(self._request("POST", "/register", blob, "application/octet-stream"))

    def unregister(self, name: str) -> None:
        self._request("DELETE", f"/pipelines/{_quote(name)}")

    def pipelines(self) -> list[dict[str, Any]]:
        """Registered pipelines: name, status (running, finished, failed or stopped) and rows so far."""
        return json.loads(self._request("GET", "/pipelines"))

    async def listen(self, name: str, from_start: bool = False) -> AsyncIterator[ServerEvent]:
        """Each new or updated row (and closed group) from a registered pipeline, as it happens.

        `from_start=True` first sends the rows it has produced so far. The stream ends when the
        pipeline is unregistered or stops.
        """
        import aiohttp

        url = self.base.replace("http", "ws", 1) + f"/listen/{_quote(name)}" + ("?from_start=1" if from_start else "")
        async with aiohttp.ClientSession(headers=self._headers()) as session, session.ws_connect(url) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                data = msg.json()
                if data["kind"] == "error":
                    raise RuntimeError(f"pipeline {name!r} failed on the server: {data['error']}")
                event = decode(data)
                if isinstance(event, Closed):
                    yield ServerEvent("closed", prefix=event.prefix)
                else:
                    yield ServerEvent("updated" if isinstance(event, RowUpdated) else "added", row=event.row)


def connect(address: str = "localhost:7700", token: str | None = None) -> Client:
    """A client for a TracePress server started with `tracepress serve`.

    `token` defaults to the `TRACEPRESS_TOKEN` environment variable, so it needn't appear in code.
    """
    return Client(address, token if token is not None else os.environ.get("TRACEPRESS_TOKEN") or None)
