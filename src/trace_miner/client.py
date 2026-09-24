from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from trace_miner.errors import TraceMinerError


class Alert:
    def __init__(self, record: dict[str, Any]):
        self.name = record.get("name")
        self.reason = record.get("reason")
        self.record = record

    def __bool__(self) -> bool:
        return True


class Events:
    def __init__(self) -> None:
        self._pending: list[Alert] = []
        self._lock = threading.Lock()

    def take(self, timeout: float | None = None) -> list[Alert]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if self._pending or deadline is None or time.monotonic() >= deadline:
                    found = list(self._pending)
                    self._pending.clear()
                    return found
            time.sleep(0.02)

    def _note(self, record: Any) -> None:
        if not isinstance(record, dict) or not record.get("name"):
            return
        with self._lock:
            self._pending.append(Alert(record))


class Client:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def post_step(self, record: dict[str, Any]) -> None:
        body = json.dumps(record).encode("utf-8")
        request = Request(
            self.base_url + "/v1/records",
            data=body,
            method="POST",
            headers=self._headers({"Content-Type": "application/json"}),
        )
        try:
            with urlopen(request, timeout=30) as response:
                response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise TraceMinerError(f"POST /v1/records failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise TraceMinerError(f"POST /v1/records failed: {exc.reason}") from exc

    @contextmanager
    def watch_events(self, **match: str) -> Iterator[Events]:
        if len(match) != 1:
            raise TraceMinerError("`watch_events` matches one field, for example watch_events(session_id='s1').")
        field, value = next(iter(match.items()))
        events = Events()
        stop = threading.Event()
        ready = threading.Event()
        error: list[BaseException] = []
        session = _WatchSession()
        thread = threading.Thread(
            target=self._listen,
            args=(field, value, events, stop, ready, error, session),
            name="trace-miner-watch",
            daemon=True,
        )
        thread.start()
        if not ready.wait(timeout=5):
            stop.set()
            session.close()
            thread.join(timeout=2)
            raise TraceMinerError("Timed out waiting for the alert stream.")
        if error:
            stop.set()
            session.close()
            thread.join(timeout=2)
            raise error[0]
        try:
            yield events
        finally:
            stop.set()
            session.close()
            thread.join(timeout=5)

    def _listen(
        self,
        field: str,
        value: str,
        events: Events,
        stop: threading.Event,
        ready: threading.Event,
        error: list[BaseException],
        session: _WatchSession,
    ) -> None:
        last_id: str | None = None
        connected = False
        query = urlencode([("field", field), (field, value)])
        url = f"{self.base_url}/v1/events?{query}"
        while not stop.is_set():
            headers = self._headers({"Accept": "text/event-stream"})
            if last_id is not None:
                headers["Last-Event-ID"] = last_id
            request = Request(url, headers=headers)
            try:
                response = urlopen(request, timeout=5)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").strip()
                if not connected:
                    error.append(TraceMinerError(f"GET /v1/events failed ({exc.code}): {detail}"))
                    ready.set()
                if stop.is_set() or not connected:
                    return
                time.sleep(0.2)
                continue
            except URLError as exc:
                if not connected:
                    error.append(TraceMinerError(f"GET /v1/events failed: {exc.reason}"))
                    ready.set()
                    return
                if stop.is_set():
                    return
                time.sleep(0.2)
                continue
            session.set(response)
            if not connected:
                connected = True
                ready.set()
            try:
                for event_id, data in _iter_sse(response, stop):
                    if event_id:
                        last_id = event_id
                    try:
                        record = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    events._note(record)
            except Exception:
                if stop.is_set():
                    return
            finally:
                session.clear(response)
                response.close()
            if stop.is_set():
                return
            time.sleep(0.2)

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        return dict(extra or {})


class _WatchSession:
    def __init__(self) -> None:
        self._response: Any = None
        self._lock = threading.Lock()

    def set(self, response: Any) -> None:
        with self._lock:
            self._response = response

    def clear(self, response: Any) -> None:
        with self._lock:
            if self._response is response:
                self._response = None

    def close(self) -> None:
        with self._lock:
            response = self._response
        if response is None:
            return
        try:
            sock = response.fp.raw._sock
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            try:
                response.close()
            except Exception:
                return


def _iter_sse(response: Any, stop: threading.Event) -> Iterator[tuple[str | None, str]]:
    # A blocking read with no socket-level timeout, by design: interrupting a
    # read on a chunked-transfer connection mid-chunk (as a raw socket
    # timeout can do) corrupts http.client's internal chunk-decoder state,
    # silently breaking the stream without raising anything obvious — and
    # since a fresh reconnect has no Last-Event-ID to replay from, whatever
    # event arrived during the broken window is gone for good, not just
    # delayed. `session.close()` already interrupts a pending read cleanly
    # via `socket.shutdown()` when the caller actually wants to stop.
    event_id: str | None = None
    data: list[str] = []
    while not stop.is_set():
        raw = response.readline()
        if not raw:
            break
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                yield event_id, "\n".join(data)
            event_id = None
            data = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("id:"):
            event_id = line[3:].lstrip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
