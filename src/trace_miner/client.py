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


class Alerts:
    def __init__(self) -> None:
        self._latest: Alert | None = None
        self._lock = threading.Lock()

    @property
    def latest(self) -> Alert | None:
        with self._lock:
            return self._latest

    def _note(self, record: Any) -> None:
        if not isinstance(record, dict) or not record.get("name"):
            return
        with self._lock:
            self._latest = Alert(record)


class Client:
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token

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
    def watch(self, output: str, **match: str) -> Iterator[Alerts]:
        if len(match) != 1:
            raise TraceMinerError("`watch` matches one field, for example watch('alerts', rollout_id='r1').")
        field, value = next(iter(match.items()))
        alerts = Alerts()
        stop = threading.Event()
        ready = threading.Event()
        error: list[BaseException] = []
        session = _WatchSession()
        thread = threading.Thread(
            target=self._listen,
            args=(output, field, value, alerts, stop, ready, error, session),
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
            yield alerts
        finally:
            stop.set()
            session.close()
            thread.join(timeout=5)

    def _listen(
        self,
        output: str,
        field: str,
        value: str,
        alerts: Alerts,
        stop: threading.Event,
        ready: threading.Event,
        error: list[BaseException],
        session: _WatchSession,
    ) -> None:
        last_id: str | None = None
        connected = False
        query = urlencode([("output", output), ("field", field), (field, value)])
        url = f"{self.base_url}/v1/events?{query}"
        while not stop.is_set():
            headers = self._headers({"Accept": "text/event-stream"})
            if last_id is not None:
                headers["Last-Event-ID"] = last_id
            request = Request(url, headers=headers)
            try:
                response = urlopen(request, timeout=5)
                response.fp.raw._sock.settimeout(0.5)
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
                    alerts._note(record)
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
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers


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
    event_id: str | None = None
    data: list[str] = []
    while not stop.is_set():
        try:
            raw = response.readline()
        except TimeoutError:
            continue
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
