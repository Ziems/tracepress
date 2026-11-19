"""How events travel between clients and the server: JSON, one object per event."""

from __future__ import annotations

import json
from typing import Any

from ..operators import _plain
from ..rows import Closed, Event, Key, Row, RowAdded, RowUpdated


def _key(key: Key) -> list:
    return [[level, ident] for level, ident in key]


def _unkey(data: list) -> Key:
    return tuple((level, ident) for level, ident in data)


def _safe(values: dict[str, Any]) -> dict[str, Any]:
    """Values as JSON can carry them: pydantic models as dicts, anything else odd as a string."""
    return json.loads(json.dumps(_plain(values), default=str))


def encode(event: Event, table: str | None = None) -> dict[str, Any]:
    if isinstance(event, Closed):
        out: dict[str, Any] = {"kind": "closed", "prefix": _key(event.prefix)}
    else:
        row = event.row
        out = {"kind": "updated" if isinstance(event, RowUpdated) else "added", "key": _key(row.key),
               "values": _safe(row.values), "final": row.final, "version": row.version}
    if table is not None:
        out["table"] = table
    return out


def decode(data: dict[str, Any]) -> Event:
    if data["kind"] == "closed":
        return Closed(_unkey(data["prefix"]))
    row = Row(_unkey(data["key"]), data.get("values", {}), data.get("final", True), data.get("version", 0))
    return RowUpdated(row) if data["kind"] == "updated" else RowAdded(row)
