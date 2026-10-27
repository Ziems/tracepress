"""Shared pieces for local coding-agent session logs (Claude Code, Codex).

Both are one `.jsonl` file per session, so both adapters have the same shape:
project > session > step (> call), where the project is the session's working directory.
"""

from __future__ import annotations

import asyncio
import glob
import json
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..rows import Closed, Event, Key, Row, RowAdded
from ..schema import Schema
from .atif import STEP_COLUMNS, TOOL_CALL_COLUMNS
from .base import Adapter

SESSION_STEP_COLUMNS = {**STEP_COLUMNS, "agent_id": str}  # agent_id: the subagent, None on the main thread
SESSION_CALL_COLUMNS = {**TOOL_CALL_COLUMNS, "is_error": bool, "agent_id": str}


@dataclass
class SessionFile:
    path: Path
    project: str
    session: str
    extra: list[Path] = field(default_factory=list)  # e.g. subagent transcripts

    @property
    def key(self) -> Key:
        return (("project", self.project), ("session", self.session))


@dataclass
class Parsed:
    values: dict[str, Any]
    steps: list[dict[str, Any]]  # step columns; tool_calls as [{id, name, arguments, result, is_error}]


def read_lines(path: Path) -> Iterator[dict]:
    """Every JSON object in a .jsonl file. A torn last line (a live session) is skipped."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


def seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    try:
        return (parse(end) - parse(start)).total_seconds()
    except ValueError:
        return None


def find_files(path: str | Path, pattern: str, skip: callable = lambda p: False) -> list[Path]:
    """A file, a directory searched for `pattern`, or a glob."""
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        return [f for f in p.rglob(pattern) if not skip(f)]
    return [Path(x) for x in glob.glob(str(p), recursive=True) if not skip(Path(x))]


def select(files: list[SessionFile], project: str | None, limit: int | None) -> list[SessionFile]:
    """Keep sessions whose project contains `project`, then the `limit` most recent."""
    if project:
        files = [f for f in files if project.lower() in f.project.lower()]
    if limit is not None:
        files = sorted(files, key=lambda f: f.path.stat().st_mtime, reverse=True)[:limit]
    return sorted(files, key=lambda f: (f.project, f.session))


class SessionAdapter(Adapter):
    """One row per session, its steps, and their tool calls. Subclasses parse one session."""

    levels = ("project", "session", "step")
    session_columns: dict[str, Any] = {}
    step_columns: dict[str, Any] = SESSION_STEP_COLUMNS

    def __init__(self, files: list[SessionFile]):
        self.files = files

    def tables(self) -> dict[str, Schema]:
        return {
            "sessions": Schema(self.levels[:2], dict(self.session_columns)),
            "steps": Schema(self.levels, dict(self.step_columns)),
            "tool_calls": Schema((*self.levels, "call"), dict(SESSION_CALL_COLUMNS)),
        }

    @abstractmethod
    def parse(self, file: SessionFile) -> Parsed: ...

    async def prepare(self) -> None:
        """Work across all sessions, done once before any of them is parsed."""

    async def events(self, table: str) -> AsyncIterator[Event]:
        await self.prepare()
        by_project: dict[str, list[SessionFile]] = {}
        for file in self.files:
            by_project.setdefault(file.project, []).append(file)
        for project, files in sorted(by_project.items()):
            for file in files:
                parsed = await asyncio.to_thread(self.parse, file)
                for event in session_events(file.key, parsed, table):
                    yield event
                yield Closed(file.key)
            yield Closed((("project", project),))


def session_events(prefix: Key, parsed: Parsed, table: str) -> Iterable[Event]:
    if table == "sessions":
        yield RowAdded(Row(prefix, parsed.values))
        return
    for i, step in enumerate(parsed.steps, start=1):
        key = (*prefix, ("step", i))
        if table == "steps":
            yield RowAdded(Row(key, step))
            continue
        for j, call in enumerate(step["tool_calls"]):
            yield RowAdded(Row((*key, ("call", j)), {
                "name": call["name"], "arguments": call["arguments"], "result": call["result"],
                "is_error": call.get("is_error"), "timestamp": step["timestamp"],
                "agent_id": step["agent_id"],
            }))
        yield Closed(key)


def new_step(source: str, timestamp: str | None, agent_id: str | None = None, **values) -> dict[str, Any]:
    return {
        "source": source, "message": None, "reasoning": None, "tool_calls": [], "observation": None,
        "model_name": None, "timestamp": timestamp, "metrics": None,
        "is_sidechain": agent_id is not None, "agent_id": agent_id, **values,
    }


def join(parts: list[str]) -> str | None:
    parts = [p for p in parts if p]
    return "\n\n".join(parts) or None


def first_prompt(steps: list[dict]) -> str | None:
    return next((s["message"] for s in steps if s["source"] == "user" and not s["is_sidechain"] and s["message"]), None)


def read_head(path: Path, wanted: Iterable[str], max_lines: int = 200) -> dict[str, Any]:
    """Scan the start of a file for the first value of each wanted top-level field."""
    wanted, found = set(wanted), {}
    for i, data in enumerate(read_lines(path)):
        for name in wanted - found.keys():
            if data.get(name):
                found[name] = data[name]
        if found.keys() >= wanted or i >= max_lines:
            break
    return found
