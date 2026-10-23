"""ATIF (Agent Trajectory Interchange Format) trajectories, as written by Harbor."""

from __future__ import annotations

import asyncio
import glob
import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from ..rows import Closed, Event, Key, Row, RowAdded
from ..schema import Schema
from .base import Adapter

IMAGE_PLACEHOLDER = "[image omitted]"

STEP_COLUMNS = {
    "source": str,  # "user" | "agent" | "system"
    "message": str,
    "reasoning": str,
    "tool_calls": list,  # [{id, name, arguments, result}]
    "observation": str,  # results not matched to a tool call
    "model_name": str,
    "timestamp": str,
    "metrics": dict,
    "is_sidechain": bool,
}

TOOL_CALL_COLUMNS = {"name": str, "arguments": dict, "result": str, "timestamp": str}


def clean_content(content: Any) -> Any:
    """Drop images; they would swamp the LM context and we don't analyze them yet."""
    if isinstance(content, str):
        return IMAGE_PLACEHOLDER if content.startswith('{"type": "image"') else content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                parts.append(IMAGE_PLACEHOLDER)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(block if isinstance(block, str) else json.dumps(block))
        return "\n".join(parts)
    return content


def step_rows(trajectory: dict, prefix: Key, carried: dict[str, Any]) -> list[tuple[Row, list[Row]]]:
    """Each step as a row, paired with its tool-call rows."""
    out = []
    for step in trajectory.get("steps", []):
        key = (*prefix, ("step", step["step_id"]))
        results = {
            r.get("source_call_id"): clean_content(r.get("content"))
            for r in (step.get("observation") or {}).get("results", [])
        }
        calls = []
        for call in step.get("tool_calls") or []:
            calls.append({
                "id": call.get("tool_call_id"),
                "name": call.get("function_name"),
                "arguments": call.get("arguments"),
                "result": results.pop(call.get("tool_call_id"), None),
            })
        unmatched = [str(v) for v in results.values() if v is not None]
        values = {
            **carried,
            "source": step.get("source"),
            "message": clean_content(step.get("message")),
            "reasoning": step.get("reasoning_content"),
            "tool_calls": calls,
            "observation": "\n".join(unmatched) or None,
            "model_name": step.get("model_name"),
            "timestamp": step.get("timestamp"),
            "metrics": step.get("metrics"),
            "is_sidechain": bool((step.get("extra") or {}).get("is_sidechain")),
        }
        call_rows = [
            Row((*key, ("call", i)), {**carried, **{k: c[k] for k in ("name", "arguments", "result")},
                                       "timestamp": step.get("timestamp")})
            for i, c in enumerate(calls)
        ]
        out.append((Row(key, values), call_rows))
    return out


async def load_json(path: Path) -> dict:
    return await asyncio.to_thread(lambda: json.loads(path.read_text()))


class ATIFAdapter(Adapter):
    """Bare ATIF trajectory files. Levels: trajectory > step (> call)."""

    levels = ("trajectory", "step")

    def __init__(self, paths: Iterable[Path]):
        self.paths = sorted(paths)

    def tables(self) -> dict[str, Schema]:
        return {
            "trajectories": Schema(("trajectory",), {"agent": str, "model": str, "n_steps": int, "final_metrics": dict}),
            "steps": Schema(("trajectory", "step"), dict(STEP_COLUMNS)),
            "tool_calls": Schema(("trajectory", "step", "call"), dict(TOOL_CALL_COLUMNS)),
        }

    def trajectory_id(self, path: Path, data: dict) -> str:
        return data.get("session_id") or str(path)

    async def events(self, table: str) -> AsyncIterator[Event]:
        for path in self.paths:
            data = await load_json(path)
            prefix = (("trajectory", self.trajectory_id(path, data)),)
            if table == "trajectories":
                agent = data.get("agent") or {}
                yield RowAdded(Row(prefix, {
                    "agent": agent.get("name"),
                    "model": agent.get("model_name"),
                    "n_steps": len(data.get("steps", [])),
                    "final_metrics": data.get("final_metrics"),
                }))
            else:
                for step, calls in step_rows(data, prefix, {}):
                    if table == "steps":
                        yield RowAdded(step)
                    else:
                        for call in calls:
                            yield RowAdded(call)
                        yield Closed(step.key)
            yield Closed(prefix)


def read_atif(path: str | Path | Iterable[str | Path]) -> ATIFAdapter:
    """A file, a directory (searched for *.json), a glob, or a list of paths."""
    if isinstance(path, (str, Path)):
        p = Path(path)
        if p.is_dir():
            paths = list(p.rglob("*.json"))
        elif p.exists():
            paths = [p]
        else:
            paths = [Path(x) for x in glob.glob(str(path), recursive=True)]
    else:
        paths = [Path(x) for x in path]
    if not paths:
        raise FileNotFoundError(f"no ATIF files found at {path}")
    return ATIFAdapter(paths)
