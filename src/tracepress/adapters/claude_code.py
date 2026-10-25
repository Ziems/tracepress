"""Claude Code and Claude Agent SDK sessions, from `~/.claude/projects/*/<session>.jsonl`.

Levels: project > session > step (> call). The project is the session's working directory.

Claude Code writes one line per content block, so an assistant turn with thinking, text and
two tool calls is four lines sharing a `message.id`; they become one step. Tool results come
back in user lines and are paired with their call by id. Subagent transcripts (in
`<session>/subagents/`, or inline as sidechain lines in older versions) are steps of the same
session with `is_sidechain=True` and their `agent_id`. Bookkeeping lines (attachments,
titles, queue operations, API request logs) are skipped.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .atif import clean_content
from .sessions import (
    Parsed, SessionAdapter, SessionFile, find_files, first_prompt, join, new_step, read_head,
    read_lines, seconds, select,
)

SESSION_COLUMNS = {
    "title": str,
    "prompt": str,  # the first thing the user asked
    "cwd": str,
    "git_branch": str,
    "model": str,
    "version": str,
    "entrypoint": str,
    "started_at": str,
    "finished_at": str,
    "duration_s": float,
    "n_steps": int,
    "n_subagents": int,
    "n_tool_errors": int,
    "input_tokens": int,
    "output_tokens": int,
    "cache_read_tokens": int,
    "cache_write_tokens": int,
    "path": str,
}


def _thread(lines, agent_id: str | None, calls: dict[str, dict]) -> list[dict]:
    """The steps of one transcript (the main thread or one subagent)."""
    steps: list[dict] = []
    by_message: dict[str, dict] = {}
    for line in lines:
        kind = line.get("type")
        message = line.get("message")
        if kind not in ("user", "assistant") or not isinstance(message, dict):
            continue
        thread = (line.get("agentId") or agent_id or "sidechain") if line.get("isSidechain") else agent_id
        ts = line.get("timestamp")
        content = message.get("content")
        if kind == "assistant":
            mid = message.get("id") or line.get("uuid")
            step = by_message.get(mid)
            if step is None:
                step = by_message[mid] = new_step("agent", ts, thread, model_name=message.get("model"),
                                                  _text=[], _thinking=[])
                steps.append(step)
            step["metrics"] = message.get("usage") or step["metrics"]
            for block in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                btype = block.get("type")
                if btype == "text":
                    step["_text"].append(block.get("text"))
                elif btype == "thinking":
                    step["_thinking"].append(block.get("thinking"))
                elif btype in ("tool_use", "server_tool_use"):
                    call = {"id": block.get("id"), "name": block.get("name"), "arguments": block.get("input"),
                            "result": None, "is_error": None}
                    step["tool_calls"].append(call)
                    calls[call["id"]] = call
                elif btype.endswith("_tool_result") and block.get("tool_use_id") in calls:
                    calls[block["tool_use_id"]]["result"] = clean_content(json.dumps(block.get("content")))
            continue
        # A user line: either something the user (or the harness) said, or tool results.
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        said, unmatched = [], []
        for block in blocks:
            if block.get("type") == "tool_result":
                result = clean_content(block.get("content"))
                call = calls.get(block.get("tool_use_id"))
                if call is not None:
                    call["result"], call["is_error"] = result, bool(block.get("is_error"))
                elif result:
                    unmatched.append(str(result))
            else:
                said.append(block)
        if unmatched and steps:
            steps[-1]["observation"] = join([steps[-1]["observation"], *unmatched])
        text = clean_content(said) if said else None
        if text:
            harness = line.get("isMeta") or line.get("isCompactSummary")
            step = new_step("system" if harness else "user", ts, thread)
            step["message"] = text
            steps.append(step)
    for step in steps:
        if step["source"] == "agent":
            step["message"] = join(step.pop("_text"))
            step["reasoning"] = join(step.pop("_thinking"))
    return steps


def _title(file: SessionFile, lines: list[dict]) -> str | None:
    title = next((l.get("customTitle") for l in reversed(lines) if l.get("type") == "custom-title"), None)
    title = title or next((l.get("summary") for l in reversed(lines) if l.get("type") == "summary"), None)
    sidecar = file.path.with_suffix("") / "custom-title.json"
    if not title and sidecar.exists():
        try:
            title = json.loads(sidecar.read_text()).get("customTitle")
        except (OSError, json.JSONDecodeError):
            pass
    return title


class ClaudeCodeAdapter(SessionAdapter):
    """Claude Code / Agent SDK sessions. Levels: project > session > step (> call)."""

    session_columns = SESSION_COLUMNS

    def parse(self, file: SessionFile) -> Parsed:
        lines = list(read_lines(file.path))
        calls: dict[str, dict] = {}
        steps = _thread(lines, None, calls)
        for path in file.extra:
            agent_id = path.stem.removeprefix("agent-")
            steps += _thread(read_lines(path), agent_id, calls)
        steps.sort(key=lambda s: s["timestamp"] or "")  # stable, so a turn's own order is kept

        # Usage is repeated on every line of a message; count each message once.
        usage = {}
        for line in lines:
            message = line.get("message")
            if line.get("type") == "assistant" and isinstance(message, dict) and message.get("usage"):
                usage[message.get("id") or line.get("uuid")] = message["usage"]
        total = lambda name: sum(u.get(name) or 0 for u in usage.values())
        models = Counter(s["model_name"] for s in steps if s["model_name"] and s["model_name"] != "<synthetic>")
        first = next((l for l in lines if l.get("cwd")), {})
        stamps = [l["timestamp"] for l in lines if isinstance(l.get("timestamp"), str)]
        started, finished = (min(stamps), max(stamps)) if stamps else (None, None)
        return Parsed({
            "title": _title(file, lines),
            "prompt": first_prompt(steps),
            "cwd": first.get("cwd"),
            "git_branch": first.get("gitBranch"),
            "model": models.most_common(1)[0][0] if models else None,
            "version": first.get("version"),
            "entrypoint": first.get("entrypoint"),
            "started_at": started,
            "finished_at": finished,
            "duration_s": seconds(started, finished),
            "n_steps": len(steps),
            "n_subagents": len({s["agent_id"] for s in steps if s["agent_id"]}),
            "n_tool_errors": sum(1 for c in calls.values() if c["is_error"]),
            "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "cache_read_tokens": total("cache_read_input_tokens"),
            "cache_write_tokens": total("cache_creation_input_tokens"),
            "path": str(file.path),
        }, steps)


def default_root() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"


def _is_subagent(path: Path) -> bool:
    return path.parent.name == "subagents" or path.name.startswith("agent-")


def read_claude_code(path: str | Path | None = None, project: str | None = None,
                     limit: int | None = None, subagents: bool = True) -> ClaudeCodeAdapter:
    """Claude Code or Agent SDK sessions.

    `path` is `~/.claude/projects` by default (or `$CLAUDE_CONFIG_DIR/projects`); it can also
    be one project's directory, one session `.jsonl`, or a glob. `project` keeps sessions
    whose working directory contains it; `limit` keeps the N most recently active.
    `subagents=False` leaves out subagent transcripts.
    """
    root = Path(path).expanduser() if path is not None else default_root()
    found = []
    for f in find_files(root, "*.jsonl", skip=_is_subagent):
        head: dict[str, Any] = read_head(f, ("cwd", "sessionId"))
        if "cwd" not in head:
            continue  # no conversation in it (e.g. only a title or summary)
        extra = sorted((f.with_suffix("") / "subagents").glob("agent-*.jsonl")) if subagents else []
        found.append(SessionFile(f, head["cwd"], head.get("sessionId") or f.stem, extra))
    found = select(found, project, limit)
    if not found:
        raise FileNotFoundError(f"no Claude Code sessions found at {root}")
    return ClaudeCodeAdapter(found)
