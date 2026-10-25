"""Codex CLI sessions, from `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`.

Levels: project > session > step (> call). The project is the session's working directory.

Each line is `{timestamp, type, payload}`. `session_meta` describes the session,
`turn_context` the model in use, and `response_item`s are the conversation: messages,
reasoning, tool calls and their outputs (paired by `call_id`). One model response (reasoning,
a message and any tool calls) becomes one step. `event_msg`s mostly repeat the response items,
so only their token counts are used. Older rollouts without the wrapper are read too.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .atif import IMAGE_PLACEHOLDER
from .sessions import (
    Parsed, SessionAdapter, SessionFile, find_files, first_prompt, join, new_step, read_lines, seconds,
    select,
)

SESSION_COLUMNS = {
    "prompt": str,  # the first thing the user asked
    "cwd": str,
    "git_branch": str,
    "git_commit": str,
    "repository_url": str,
    "model": str,
    "reasoning_effort": str,
    "cli_version": str,
    "originator": str,
    "started_at": str,
    "finished_at": str,
    "duration_s": float,
    "n_steps": int,
    "n_tool_errors": int,
    "input_tokens": int,
    "cached_input_tokens": int,
    "output_tokens": int,
    "reasoning_tokens": int,
    "path": str,
}

RESPONSE_ITEMS = {
    "message", "reasoning", "function_call", "function_call_output", "custom_tool_call",
    "custom_tool_call_output", "local_shell_call", "web_search_call",
}
CALLS = {"function_call", "custom_tool_call", "local_shell_call", "web_search_call"}
OUTPUTS = {"function_call_output", "custom_tool_call_output"}
# User-role messages Codex injects itself, not typed by the user.
HARNESS_PREFIXES = ("<environment_context>", "<user_instructions>", "# AGENTS.md instructions", "<turn_aborted>")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _unwrap(line: dict, first: bool) -> tuple[str | None, dict, str | None]:
    """(kind, payload, timestamp) for both the wrapped and the older bare format."""
    if "payload" in line and isinstance(line["payload"], dict):
        return line.get("type"), line["payload"], line.get("timestamp")
    if line.get("type") in RESPONSE_ITEMS:
        return "response_item", line, line.get("timestamp")
    if first and "id" in line and "timestamp" in line:
        return "session_meta", line, line["timestamp"]
    return None, line, None


def _text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "input_image":
            parts.append(IMAGE_PLACEHOLDER)
        elif block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts) or None


def _arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"input": value}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    return value


def _output(value: Any) -> tuple[str | None, bool | None]:
    """A tool output's text, and whether it failed when the output says so."""
    if isinstance(value, dict):
        text = value.get("content") if "content" in value else json.dumps(value)
        return _text(text), (value["success"] is False) if "success" in value else None
    if not isinstance(value, str):
        return (None if value is None else json.dumps(value)), None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and "output" in parsed:  # older shell output: {output, metadata}
        code = (parsed.get("metadata") or {}).get("exit_code")
        return parsed["output"], (code != 0) if code is not None else None
    match = re.match(r"Exit code: (-?\d+)", value)
    return value, (int(match.group(1)) != 0) if match else None


class CodexAdapter(SessionAdapter):
    """Codex CLI sessions. Levels: project > session > step (> call)."""

    session_columns = SESSION_COLUMNS

    def parse(self, file: SessionFile) -> Parsed:
        meta: dict = {}
        context: dict = {}
        usage: dict = {}
        steps: list[dict] = []
        calls: dict[str, dict] = {}
        current: dict | None = None  # the agent step still receiving items from one response
        stamps: list[str] = []
        for i, line in enumerate(read_lines(file.path)):
            kind, item, ts = _unwrap(line, first=i == 0)
            if ts:
                stamps.append(ts)
            if kind == "session_meta":
                meta = item
            elif kind == "turn_context":
                context = {**context, **item}
            elif kind == "event_msg" and item.get("type") == "token_count":
                usage = (item.get("info") or {}).get("total_token_usage") or usage
            elif kind == "compacted":
                step = new_step("system", ts)
                step["message"] = join(["[context compacted]", item.get("message")])
                steps.append(step)
                current = None
            elif kind == "response_item":
                current = self._item(item, ts, context, steps, calls, current)
        for step in steps:
            if step["source"] == "agent":
                step["message"] = join(step.pop("_text"))
                step["reasoning"] = join(step.pop("_thinking"))
        git = meta.get("git") or {}
        started = meta.get("timestamp") or (min(stamps) if stamps else None)
        finished = max(stamps) if stamps else None
        return Parsed({
            "prompt": first_prompt(steps),
            "cwd": meta.get("cwd") or context.get("cwd"),
            "git_branch": git.get("branch"),
            "git_commit": git.get("commit_hash"),
            "repository_url": git.get("repository_url"),
            "model": context.get("model") or meta.get("model"),
            "reasoning_effort": context.get("effort"),
            "cli_version": meta.get("cli_version"),
            "originator": meta.get("originator"),
            "started_at": started,
            "finished_at": finished,
            "duration_s": seconds(started, finished),
            "n_steps": len(steps),
            "n_tool_errors": sum(1 for c in calls.values() if c["is_error"]),
            "input_tokens": usage.get("input_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": usage.get("reasoning_output_tokens"),
            "path": str(file.path),
        }, steps)

    @staticmethod
    def _item(item: dict, ts: str | None, context: dict, steps: list[dict], calls: dict[str, dict],
              current: dict | None) -> dict | None:
        """Add one response item; returns the agent step still open after it."""
        kind = item.get("type")
        if kind in OUTPUTS:
            text, failed = _output(item.get("output"))
            call = calls.get(item.get("call_id"))
            if call is not None:
                call["result"], call["is_error"] = text, failed
            elif text and steps:
                steps[-1]["observation"] = join([steps[-1]["observation"], text])
            return None  # the next agent item is a new response
        if kind == "message" and item.get("role") != "assistant":
            text = _text(item.get("content"))
            if text:
                harness = item.get("role") in ("developer", "system") or text.lstrip().startswith(HARNESS_PREFIXES)
                step = new_step("system" if harness else "user", ts)
                step["message"] = text
                steps.append(step)
            return None
        if kind not in RESPONSE_ITEMS:
            return current
        # Reasoning always starts a response, and a response has at most one message.
        if current is None or kind == "reasoning" and (current["_text"] or current["tool_calls"]) \
                or kind == "message" and current["_text"]:
            current = new_step("agent", ts, model_name=context.get("model"), _text=[], _thinking=[])
            steps.append(current)
        if kind == "reasoning":
            raw = [_text([c]) for c in item.get("content") or [] if isinstance(c, dict)]
            summary = [s.get("text") for s in item.get("summary") or [] if isinstance(s, dict)]
            current["_thinking"].extend(raw if any(raw) else summary)
        elif kind == "message":
            current["_text"].append(_text(item.get("content")))
        elif kind in CALLS:
            if kind == "local_shell_call":
                arguments = item.get("action")
            elif kind == "web_search_call":
                arguments = item.get("action") or {}
            else:
                arguments = _arguments(item.get("arguments", item.get("input")))
            call = {"id": item.get("call_id") or item.get("id"),
                    "name": item.get("name") or kind.removesuffix("_call"),
                    "arguments": arguments, "result": None, "is_error": None}
            current["tool_calls"].append(call)
            if call["id"]:
                calls[call["id"]] = call
        return current


def default_root() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"


def _head(path: Path) -> tuple[str | None, str | None]:
    """(cwd, session id) from the start of a rollout."""
    cwd = session = None
    for i, line in enumerate(read_lines(path)):
        kind, item, _ = _unwrap(line, first=i == 0)
        if kind == "session_meta":
            cwd, session = item.get("cwd") or cwd, item.get("id") or session
        elif kind == "turn_context":
            cwd = cwd or item.get("cwd")
        elif kind == "response_item" and item.get("type") == "message" and not cwd:
            found = re.search(r"<cwd>(.*?)</cwd>", _text(item.get("content")) or "")
            cwd = found.group(1) if found else None
        if (cwd and session) or i >= 50:
            break
    return cwd, session


def read_codex(path: str | Path | None = None, project: str | None = None,
               limit: int | None = None) -> CodexAdapter:
    """Codex CLI sessions.

    `path` is `~/.codex/sessions` by default (or `$CODEX_HOME/sessions`); it can also be one
    rollout `.jsonl`, any directory of them, or a glob. `project` keeps sessions whose
    working directory contains it; `limit` keeps the N most recently active.
    """
    root = Path(path).expanduser() if path is not None else default_root()
    found = []
    for f in find_files(root, "*.jsonl"):
        cwd, session = _head(f)
        if session is None:
            match = UUID.search(f.name)
            session = match.group(0) if match else f.stem
        found.append(SessionFile(f, cwd or "unknown", session))
    found = select(found, project, limit)
    if not found:
        raise FileNotFoundError(f"no Codex sessions found at {root}")
    return CodexAdapter(found)
