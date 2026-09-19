from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace_miner.model import RunStats


HOME_ENV = "TRACE_MINER_HOME"
HOME_DIRNAME = ".trace-miner"
STATS_VERSION = 1


def home_dir() -> Path:
    override = os.environ.get(HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / HOME_DIRNAME


def cache_dir() -> Path:
    return home_dir() / "cache"


def stats_path() -> Path:
    return home_dir() / "stats.json"


def runs_path() -> Path:
    return home_dir() / "runs.jsonl"


def display_home() -> str:
    root = home_dir()
    default = Path.home() / HOME_DIRNAME
    if root == default or root == default.resolve():
        return f"~/{HOME_DIRNAME}"
    return str(root)


def ensure_home() -> Path:
    root = home_dir()
    cache_dir().mkdir(parents=True, exist_ok=True)
    return root


def configure_lm_cache() -> Path:
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["DSPY_CACHEDIR"] = str(cache)
    try:
        import dspy

        if hasattr(dspy, "configure_cache"):
            dspy.configure_cache(disk_cache_dir=str(cache))
    except ImportError:
        pass
    return cache


def load_stats() -> dict[str, Any]:
    path = stats_path()
    if not path.is_file():
        return _empty_stats()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_stats()
    if not isinstance(raw, dict):
        return _empty_stats()
    return _normalize_stats(raw)


def record_command(command: str, extra: dict[str, Any] | None = None) -> None:
    _record(command, stats=None, extra=extra)


def record_run(command: str, stats: RunStats, extra: dict[str, Any] | None = None) -> None:
    _record(command, stats=stats, extra=extra)


def format_usage(stats: dict[str, Any] | None = None) -> str:
    payload = load_stats() if stats is None else stats
    commands = payload.get("commands") if isinstance(payload.get("commands"), dict) else {}
    runs = int(payload.get("runs") or 0)
    if runs == 0 and not any(int(count or 0) for count in commands.values()):
        return f"No usage recorded yet. Local data lives in {display_home()}"

    lines = [display_home()]
    lm_calls = int(payload.get("lm_calls") or 0)
    if lm_calls:
        tokens = int(payload.get("lm_prompt_tokens") or 0) + int(payload.get("lm_completion_tokens") or 0)
        cost = float(payload.get("lm_cost_usd") or 0.0)
        lines.append(f"{runs} runs  {lm_calls} LM calls  {tokens:,} tokens  ${cost:.4f}")
    else:
        lines.append(f"{runs} runs")

    parts = []
    for name in ("init", "run", "stream", "replay"):
        count = int(commands.get(name) or 0)
        if count:
            parts.append(f"{name} {count}")
    if parts:
        lines.append("  " + "  ".join(parts))
    return "\n".join(lines)


def _record(command: str, stats: RunStats | None, extra: dict[str, Any] | None) -> None:
    try:
        ensure_home()
        payload = load_stats()
        now = _now()
        if not payload.get("created_at"):
            payload["created_at"] = now
        payload["updated_at"] = now
        commands = payload.setdefault("commands", {})
        commands[command] = int(commands.get(command) or 0) + 1
        event: dict[str, Any] = {"ts": now, "command": command}
        if extra:
            event.update(extra)
        if stats is not None:
            payload["runs"] = int(payload.get("runs") or 0) + 1
            payload["records_read"] = int(payload.get("records_read") or 0) + stats.records_read
            payload["emitted"] = int(payload.get("emitted") or 0) + stats.emitted
            payload["retired"] = int(payload.get("retired") or 0) + stats.retired
            payload["failed"] = int(payload.get("failed") or 0) + stats.failed
            payload["lm_calls"] = int(payload.get("lm_calls") or 0) + stats.lm_calls
            payload["lm_prompt_tokens"] = int(payload.get("lm_prompt_tokens") or 0) + stats.lm_prompt_tokens
            payload["lm_completion_tokens"] = int(payload.get("lm_completion_tokens") or 0) + stats.lm_completion_tokens
            payload["lm_cost_usd"] = float(payload.get("lm_cost_usd") or 0.0) + stats.lm_cost_usd
            payload["elapsed_s"] = float(payload.get("elapsed_s") or 0.0) + stats.elapsed_s
            event.update(
                {
                    "records_read": stats.records_read,
                    "emitted": stats.emitted,
                    "retired": stats.retired,
                    "failed": stats.failed,
                    "lm_calls": stats.lm_calls,
                    "lm_prompt_tokens": stats.lm_prompt_tokens,
                    "lm_completion_tokens": stats.lm_completion_tokens,
                    "lm_cost_usd": stats.lm_cost_usd,
                    "elapsed_s": stats.elapsed_s,
                }
            )
        _write_json(stats_path(), payload)
        _append_jsonl(runs_path(), event)
    except OSError:
        return


def _empty_stats() -> dict[str, Any]:
    return {
        "version": STATS_VERSION,
        "created_at": None,
        "updated_at": None,
        "commands": {},
        "runs": 0,
        "records_read": 0,
        "emitted": 0,
        "retired": 0,
        "failed": 0,
        "lm_calls": 0,
        "lm_prompt_tokens": 0,
        "lm_completion_tokens": 0,
        "lm_cost_usd": 0.0,
        "elapsed_s": 0.0,
    }


def _normalize_stats(raw: dict[str, Any]) -> dict[str, Any]:
    payload = _empty_stats()
    payload.update({key: raw[key] for key in payload if key in raw})
    commands = raw.get("commands")
    payload["commands"] = dict(commands) if isinstance(commands, dict) else {}
    payload["version"] = STATS_VERSION
    return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
