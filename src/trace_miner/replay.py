from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from trace_miner.errors import TraceMinerError


def replay_jsonl(
    source: Path,
    dest: Path,
    *,
    interval: float = 3.0,
    limit: int = 0,
    reset: bool = False,
    reset_only: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    if interval < 0:
        raise TraceMinerError("--interval must be non-negative.")
    if limit < 0:
        raise TraceMinerError("--limit must be non-negative.")

    dest = dest.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if reset_only or reset:
        dest.write_text("", encoding="utf-8")
        if reset_only:
            log(f"reset live input {dest}")
            return 0

    source = source.expanduser().resolve()
    if not source.is_file():
        raise TraceMinerError(f"{source} is not a file.")

    appended = 0
    pending_sleep = False
    with source.open("r", encoding="utf-8") as source_handle, dest.open("a", encoding="utf-8") as dest_handle:
        for line_number, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            if limit and appended >= limit:
                break
            if pending_sleep and interval:
                sleep(interval)
            dest_handle.write(line if line.endswith("\n") else line + "\n")
            dest_handle.flush()
            appended += 1
            log(f"appended trajectory {appended} from source line {line_number} -> {dest}")
            pending_sleep = True

    log(f"done: appended {appended} trajectories")
    return appended
