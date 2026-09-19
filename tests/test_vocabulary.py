from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = "mi" + "nor"
PATTERN = re.compile(rf"\b{re.escape(FORBIDDEN)}\b", re.IGNORECASE)


def test_tracked_python_files_reject_forbidden_term() -> None:
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=ROOT,
        text=True,
    )
    hits: list[str] = []
    for rel in listed.split("\0"):
        if not rel:
            continue
        path = ROOT / rel
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{rel}:{line_no}:{line.rstrip()}")

    assert not hits, (
        f"the word {FORBIDDEN!r} is not allowed in Python files:\n" + "\n".join(hits)
    )
