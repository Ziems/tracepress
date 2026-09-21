from __future__ import annotations

from pathlib import Path

from trace_miner.errors import TraceMinerError


TEMPLATE_FILES = (
    "README.md",
    "AGENTS.md",
    "summarize.yaml",
    "pii.yaml",
    "frustration.yaml",
    "category.yaml",
    "stream.yaml",
    "alerts.yaml",
    "client_demo.py",
    "stages.py",
    "input/trajectories.jsonl",
    ".gitignore",
)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "hello"


def init_project(dest: Path, *, force: bool = False) -> list[Path]:
    dest = dest.expanduser().resolve()
    if dest.exists() and not dest.is_dir():
        raise TraceMinerError(f"{dest} exists and is not a directory.")

    existing = [rel for rel in TEMPLATE_FILES if (dest / rel).exists()]
    if existing and not force:
        raise TraceMinerError(
            "refusing to overwrite existing files: "
            + ", ".join(existing)
            + "\nUse --force to replace them."
        )

    written: list[Path] = []
    for rel in TEMPLATE_FILES:
        source = _TEMPLATE_DIR / rel
        if not source.is_file():
            raise TraceMinerError(f"Missing bundled template file: {rel}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        written.append(target)

    for name in ("live.jsonl", "alerts.jsonl"):
        path = dest / "input" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if force or not path.exists():
            path.write_text("", encoding="utf-8")
            written.append(path)
    return written
