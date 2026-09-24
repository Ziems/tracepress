from __future__ import annotations

from pathlib import Path

from trace_miner.errors import TraceMinerError


TEMPLATE_FILES = (
    "README.md",
    "AGENTS.md",
    "pipeline.yaml",
    "stages.py",
    "cookbook/summarize.yaml",
    "cookbook/pii.yaml",
    "cookbook/frustration.yaml",
    "cookbook/category.yaml",
    "cookbook/classify.yaml",
    "cookbook/alerts.yaml",
    "cookbook/client_demo.py",
    "cookbook/doom_loop.yaml",
    "cookbook/doom_loop/README.md",
    "cookbook/doom_loop_demo.py",
    "cookbook/stages.py",
    "input/trajectories.jsonl",
    "input/doom_loop_examples.jsonl",
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

    return written
