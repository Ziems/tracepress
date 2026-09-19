from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trace_miner.config import load_config
from trace_miner.errors import TraceMinerError
from trace_miner.init import init_project
from trace_miner.model import format_run_profile
from trace_miner.runtime import run_config, run_stream


_ROOT_HELP = """\
usage: trace-miner [-h] <command> ...

Config-driven JSONL pipelines for language-model traces.

commands:
  init      Create a hello-world project
  run       Run a batch pipeline
  stream    Follow a JSONL file as records arrive

Pass a config path to run a batch pipeline:
  trace-miner pipeline.yaml

Run `trace-miner <command> --help` for details.
"""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(_ROOT_HELP, end="")
        return 2
    command, *rest = argv
    if command in {"-h", "--help"}:
        print(_ROOT_HELP, end="")
        return 0
    if command == "init":
        return _init(rest)
    if command == "stream":
        return _stream(rest)
    if command == "run":
        return _batch(rest, prog="trace-miner run")
    return _batch(argv)


def _init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="trace-miner init",
        description="Create a hello-world Trace Miner project.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        type=Path,
        help="New project directory. Prompted if omitted, default hello-world.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing project files.",
    )
    args = parser.parse_args(argv)

    try:
        dest = args.directory if args.directory is not None else _prompt_project_name()
        init_project(dest, force=args.force)
    except TraceMinerError as exc:
        print(f"trace-miner: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130

    dest = dest.expanduser().resolve()
    rel = _display_path(dest)
    print(f"Created a hello-world project in {rel}/")
    print()
    print("  summarize.yaml            summarize and diagnose")
    print("  pii.yaml                  PII detection")
    print("  frustration.yaml          user frustration")
    print("  category.yaml             domain and outcome")
    print("  stages.py                 shared Python and LM stages")
    print("  AGENTS.md                 guide for coding agents")
    print("  input/trajectories.jsonl  slim harbor traces")
    print()
    print("Run it:")
    print()
    print("  export OPENROUTER_API_KEY=...")
    if dest != Path.cwd().resolve():
        print(f"  cd {rel}")
    print("  trace-miner summarize.yaml")
    print()
    print("Then try pii.yaml, frustration.yaml, or category.yaml")
    return 0


_DEFAULT_PROJECT_NAME = "hello-world"


def _prompt_project_name() -> Path:
    if not sys.stdin.isatty():
        return Path(_DEFAULT_PROJECT_NAME)
    while True:
        try:
            name = input(f"Project name [{_DEFAULT_PROJECT_NAME}]: ").strip()
        except EOFError:
            return Path(_DEFAULT_PROJECT_NAME)
        if not name:
            return Path(_DEFAULT_PROJECT_NAME)
        if name in {".", ".."}:
            print("Project name must be a new directory.", file=sys.stderr)
            continue
        return Path(name)


def _batch(argv: list[str], *, prog: str = "trace-miner") -> int:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("config", type=Path)
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="progress", help="Always show stage progress bars.")
    progress.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress bars.")
    parser.set_defaults(progress=None)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        result = run_config(config, progress=args.progress)
    except TraceMinerError as exc:
        print(f"trace-miner: {exc}", file=sys.stderr)
        return 1
    for pipeline in config.pipelines:
        print(f"Wrote {_display_path(pipeline.output)}")
    print(format_run_profile(result.stats))
    return 0


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _stream(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="trace-miner stream")
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Override the checkpoint path.")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--from-start", action="store_true", help="Ignore any checkpoint and start at byte 0.")
    start.add_argument("--from-end", action="store_true", help="Ignore any checkpoint and start at the current end.")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Ignore existing checkpoint state.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds to sleep when no complete line is available.")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="progress", help="Show streaming counters.")
    progress.add_argument("--no-progress", action="store_false", dest="progress", help="Disable streaming counters.")
    parser.set_defaults(progress=None)
    args = parser.parse_args(argv)

    try:
        result = run_stream(
            args.config,
            checkpoint_path=args.checkpoint,
            from_start=args.from_start,
            from_end=args.from_end,
            reset_checkpoint=args.reset_checkpoint,
            poll_interval=args.poll_interval,
            progress=args.progress,
        )
    except KeyboardInterrupt:
        return 130
    except TraceMinerError as exc:
        print(f"trace-miner: {exc}", file=sys.stderr)
        return 1
    print(format_run_profile(result.stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
