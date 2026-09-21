from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trace_miner.config import load_config
from trace_miner.errors import TraceMinerError
from trace_miner.home import ensure_home, format_usage, record_command, record_run
from trace_miner.init import init_project
from trace_miner.model import format_run_profile
from trace_miner.replay import replay_jsonl
from trace_miner.runtime import run_config, run_stream
from trace_miner.serve import serve, token_from_env


_ROOT_HELP = """\
usage: trace-miner [-h] <command> ...

YAML pipelines over JSONL traces.

commands:
  init      Create a hello-world project
  run       Run a batch pipeline
  stream    Follow a JSONL file as records arrive
  serve     Accept records over HTTP and push alerts
  replay    Append JSONL records into a live input file
  stats     Show local usage totals

Run a batch pipeline:
  trace-miner pipeline.yaml

`trace-miner <command> --help` documents each command.
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
    if command == "stats":
        return _stats(rest)
    ensure_home()
    if command == "init":
        return _init(rest)
    if command == "stream":
        return _stream(rest)
    if command == "serve":
        return _serve(rest)
    if command == "replay":
        return _replay(rest)
    if command == "run":
        return _batch(rest, prog="trace-miner run")
    return _batch(argv)


def _init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="trace-miner init",
        description="Create a hello-world project with example pipelines and traces.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        type=Path,
        help="New project directory. Prompted if omitted; default hello-world.",
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
        record_command("init", extra={"directory": str(dest.expanduser().resolve())})
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
    print("  stream.yaml               live-tail summarize")
    print("  alerts.yaml               alert on a planted signal")
    print("  client_demo.py            replay traces and stop on an abort")
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
    print("To stream: trace-miner stream stream.yaml")
    print("Feed it:   trace-miner replay input/trajectories.jsonl --into input/live.jsonl --reset")
    print("To serve:  trace-miner serve alerts.yaml")
    print("Watch it:  python client_demo.py")
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
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a batch pipeline from a YAML config. Same as `trace-miner pipeline.yaml`.",
    )
    parser.add_argument("config", type=Path, help="Pipeline YAML path.")
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
    record_run("run", result.stats, extra={"config": str(config.config_path)})
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


def _replay(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="trace-miner replay",
        description="Append JSONL records into a live input file.",
    )
    parser.add_argument("source", type=Path, help="Existing JSONL file to replay.")
    parser.add_argument("--into", type=Path, required=True, dest="dest", help="Live JSONL file to append to.")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds to wait between appended records. Default: 3.")
    parser.add_argument("--limit", type=int, default=0, help="Number of records to append. 0, the default, appends all records.")
    parser.add_argument("--reset", action="store_true", help="Truncate the target before appending.")
    parser.add_argument("--reset-only", action="store_true", help="Truncate the target and exit without appending.")
    args = parser.parse_args(argv)

    if args.interval < 0:
        parser.error("--interval must be non-negative.")
    if args.limit < 0:
        parser.error("--limit must be non-negative.")

    def log(message: str) -> None:
        print(message, flush=True)

    try:
        replay_jsonl(
            args.source,
            args.dest,
            interval=args.interval,
            limit=args.limit,
            reset=args.reset,
            reset_only=args.reset_only,
            log=log,
        )
        record_command(
            "replay",
            extra={
                "source": str(args.source.expanduser().resolve()),
                "dest": str(args.dest.expanduser().resolve()),
            },
        )
    except KeyboardInterrupt:
        return 130
    except TraceMinerError as exc:
        print(f"trace-miner: {exc}", file=sys.stderr)
        return 1
    return 0


def _stream(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="trace-miner stream",
        description="Follow a JSONL input as records are appended.",
    )
    parser.add_argument("config", type=Path, help="Pipeline YAML path.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint file. Default: .trace-miner/ beside the config.")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--from-start", action="store_true", help="Start at byte 0.")
    start.add_argument("--from-end", action="store_true", help="Skip lines already in the file.")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Ignore saved checkpoint state and start at byte 0.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds to wait when no complete line is available. Default: 1.",
    )
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
    record_run("stream", result.stats, extra={"config": str(args.config.expanduser().resolve())})
    print(format_run_profile(result.stats))
    return 0


def _serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="trace-miner serve",
        description="Run a pipeline stream and accept records over HTTP.",
    )
    parser.add_argument("config", type=Path, help="Pipeline YAML path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Default: 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port. Default: 8765.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint file. Default: .trace-miner/ beside the config.")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--from-start", action="store_true", help="Start at byte 0.")
    start.add_argument("--from-end", action="store_true", help="Skip lines already in the file.")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Ignore saved checkpoint state and start at byte 0.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds to wait when no complete line is available. Default: 1.",
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="progress", help="Show streaming counters.")
    progress.add_argument("--no-progress", action="store_false", dest="progress", help="Disable streaming counters.")
    parser.set_defaults(progress=None)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be from 0 to 65535.")

    try:
        result = serve(
            args.config,
            host=args.host,
            port=args.port,
            token=token_from_env(),
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
    record_run("serve", result.stats, extra={"config": str(args.config.expanduser().resolve())})
    print(format_run_profile(result.stats))
    return 0


def _stats(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="trace-miner stats",
        description="Show local usage totals from ~/.trace-miner.",
    )
    parser.parse_args(argv)
    ensure_home()
    print(format_usage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
