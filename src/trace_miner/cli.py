from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from trace_miner.config import load_config
from trace_miner.errors import TraceMinerError
from trace_miner.home import ensure_home, format_usage, record_command, record_run
from trace_miner.init import init_project
from trace_miner.model import format_run_profile
from trace_miner.runtime import run_config
from trace_miner.serve import serve
from trace_miner.telemetry import capture_cli_finished, command_telemetry_from_argv, reset_command_run_ids, set_command_flags


_ROOT_HELP = """\
usage: trace-miner [-h] <command> ...

YAML pipelines over JSONL traces.

commands:
  init      Create a hello-world project
  run       Run a batch pipeline
  serve     Accept records over HTTP and push alerts
  stats     Show local usage totals

Run a batch pipeline:
  trace-miner run pipeline.yaml --in traces.jsonl --out findings.jsonl

`trace-miner <command> --help` documents each command.
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    started = time.monotonic()
    command, _, flags, _ = command_telemetry_from_argv(argv, _option_strings)
    reset_command_run_ids()
    set_command_flags(flags)
    exit_code = 0
    caught: BaseException | None = None
    try:
        exit_code = _dispatch(argv)
        return exit_code
    except KeyboardInterrupt as exc:
        exit_code = 130
        caught = exc
        raise
    except SystemExit as exc:
        exit_code = _exit_code(exc)
        if exit_code != 0:
            caught = exc
        raise
    except Exception as exc:
        exit_code = 1
        caught = exc
        raise
    finally:
        capture_cli_finished(
            command=command,
            command_flags=flags,
            duration_seconds=time.monotonic() - started,
            exit_code=exit_code,
            exception=caught,
        )


def _dispatch(argv: list[str]) -> int:
    if not argv:
        print(_ROOT_HELP, end="")
        return 2
    command, *rest = argv
    if command in {"-h", "--help"}:
        print(_ROOT_HELP, end="")
        return 0
    if command == "stats":
        return _stats(rest)
    if command not in {"init", "run", "serve"}:
        print(f"trace-miner: unknown command `{command}`", file=sys.stderr)
        print("Run a pipeline with `trace-miner run <config.yaml> --in <file> --out <file>`.", file=sys.stderr)
        return 2
    ensure_home()
    if command == "init":
        return _init(rest)
    if command == "serve":
        return _serve(rest)
    return _batch(rest, prog="trace-miner run")


def _exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None or code == 0:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _option_strings(command: str) -> set[str]:
    parser = _parser_for(command)
    names: set[str] = set()
    for action in parser._actions:
        names.update(action.option_strings)
    return names


def _parser_for(command: str) -> argparse.ArgumentParser:
    if command == "init":
        return _init_parser()
    if command == "serve":
        return _serve_parser()
    if command == "stats":
        return _stats_parser()
    return _batch_parser()


def _init_parser() -> argparse.ArgumentParser:
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
    return parser


def _init(argv: list[str]) -> int:
    parser = _init_parser()
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
    print("  classify.yaml             Jev classification, one map")
    print("  alerts.yaml               alert on a planted signal")
    print("  client_demo.py            post traces and stop on an abort")
    print("  doom_loop.yaml            alert on a repeated-action doom loop")
    print("  doom_loop_demo.py         post a step window and watch for it")
    print("  stages.py                 shared Python and LM stages")
    print("  AGENTS.md                 guide for coding agents")
    print("  input/trajectories.jsonl  slim harbor traces")
    print()
    print("Run it:")
    print()
    print("  export OPENROUTER_API_KEY=...")
    if dest != Path.cwd().resolve():
        print(f"  cd {rel}")
    print("  trace-miner run summarize.yaml --in input/trajectories.jsonl --out output/summaries.jsonl")
    print()
    print("Then try pii.yaml, frustration.yaml, or category.yaml")
    print("To serve:  trace-miner serve alerts.yaml --out output/alerts.jsonl")
    print("Watch it:  python client_demo.py")
    print("Doom loop: trace-miner serve doom_loop.yaml  &&  python doom_loop_demo.py")
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


def _batch_parser(prog: str = "trace-miner") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a batch pipeline from a YAML config.",
    )
    parser.add_argument("config", type=Path, help="Pipeline YAML path.")
    parser.add_argument("--in", dest="inputs", action="append", required=True, type=Path, help="Input JSONL. Repeat to read several files in order.")
    parser.add_argument("--out", dest="output", required=True, type=Path, help="Output JSONL.")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="progress", help="Always show stage progress bars.")
    progress.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress bars.")
    parser.set_defaults(progress=None)
    return parser


def _batch(argv: list[str], *, prog: str = "trace-miner") -> int:
    parser = _batch_parser(prog)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        result = run_config(config, inputs=args.inputs, output=args.output, progress=args.progress)
    except TraceMinerError as exc:
        print(f"trace-miner: {exc}", file=sys.stderr)
        return 1
    record_run("run", result.stats, extra={"config": str(config.config_path)})
    print(f"Wrote {_display_path(args.output)}")
    print(format_run_profile(result.stats))
    return 0


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-miner serve",
        description="Accept records over HTTP and push alerts.",
    )
    parser.add_argument("config", type=Path, help="Pipeline YAML path.")
    parser.add_argument("--out", dest="output", type=Path, help="Also write finished records to this JSONL.")
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
    progress.add_argument("--progress", action="store_true", dest="progress", help="Show live counters.")
    progress.add_argument("--no-progress", action="store_false", dest="progress", help="Disable live counters.")
    parser.set_defaults(progress=None)
    return parser


def _serve(argv: list[str]) -> int:
    parser = _serve_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be from 0 to 65535.")

    try:
        result = serve(
            args.config,
            output=args.output,
            host=args.host,
            port=args.port,
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


def _stats_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="trace-miner stats",
        description="Show local usage totals from ~/.trace-miner.",
    )


def _stats(argv: list[str]) -> int:
    parser = _stats_parser()
    parser.parse_args(argv)
    ensure_home()
    print(format_usage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
