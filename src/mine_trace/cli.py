from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mine_trace.errors import MineTraceError
from mine_trace.runtime import run_config, run_stream


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "stream":
        return _stream(argv[1:])
    return _batch(argv)


def _batch(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mine-trace")
    parser.add_argument("config", type=Path)
    parser.add_argument("--manifest", type=Path, help="Write the run manifest as JSON.")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="progress", help="Always show stage progress bars.")
    progress.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress bars.")
    parser.set_defaults(progress=None)
    args = parser.parse_args(argv)

    try:
        result = run_config(args.config, progress=args.progress)
    except MineTraceError as exc:
        print(f"mine-trace: {exc}", file=sys.stderr)
        return 1

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "skipped_groups": result.manifest.skipped_groups,
                    "stats": result.stats.__dict__,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


def _stream(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mine-trace stream")
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Override the checkpoint path.")
    parser.add_argument("--from-start", action="store_true", help="Ignore any checkpoint and start at byte 0.")
    parser.add_argument("--from-end", action="store_true", help="Ignore any checkpoint and start at the current end.")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Ignore existing checkpoint state.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds to sleep when no complete line is available.")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", dest="progress", help="Show streaming counters.")
    progress.add_argument("--no-progress", action="store_false", dest="progress", help="Disable streaming counters.")
    parser.set_defaults(progress=None)
    args = parser.parse_args(argv)

    try:
        run_stream(
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
    except MineTraceError as exc:
        print(f"mine-trace: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
