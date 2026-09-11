from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mine_trace.errors import MineTraceError
from mine_trace.runtime import run_config


def main(argv: list[str] | None = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
