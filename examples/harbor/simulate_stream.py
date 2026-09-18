from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Append harbor trajectories to a live JSONL input.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent / "input" / "trajectories.jsonl",
        help="Source JSONL file to replay.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).parent / "input" / "live.jsonl",
        help="Live JSONL file to append to.",
    )
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds to wait between appended records.")
    parser.add_argument("--limit", type=int, default=1, help="Number of records to append. Use 0 for all records.")
    parser.add_argument("--reset", action="store_true", help="Truncate the target before appending.")
    parser.add_argument("--reset-only", action="store_true", help="Truncate the target and exit without appending.")
    args = parser.parse_args()

    if args.interval < 0:
        parser.error("--interval must be non-negative.")
    if args.limit < 0:
        parser.error("--limit must be non-negative.")

    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.reset_only:
        target.write_text("", encoding="utf-8")
        print(f"reset live input {target}", flush=True)
        return 0
    if args.reset:
        target.write_text("", encoding="utf-8")

    appended = 0
    with source.open("r", encoding="utf-8") as source_handle, target.open("a", encoding="utf-8") as target_handle:
        for line_number, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            if args.limit and appended >= args.limit:
                break
            target_handle.write(line if line.endswith("\n") else line + "\n")
            target_handle.flush()
            appended += 1
            print(f"appended trajectory {appended} from source line {line_number} -> {target}", flush=True)
            if args.interval and (not args.limit or appended < args.limit):
                time.sleep(args.interval)

    print(f"done: appended {appended} trajectories", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
