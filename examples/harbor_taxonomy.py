"""The Terminal-Bench failure taxonomies over a Harbor job: per-trial failure modes and per-command errors.

    uv run python examples/harbor_taxonomy.py example_harbor_jobs/trials --tasks wal-recovery-ordering --limit 4
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import tracepress as tp
from tracepress.pipelines import harbor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-trials", action="store_true", help="judge passing trials too")
    parser.add_argument("--out", default="out/harbor_taxonomy.json")
    args = parser.parse_args()

    src = tp.read.harbor(args.path, tasks=args.tasks, limit=args.limit)
    trajectories = harbor.trajectory_taxonomy(src, failed_only=not args.all_trials)
    commands = harbor.step_taxonomy(src, failed_only=not args.all_trials)
    mode_rows, command_rows = tp.collect(trajectories, commands)
    stats = tp.last_run()

    wide = harbor.trajectory_labels_wide(mode_rows)
    print(f"\n=== Trajectory failure modes ({len(wide)} trials) ===")
    for cls in harbor.TRAJECTORY_CLASSES:
        print(f"{cls}: {sum(w[cls] for w in wide)}/{len(wide)}")
        for mode_id, mode in harbor.TRAJECTORY_MODES.items():
            if mode.cls == cls:
                print(f"  {mode.name}: {sum(w.get(mode_id, False) for w in wide)}")

    print(f"\n=== Command failures ({len(command_rows)}) ===")
    for (cat, sub), n in Counter((r.error_category, r.error_subcategory) for r in command_rows).most_common(15):
        print(f"  {n:4d}  {cat} / {sub}")
    print(f"\n{stats}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "trials": wide,
        "commands": [{k: v for k, v in r.to_dict().items() if k != "command_text"} for r in command_rows],
    }, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
