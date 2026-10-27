"""Watch agent trials as they stream in: a running progress note per trial, updated each step.

    uv run python examples/live_monitor.py example_harbor_jobs/trials --tasks wal-recovery-ordering --limit 2 --speed 30

Harbor trials are replayed in timestamp order (`--speed 30` = 30x real time). Pass
`--batch` to compute the same folds without streaming; the final rows match.
"""

import argparse
import asyncio

import tracepress as tp
from tracepress.pipelines.harbor import STEP_INPUTS, clip


def pipeline(src):
    return src.table("steps").group_by("trial").lm_fold(
        f"progress, {STEP_INPUTS} -> progress, stuck: bool, concern",
        instructions=(
            "You are monitoring a coding agent live. The first step is the task it was given. "
            "`previous_progress` is your running note on what the agent has done so far. Update it (at most "
            "3 sentences) with the newest step. Set `stuck` if the agent seems to be looping or making no "
            "headway, and give a short `concern` (or 'none')."
        ),
        emit="update",
    )


async def watch(table) -> None:
    async for event in table.events():
        if isinstance(event, tp.Closed):
            continue
        row = event.row
        tag = "FINAL" if row.final else f"v{row.version:<3}"
        flag = "STUCK " if row.stuck else ""
        print(f"[{row.trial} {tag}] {flag}{clip(row.progress, 160)}  | concern: {clip(row.concern, 80)}")
    print(tp.last_run())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--lm", default="openrouter:openai/gpt-6-luna",
                        help="provider:model (openrouter:..., portkey:<slug>/<model>) or a litellm model string")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--speed", type=float, help="replay speed multiplier (default: as fast as possible)")
    parser.add_argument("--batch", action="store_true", help="no streaming; print the final rows")
    args = parser.parse_args()
    tp.configure(lm=args.lm)

    src = tp.read.harbor(args.path, tasks=args.tasks, limit=args.limit)
    if args.batch:
        for row in pipeline(src).collect():
            print(f"[{row.trial} FINAL] {'STUCK ' if row.stuck else ''}{clip(row.progress, 160)}  | concern: {clip(row.concern, 80)}")
        print(tp.last_run())
    else:
        asyncio.run(watch(pipeline(tp.replay(src, speed=args.speed))))


if __name__ == "__main__":
    main()
