"""Watch agent trials as they stream in: a running progress note per trial, updated each step.

    uv run python examples/live_monitor.py example_harbor_jobs/trials --tasks wal-recovery-ordering --limit 2 --speed 30

Harbor trials are replayed in timestamp order (`--speed 30` = 30x real time). Pass
`--batch` to compute the same folds without streaming; the final rows match.
"""

import argparse
import asyncio
import json

import tracepress as tp


def clip(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def render_step(row) -> str:
    parts = [f"Agent says: {clip(row.message, 800)}"] if row.message else []
    for call in row.tool_calls:
        parts.append(f"Tool call {call['name']}: {clip(call['arguments'], 400)}")
        parts.append(f"Result: {clip(call['result'] or '', 800)}")
    return "\n".join(parts) or "(empty step)"


def pipeline(src):
    steps = (
        src.table("steps")
        .filter(lambda r: r.source == "agent")
        .map(lambda r: {"step_text": render_step(r)}, columns={"step_text": str})
    )
    return steps.group_by("trial").lm_fold(
        "progress, step_text -> progress, stuck: bool, concern",
        instructions=(
            "You are monitoring a coding agent live. `previous_progress` is your running note on what the agent "
            "has done so far. Update it (at most 3 sentences) with the newest step. Set `stuck` if the agent "
            "seems to be looping or making no headway, and give a short `concern` (or 'none')."
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
