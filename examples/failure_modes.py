"""Hierarchical failure analysis over a Harbor job: steps -> trials -> tasks -> benchmark.

    uv run python examples/failure_modes.py example_harbor_jobs/trials --tasks wal-recovery-ordering gsea-proteomics
"""

import argparse
import json
from pathlib import Path

import tracepress as tp
from tracepress.pipelines.harbor import render_step


def build(src, rewards: dict[str, float]):
    """The analysis as lazy tables: (step notes, per trial, per task, overall).

    `rewards` maps trial name to reward. There are no joins yet, so each step gets its
    trial's reward through this lookup.
    """
    steps = (
        src.table("steps")
        .filter(lambda r: r.source == "agent")
        .map(lambda r: {"step_text": render_step(r), "reward": rewards.get(r.trial)}, columns={"step_text": str, "reward": float})
    )

    notes = steps.lm_map(
        "step_text -> what_happened, is_mistake: bool",
        instructions="Summarize what the coding agent did in this step in one sentence, and say whether it was a mistake.",
    )

    # Grouping by trial *and* reward makes the reward a scalar input to the reduce.
    per_trial = notes.group_by("trial", "reward").lm_reduce(
        "reward, what_happened, is_mistake -> summary, failure_mode",
        instructions=(
            "These are the steps of one agent attempt at a task, in order; reward 1.0 means it passed "
            "and 0.0 means it failed. Summarize the attempt in 2-3 sentences. For failures, name the root "
            "cause in a short phrase as failure_mode; for passes, use 'none'."
        ),
    )

    per_task = per_trial.group_by("task").lm_reduce(
        "task, summary, failure_mode, reward -> common_failures: list[str], what_separates_passes: str",
        instructions="Compare the attempts at this task. List the recurring failure modes, and explain what the passing attempts did differently.",
    )

    overall = per_task.group_by("dataset").lm_reduce(
        "task, common_failures -> top_failure_modes: list[str]",
        instructions="Across all tasks, list the most common failure modes for this agent, most frequent first.",
    )
    return notes, per_trial, per_task, overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--lm", default="openrouter:openai/gpt-6-luna",
                        help="provider:model (openrouter:..., portkey:<slug>/<model>) or a litellm model string")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="out/failure_modes.json")
    parser.add_argument("--replay", action="store_true", help="feed the steps in as a simulated stream")
    parser.add_argument("--speed", type=float, help="replay speed multiplier (default: as fast as possible)")
    parser.add_argument("--faults", action="store_true", help="replay with shuffled, duplicated rows and dropped closes")
    args = parser.parse_args()
    tp.configure(lm=args.lm)

    src = tp.read.harbor(args.path, tasks=args.tasks, limit=args.limit)
    rewards = {row.trial: row.reward for row in src.table("trials").collect()}

    if args.replay:
        faults = {"shuffle_within": 8, "duplicates": 0.05, "drop_closes": 0.2} if args.faults else {}
        src = tp.replay(src, speed=args.speed, **faults)

    _, per_trial, per_task, overall = build(src, rewards)
    trial_rows, task_rows, overall_rows = tp.collect(per_trial, per_task, overall)
    stats = tp.last_run()

    for row in overall_rows:
        print(f"\n=== {row.dataset}: top failure modes ===")
        for mode in row.top_failure_modes:
            print(f"  - {mode}")
    for row in task_rows:
        print(f"\n--- {row.task} ---")
        for mode in row.common_failures:
            print(f"  - {mode}")
        print(f"  passes: {row.what_separates_passes}")
    print(f"\n{len(trial_rows)} trials, {len(task_rows)} tasks. {stats}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "trials": [r.to_dict() for r in trial_rows],
        "tasks": [r.to_dict() for r in task_rows],
        "overall": [r.to_dict() for r in overall_rows],
    }, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
