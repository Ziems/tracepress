"""Hierarchical failure analysis over a Harbor job: steps -> trials -> tasks -> benchmark.

    uv run python examples/failure_modes.py example_harbor_jobs/trials --tasks wal-recovery-ordering gsea-proteomics

Each trial's summary sees the task it was given and the verifier's failed tests, so failures
are explained by what actually broke. Task-level failures say what the agent was supposed to
do, what it did instead, and how many failed runs it affected.
"""

import argparse
import json
from pathlib import Path

import dspy

import tracepress as tp


def build(src):
    """The analysis as lazy tables: (step notes, per trial, per task, overall)."""
    # Trial- and task-level facts, as small tables joined onto the rows that need them.
    failed_tests = (
        src.tests.filter(lambda r: r.status != "passed")
        .group_by("trial")
        .reduce(lambda rs: {"failed_tests": ", ".join(r.name.split("::")[-1] for r in rs)},
                columns={"failed_tests": str})
    )
    failed_runs = src.trials.group_by("task").reduce(
        lambda rs: {"failed_runs": sum((r.reward or 0) < 1 for r in rs)}, columns={"failed_runs": int})

    notes = src.steps.lm_map(
        "step -> what_happened, is_mistake: bool",
        instructions="Summarize this step of a coding agent's run in one sentence, and say whether it was a mistake. "
                     "For a user step (the first one is the task), say what the agent was asked to do.",
    )

    # reward and failed_tests have one value per trial, so the reduce gets each once.
    per_trial = notes.join(src.trials).join(failed_tests).group_by("trial").lm_reduce(
        "instruction, reward, failed_tests, what_happened, is_mistake -> summary, failure_mode",
        instructions="One agent run, step by step, and the task it was given. Reward 1.0 means it passed, and "
                     "failed_tests are the verifier's tests that failed (empty when none did). Summarize what it "
                     "was asked to do and how it went, and explain the failure using the failed tests. "
                     "For passes, use 'none' as failure_mode.",
    )

    per_task = per_trial.join(failed_runs).group_by("task").lm_reduce(
        "task, failed_runs, summary, failure_mode, reward -> common_failures: list[str], what_passing_runs_did_differently",
        instructions="Compare these runs of one task. Write each failure as what the agent was supposed to do and "
                     "what it did instead, so it makes sense to someone who hasn't read the task. End each failure "
                     "with how many of the failed_runs it affected, e.g. (3 of 4 failed runs).",
        module=dspy.ChainOfThought,
    )

    overall = per_task.group_by().lm_reduce(
        "task, common_failures -> top_failure_modes: list[str]",
        instructions="Across all tasks, list the most common failure modes for this agent, most frequent first.",
        module=dspy.ChainOfThought,
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
    if args.replay:
        faults = {"shuffle_within": 8, "duplicates": 0.05, "drop_closes": 0.2} if args.faults else {}
        src = tp.replay(src, speed=args.speed, **faults)

    _, per_trial, per_task, overall = build(src)
    trial_rows, task_rows, overall_rows = tp.collect(per_trial, per_task, overall)
    stats = tp.last_run()

    for row in overall_rows:
        print("\n=== Top failure modes ===")
        for mode in row.top_failure_modes:
            print(f"  - {mode}")
    for row in task_rows:
        print(f"\n--- {row.task} ---")
        for mode in row.common_failures:
            print(f"  - {mode}")
        print(f"  passes: {row.what_passing_runs_did_differently}")
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
