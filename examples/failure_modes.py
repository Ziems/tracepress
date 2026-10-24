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
from tracepress.pipelines.harbor import render_step


def step_text(row) -> str:
    """Agent steps rendered compactly; the user step is the task the agent was given."""
    return render_step(row) if row.source == "agent" else f"Task given to the agent: {row.message}"


def lookups(trials, tests):
    """Trial-level facts to attach to rows: there are no joins yet, so these are plain dicts.

    Takes collected `trials` and `tests` rows (collect them with `.collect()`, or with
    `await ....acollect()` in Jupyter).
    """
    rewards = {r.trial: r.reward for r in trials}
    failed_runs: dict[str, int] = {}
    for r in trials:
        failed_runs[r.task] = failed_runs.get(r.task, 0) + int((r.reward or 0) < 1)
    failed_tests: dict[str, list[str]] = {}
    for t in tests:
        if t.status != "passed":
            failed_tests.setdefault(t.trial, []).append(t.name.split("::")[-1])
    return rewards, failed_runs, failed_tests


def build(src, trials, tests):
    """The analysis as lazy tables: (step notes, per trial, per task, overall)."""
    rewards, failed_runs, failed_tests = lookups(trials, tests)

    steps = src.table("steps").map(lambda r: {"step_text": step_text(r)}, columns={"step_text": str})
    notes = steps.lm_map(
        "step_text -> what_happened, is_mistake: bool",
        instructions="Summarize this step of a coding agent's run in one sentence, and say whether it was a mistake. "
                     "For the task step, say what the agent was asked to do.",
    )

    # Grouping by trial *and* reward / failed tests makes them scalar inputs to the reduce.
    per_trial = (
        notes.map(lambda r: {"reward": rewards.get(r.trial),
                             "failed_tests": ", ".join(failed_tests.get(r.trial, [])) or "none"},
                  columns={"reward": float, "failed_tests": str})
        .group_by("trial", "reward", "failed_tests")
        .lm_reduce(
            "reward, failed_tests, what_happened, is_mistake -> summary, failure_mode",
            instructions="One agent run, step by step; the first step is the task it was given. Reward 1.0 "
                         "means it passed, and failed_tests are the verifier's tests that failed. Summarize what "
                         "it was asked to do and how it went, and explain the failure using the failed tests. "
                         "For passes, use 'none' as failure_mode.",
        )
    )

    per_task = (
        per_trial.map(lambda r: {"failed_runs": failed_runs.get(r.task, 0)}, columns={"failed_runs": int})
        .group_by("task", "failed_runs")
        .lm_reduce(
            "task, failed_runs, summary, failure_mode, reward -> common_failures: list[str], what_passing_runs_did_differently",
            instructions="Compare these runs of one task. Write each failure as what the agent was supposed to do and "
                         "what it did instead, so it makes sense to someone who hasn't read the task. End each failure "
                         "with how many of the failed_runs it affected, e.g. (3 of 4 failed runs).",
            module=dspy.ChainOfThought,
        )
    )

    overall = per_task.group_by("dataset").lm_reduce(
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
    trials, tests = tp.collect(src.table("trials"), src.table("tests"))

    if args.replay:
        faults = {"shuffle_within": 8, "duplicates": 0.05, "drop_closes": 0.2} if args.faults else {}
        src = tp.replay(src, speed=args.speed, **faults)

    _, per_trial, per_task, overall = build(src, trials, tests)
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
