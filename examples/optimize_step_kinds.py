"""Tune an LM step's prompt against labels with GEPA: classify each step from its summary.

    uv run python examples/optimize_step_kinds.py example_harbor_jobs/trials \
        --tasks mvcc-lsm-compaction embedding-drift-monitor

Each step gets a label from a fixed rule over its real tool calls: `edit` (a Write or Edit,
even of a test file), `test` (any Bash that runs Python), `explore` (any other Bash or a
Read) or `other` (no tool calls). The step being tuned only sees the step's one-sentence
summary, so a plain "What kind of step is this?" has to guess the rule's edge cases. GEPA
learns them from its misses ("Expected kind='edit', got 'test'").

On the two tasks above (160 steps) one run took held-out accuracy from 0.62 to 0.69 for
about 2 cents, learning that writing tests is an edit.
"""

import argparse
import collections
import re

import tracepress as tp

# The same step-summary prompt as the Trace Science post, so reruns come from the cache.
STEP_SIG = "instruction, source, message, tool_calls -> step_summary"
STEP_INS = ("This is one step of a coding agent working on the task in `instruction`. In one "
            "short sentence (at most 20 words), say who did what (the agent, the user or the "
            "harness) and what came of it, in words a reader who hasn't seen the code can follow.")


def kind(step) -> str:
    """The label: what the step's tool calls actually did."""
    calls = step.tool_calls or []
    names = {c["name"] for c in calls}
    bash = " ".join(c["arguments"].get("command", "") for c in calls if c["name"] == "Bash")
    if not calls:
        return "other"
    if names & {"Write", "Edit"}:
        return "edit"  # writing a test file is still an edit
    if re.search(r"\bpython3?\b|pytest", bash):
        return "test"  # running any Python counts as testing
    return "explore"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--lm", default="openrouter:openai/gpt-6-luna",
                        help="provider:model (openrouter:..., portkey:<slug>/<model>) or a litellm model string")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--budget", type=int, default=400, help="example runs GEPA may make")
    args = parser.parse_args()
    tp.configure(lm=args.lm, max_concurrency=16)

    job = tp.read.harbor(args.path, tasks=args.tasks)
    step_summaries = job.steps.join(job.trials, on="trial").lm_map(STEP_SIG, instructions=STEP_INS)

    labels = [{"trial": s.trial, "step": s.step, "kind": kind(s)} for s in job.steps.collect()]
    print("labels:", dict(collections.Counter(label["kind"] for label in labels)))

    classify = step_summaries.lm_map(
        "step_summary -> kind: Literal['explore', 'edit', 'test', 'other']",
        instructions="What kind of step is this?")
    tuned = classify.optimize(labels, metric="kind", budget=args.budget, val_frac=0.3, seed=0)

    print("\n", tuned.optimize_report, sep="")
    print("\nBefore:", tuned.optimize_report.before)
    print("\nAfter:\n" + tuned.instructions)


if __name__ == "__main__":
    main()
