# Built-in pipelines

## Harbor: Terminal-Bench failure taxonomies

`tp.pipelines.harbor` has the two failure taxonomies from the Terminal-Bench 2.0 paper
([arXiv:2601.11868](https://arxiv.org/abs/2601.11868), §4.4–4.5). Both run on any
`tp.read.harbor(...)` source, including a replayed one.

```python
from tracepress.pipelines import harbor

src = tp.read.harbor("jobs/my-job")
modes, commands = tp.collect(harbor.trajectory_taxonomy(src), harbor.step_taxonomy(src))
per_trial = harbor.trajectory_labels_wide(modes)
```

### Trajectory taxonomy

`trajectory_taxonomy(src, failed_only=True, modes=TRAJECTORY_MODES)` judges each failed trial
against 9 failure modes in three classes:

| Class | Modes |
|---|---|
| Execution | disobey specification, step repetition, unaware of termination conditions |
| Coherence | reasoning-action mismatch, context loss, task derailment |
| Verification | premature termination, no or incorrect verification, weak verification |

**How it works:**
1. Each agent step is summarized into a short digest. The digest keeps the actions, the
   outputs, any claims and plans, and the checks the agent ran.
2. The digests are joined, in order, into one transcript per trial. The task instruction goes
   in word for word. This lets long runs fit in one prompt while the judges still see the
   whole run.
3. Each mode gets its own judge, as in the paper. The judge follows that mode's rubric, made
   of a decision procedure and exclusions.

**Output:** one row per (trial, mode), with `failure_mode`, `mode_class`, `matched`,
`rationale` (citing step numbers) and `transcript`. `trajectory_labels_wide(rows)` turns that
into one dict per trial, with a `<mode>` flag per mode and an `execution` / `coherence` /
`verification` flag that's true if any of that class's modes matched.

**Cost:** one call per agent step, plus 9 per trial.

### Step taxonomy

`step_taxonomy(src, failed_only=False, taxonomy=COMMAND_TAXONOMY)` works in two stages for
each tool call:
1. It checks whether the command failed.
2. It sorts each failure into one of 11 categories (Invocation & CLI, Filesystem & Permissions,
   Build, Toolchain & Packages, …) and one of that category's subcategories.

The category is limited to the taxonomy. A subcategory that isn't in the taxonomy becomes
`"other"`.

**Output:** one row per failed tool call, with `error_category` and `error_subcategory`.

**Cost:** one call per tool call, plus one per failure.

### Changing the labels

Both taxonomies are plain module-level data:
- `TRAJECTORY_MODES` maps a mode id to `Mode(cls, name, definition, rubric)`.
- `COMMAND_TAXONOMY` maps a category to its list of subcategories.

Copy one, edit it, and pass it in as `modes=` or `taxonomy=`.

### From the command line

The repository has a script that runs both taxonomies and prints the rates:

```bash
uv run python examples/harbor_taxonomy.py example_harbor_jobs/trials --tasks wal-recovery-ordering
```

Section 6 of [the notebook](../examples/explore_results.ipynb) shows the same results as
tables.
