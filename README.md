# TracePress

LLM-powered map/reduce over agent traces, built on DSPy. The same pipeline code is meant
to run on static datasets and, later, live streams. See [SPEC.md](SPEC.md) for the design.

```python
import tracepress as tp

src = tp.read.harbor("example_harbor_jobs/trials", tasks=["wal-recovery-ordering"])

notes = src.table("steps").lm_map("message, tool_calls -> what_happened, is_mistake: bool")
per_trial = notes.group_by("trial").lm_reduce("what_happened, is_mistake -> summary, failure_mode")
per_task = per_trial.group_by("task").lm_reduce("task, summary, failure_mode -> common_failures: list[str]")

per_task.to_pandas()
```

The default model is `openrouter/openai/gpt-6-luna`; put `OPENROUTER_API_KEY` in `.env`.

A fuller example, steps → trials → tasks → benchmark:

```bash
uv run python examples/failure_modes.py example_harbor_jobs/trials --tasks wal-recovery-ordering gsea-proteomics
```

Add `--replay` to feed the steps in as a simulated stream (`--faults` also shuffles and
duplicates rows and drops closes). With a warm cache the output matches the batch run exactly.

For Harbor jobs there are built-in versions of the Terminal-Bench 2.0 failure taxonomies
([arXiv:2601.11868](https://arxiv.org/abs/2601.11868)). `tp.pipelines.harbor.trajectory_taxonomy(src)`
judges each failed trial against 9 failure modes, and `tp.pipelines.harbor.step_taxonomy(src)`
classifies each failed command:

```bash
uv run python examples/harbor_taxonomy.py example_harbor_jobs/trials --tasks wal-recovery-ordering
```

To watch trials as a live stream, with a running progress note per trial updated each step
(`lm_fold` with `emit="update"`):

```bash
uv run python examples/live_monitor.py example_harbor_jobs/trials --tasks wal-recovery-ordering --limit 2 --speed 60
```

To dig through results interactively, open [examples/explore_results.ipynb](examples/explore_results.ipynb)
(it's saved with outputs). It covers the job overview, failure modes at each level, drill-down from a
trial to the steps flagged as mistakes, new questions over existing results, and a live fold.
Open it in VS Code with the project's `.venv` kernel, or:

```bash
uv run --with jupyterlab jupyter lab examples/explore_results.ipynb
```

Run the tests with `uv run pytest`.
