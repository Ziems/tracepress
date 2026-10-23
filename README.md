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

Run the tests with `uv run pytest`.
