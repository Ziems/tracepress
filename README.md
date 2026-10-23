# TracePress

LLM-powered map/reduce over agent traces, built on DSPy. The same pipeline code is meant
to run on static datasets and, later, live streams. See [SPEC.md](SPEC.md) for the design.

```python
import tracepress as tp

tp.configure(lm="openrouter:openai/gpt-6-luna")   # you always pick the model; this is the suggested one
src = tp.read.harbor("example_harbor_jobs/trials", tasks=["wal-recovery-ordering"])

notes = src.table("steps").lm_map("message, tool_calls -> what_happened, is_mistake: bool")
per_trial = notes.group_by("trial").lm_reduce("what_happened, is_mistake -> summary, failure_mode")
per_task = per_trial.group_by("task").lm_reduce("task, summary, failure_mode -> common_failures: list[str]")

per_task.to_pandas()
```

There's no default model. Set one with `tp.configure(lm=...)`, an operator's `lm=`, or the
`TRACEPRESS_LM` environment variable. Model strings take the form `provider:model`:

- `openrouter:openai/gpt-6-luna` (suggested) needs `OPENROUTER_API_KEY`.
- `portkey:<provider slug>/<model>`, e.g. `portkey:openai-prod/gpt-5`, sends requests through
  Portkey's gateway to a model in its Model Catalog. It needs `PORTKEY_API_KEY`, and
  `PORTKEY_BASE_URL` points it at a self-hosted gateway.
- Anything without a `provider:` prefix is a plain litellm model string, passed through as is.

Keys are read from the environment or from `.env`. The example scripts take `--lm`.

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

## Telemetry

TracePress sends one anonymous usage event per run to PostHog, which helps us see which
models, operators and pipelines people use and what runs cost. It prints a one-time notice the
first time it sends.

**What is sent:**
- TracePress, DSPy and Python versions, and the OS name.
- Whether the run succeeded. On failure, the exception class name, but only for built-in,
  TracePress, DSPy or litellm exceptions. Anything else is sent as `"other"`.
- Duration, and counts of LM calls, cache hits, errors, retries, tokens and estimated cost.
- For each model, the gateway and model name, e.g. `openrouter` + `openai/gpt-6-luna`.
  Portkey provider slugs, endpoints and fine-tune ids are stripped.
- How many of each operator the plan has (`lm_map`, `filter`, …), the source kinds
  (`harbor`, `atif`, …), and which built-in pipelines ran.
- A random install id kept in `~/.config/tracepress/telemetry.json`. It isn't derived from
  your machine; delete the file to reset it.

**Never sent:** your data, prompts, signatures, instructions, column or task names, file paths,
hostnames, API keys or error messages.

**Turning it off:** set `TRACEPRESS_TELEMETRY=0` or `DO_NOT_TRACK=1`, or call
`tp.configure(telemetry=False)`. It's also off in CI and under pytest. To see exactly what would
be sent without sending it, set `TRACEPRESS_TELEMETRY=debug`.
