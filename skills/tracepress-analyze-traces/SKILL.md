---
name: tracepress-analyze-traces
description: Analyze agent traces (Harbor jobs, ATIF trajectories) with TracePress, LLM-powered map/reduce in Python. Use when asked to find failure modes, summarize or compare agent runs, explain why trials failed, or dig into steps, trials, tasks or benchmarks of agent traces.
---

# Analyzing agent traces with TracePress

TracePress runs DSPy-style LLM operators over tables of trace data. Pipelines are lazy:
build them with table methods, then run them with `collect()`. Every LLM call goes through
DSPy's disk cache, so rerunning a pipeline only pays for prompts that changed.

## When to use

- "Why did these trials fail?", "What are the common failure modes?"
- Comparing attempts, agents or models on the same tasks
- Summarizing what an agent did, step by step or per trial
- Drilling from an aggregate finding back to the exact steps behind it

For live or streamed traces, use the `tracepress-stream-traces` skill instead.

## Setup

- In the TracePress repo: `uv sync`; run code with `uv run python ...`.
- **There is no default model.** Configure one before running anything:
  `tp.configure(lm="openrouter:openai/gpt-6-luna")`. That's the suggested model, at about $0.10
  per million input tokens and $0.50 per million output tokens. Without one, a run fails with
  "No LM configured". Use the model the user names if they name one.
- Model strings are `provider:model`:
  - `openrouter:<model>` needs `OPENROUTER_API_KEY`.
  - `portkey:<provider slug>/<model>` (Portkey Model Catalog) needs `PORTKEY_API_KEY`.
  - A plain litellm string, with no `provider:` prefix, also works.

  Keys are loaded from `.env` in the working directory. Override the model per operator with
  `lm=`, or set `TRACEPRESS_LM`.

## Steps

1. **Load the traces and look before spending.** Loading and plain Python operators cost nothing.

   ```python
   import tracepress as tp

   tp.configure(lm="openrouter:openai/gpt-6-luna")         # required; suggested model, or the user's choice
   src = tp.read.harbor("path/to/job_or_trials_dir")      # or tasks=[...], limit=N
   trials = src.table("trials").to_pandas()
   trials.groupby("task").reward.mean().sort_values()
   ```

   Harbor tables (hierarchy `dataset > task > trial > step`):
   - `trials`: `job_id, agent, model, agent_kwargs, reward, exception_type, exception_message,
     cost_usd, input_tokens, output_tokens, cache_tokens, started_at, finished_at, duration_s, path`
   - `steps`: `source` ("user"/"agent"), `message, reasoning, tool_calls` (list of
     `{id, name, arguments, result}`), `observation, model_name, timestamp, metrics, is_sidechain`,
     plus `job_id, agent, model`
   - `tool_calls`: one row per call (`name, arguments, result`)
   - `tests`: per-test `name, status, message, trace` from the verifier

   `tp.read.atif(path)` reads bare ATIF trajectory files (`trajectories`, `steps`, `tool_calls`).
   Images in tool results are replaced with `[image omitted]`.

2. **Estimate the cost.** There's roughly one call per row for `lm_map` / `lm_filter` / `lm_extract`
   and one per group for `lm_reduce`. A step-level map dominates: count agent steps first with
   `len(src.table("steps").filter(lambda r: r.source == "agent").collect())`. As a reference,
   six Harbor tasks (30 trials, about 530 agent steps) through a four-level pipeline was 546 calls,
   about $0.08 and 1–2 minutes uncached. Tell the user the estimate before running anything large.

3. **Build the pipeline.** Signatures are one-line DSPy strings. Inputs bind to columns by name,
   and outputs become new columns.

   ```python
   rewards = {r.trial: r.reward for r in src.table("trials").collect()}   # plain floats, see pitfalls

   steps = (
       src.table("steps")
       .filter(lambda r: r.source == "agent")
       .map(lambda r: {"step_text": render(r), "reward": rewards.get(r.trial)},
            columns={"step_text": str, "reward": float})
   )
   notes = steps.lm_map("step_text -> what_happened, is_mistake: bool",
                        instructions="Summarize this agent step in one sentence; was it a mistake?")
   per_trial = notes.group_by("trial", "reward").lm_reduce(
       "reward, what_happened, is_mistake -> summary, failure_mode")
   per_task = per_trial.group_by("task").lm_reduce(
       "task, summary, failure_mode, reward -> common_failures: list[str], what_separates_passes")
   ```

   Write `render(r)` to turn a step into compact text, and **clip long tool output** (to about
   1,500 characters) to keep prompts small. `tracepress.pipelines.harbor` has `render_step` and
   `clip` (head and tail) you can reuse. `examples/failure_modes.py` has the full four-level
   pipeline (`build(src, rewards)`).

   Operators:
   - `map(fn, columns=...)`, `filter(pred)`, `select(*cols)`, `explode(col)`: plain Python, free
   - `lm_map(sig)`: one call per row; outputs added as columns
   - `lm_filter(sig)`: keeps rows where the signature's single `bool` output is true
   - `lm_extract("x -> items: list[str]", keep=(...))`: one new row per list element, one level deeper
   - `group_by(*names).lm_reduce(sig)`: one row per group
   - `group_by(*names).reduce(fn)`: Python reduce, `fn(rows) -> dict`

   How `lm_reduce` binds its inputs:
   - An input naming a **group key** (a hierarchy level, or an extra column passed to `group_by`)
     gets that single value.
   - Any other input names a **child column**. The children arrive as one record per child, so
     values from the same child stay together.
   - `rows` gets whole child records.
   - Grouping by a level also groups by its ancestors. `group_by("task", "agent")` compares agents
     within each task.
   - Large groups are split into chunks and merged automatically (`reduce_chunk_chars`).

4. **Try it on a sliver first.** Run with `tp.read.harbor(path, limit=1)` (or one task), read the
   outputs, and tune the instructions. Then run the full set; unchanged prompts come from the cache.

5. **Run and check the stats.**

   ```python
   notes_rows, trial_rows, task_rows = tp.collect(notes, per_trial, per_task)  # shared work runs once
   tp.last_run()   # calls, errors, fallbacks, timeouts, tokens, seconds; .latency() for percentiles
   ```

   In Jupyter (or any running event loop), use `await tp.acollect(...)` / `await table.acollect()`.
   LM errors are skipped by default and counted in `errors`. Pass `on_error="raise"` to fail fast.

6. **Dig into the results.** Every row keeps its full key, so you can trace any summary back to its
   steps.

   ```python
   df = tp.rows_to_pandas(notes_rows)                 # key levels become columns
   df[(df.trial == "some-trial") & df.is_mistake]     # the steps behind a trial's failure
   ```

   To ask a new question without redoing the step-level work, feed results back in:
   `tp.from_rows(trial_rows).lm_map("summary, failure_mode -> category").collect()`.
   `examples/explore_results.ipynb` walks through all of this.

## Built-in pipelines

For Harbor jobs, `tp.pipelines.harbor` has the two failure taxonomies from the Terminal-Bench 2.0
paper (arXiv:2601.11868). Try these before writing your own failure analysis when the user asks
"how do these runs fail?" in Terminal-Bench terms.

```python
from tracepress.pipelines import harbor

modes = harbor.trajectory_taxonomy(src)            # one row per (failed trial, failure mode)
wide = harbor.trajectory_labels_wide(modes.collect())   # one dict per trial, plus class rollups
commands = harbor.step_taxonomy(src)               # one row per failed tool call
```

- `trajectory_taxonomy(src, failed_only=True, modes=TRAJECTORY_MODES)`: each agent step is
  digested, the digests are joined into a per-trial transcript, and each of the 9 modes gets its
  own rubric judge. Columns: `failure_mode`, `mode_class`, `matched`, `rationale` (cites step
  numbers) and `transcript`. Cost: one call per agent step plus 9 per trial.
- `step_taxonomy(src, failed_only=False, taxonomy=COMMAND_TAXONOMY)`: detects whether each tool
  call failed, then classifies it as `error_category` / `error_subcategory`. A subcategory outside
  the taxonomy becomes `"other"`. Cost: one call per tool call plus one per failure.
- Both taxonomies are plain dicts (`TRAJECTORY_MODES` of `Mode(cls, name, definition, rubric)`,
  `COMMAND_TAXONOMY` of category to subcategories). Copy and edit them to change the labels.
- `examples/harbor_taxonomy.py` runs both and prints rates. Section 6 of the notebook shows
  them as tables.

## Pitfalls

- **Telemetry:** TracePress sends one anonymous usage event per run (counts, versions, model
  and operator names, never data or prompts). If the user asks to turn it off, set
  `TRACEPRESS_TELEMETRY=0` or call `tp.configure(telemetry=False)`.
- **Pass plain Python values into pipelines, not numpy or pandas values.** `np.float64(1.0)` renders
  differently in prompts than `1.0`, which changes the prompt and misses the cache. Build lookups
  from `Row`s, not DataFrame columns.
- **Don't name a column after a key level.** For example, use `step_text`, not `step`. `map` raises
  an error when a column would shadow a level.
- **There are no joins yet.** Attach trial-level values (reward, etc.) to steps with a dict lookup in
  `map`, as above. To make such a value a scalar reduce input, include it in `group_by`.
- **`lm_filter` needs exactly one `bool` output**, and `lm_extract` exactly one `list[...]` output.
  Unknown input names fail when the pipeline is built, before any LLM call.
- **Changing prompts, instructions or input rendering re-runs those calls.** Avoid gratuitous edits
  once results are cached.
- **`emit="update"` only matters for live streams.** In batch mode it changes nothing.
- **Timeouts:** calls time out after `request_timeout` (30s) or `reduce_timeout` (120s) and are retried
  up to `timeout_retries` (2) times. Adjust with `tp.configure(...)`; this doesn't affect the cache.
