# TracePress spec (working draft)

Living record of decisions from design discussions. Open questions are marked **OPEN**.

## Goal

Semantic (LLM-powered) map/reduce over agent traces, in Python, built on DSPy.
Both static datasets and live streams, with almost identical pipeline code.

## Core model

- **Tables are immutable.** Every operation returns a new table (Spark DataFrame style).
- **Rows carry a hierarchical key**, e.g. `benchmark > task > trial > step`. Derived rows
  inherit their parent's key, which gives lineage / drill-down for free.
- **Signatures are one-line DSPy strings**: `"step -> what_happened, is_mistake: bool"`.
  A full `dspy.Signature` class is also accepted for complex cases.
  Inputs bind to columns by name; outputs become new columns.
- **Operators are prefixed `lm_`** when they call an LLM.
- **Lazy plans.** `.collect()` runs a plan in batch, `.start(sink=...)` runs it as a stream.

## Batch = finished stream

Sources emit events:

- `row_added` – a step / trace arrived
- `group_closed` – a trace finished, or all children of a group finished
- `row_updated` – a provisional result changed (only with `emit="update"`)

A static source emits every row, then every close. The engine can't tell the difference.

## v1 operators

| Operator | Rows in → out | Notes |
|---|---|---|
| `map`, `filter` | 1 → 1 / 0–1 | Plain Python, no LLM; cheap pre-filtering |
| `lm_map` | 1 → 1 | Adds output columns |
| `lm_filter` | 1 → 0–1 | Signature with a single bool output |
| `.steps()`, `explode` | 1 → N | Finer grain |
| `lm_extract` | 1 → N | Produces new entities (e.g. failures); rows inherit parent key |
| `group_by(...).lm_reduce(...)` | N → 1 | Coarser grain; output has group keys + reduce outputs |
| `group_by(...).lm_fold(...)` | N → 1 (running) | Carries state through each group's children in key order |

`group_by` accepts any field (e.g. `agent`, `model`), not just hierarchy levels.

### Folds

```python
steps.group_by("trial").lm_fold("progress, step_text -> progress, stuck: bool", emit="update")
```

- Fields on both sides of the arrow are **state**. The LM sees them as `previous_<name>`,
  because DSPy forbids the same name as both input and output. Other inputs bind to the
  newest child's columns or to group keys.
- `init={...}` sets initial state (default `""`, `[]`, `0` or `False` by type).
- One LM call per child, sequential within a group; groups run concurrently.
- The output row is the state after the last child.
- **Out-of-order children rewind.** Each group keeps its children sorted by key plus the
  state after each child. A child that lands before already-processed ones recomputes
  from its position, so the final state never depends on arrival order.

### Reduce emission

`emit="close"` is the default: one final result per group.
`emit="update"` / `emit=tp.every(n)` also emits provisional rows (`final=False`) on live
streams. Both `lm_reduce` and `lm_fold` accept it.

- In batch mode the flag is a no-op (everything is already closed), and costs nothing.
- For `lm_reduce`, provisional results come from a derived "running" signature
  (`current, new_<children> -> outputs`) fed each batch of n new children. **The final row
  always comes from a full reduce at close**, so it equals the batch result exactly.
- For `lm_fold`, provisional rows are the state so far; `every(n)` only thins out emission.
- Versions increase with each emitted row for a key, and the final row has the highest. Every
  consumer keeps the newest version, so a slow provisional update can't overwrite a final row.
- Time-based `tp.every("30s")` is not built yet.
- `table.events()` streams every output event (provisional rows and closes included);
  `collect()` returns only final rows.

### Reduce input binding

Same by-name binding as `lm_map`, with one change inside a reduce:

- An input naming a **child column** receives a **list** of that column's values.
- An input naming a **group key** receives the **scalar** key value.
- A reserved name (e.g. `rows`) receives whole child records.
- A name that is part of the group key always binds to the key (it is constant within the
  group). This lets `group_by("trial", "reward")` pass `reward` as a scalar.

```python
per_trial.group_by("task").lm_reduce(
    "task, summary, failure_mode, succeeded -> common_failures: list[str], success_rate_notes"
)
# task: str (key); summary / failure_mode / succeeded: one entry per trial
```

List inputs stay aligned per child: the LLM sees one row per child, not separate lists,
so it can tell which summary goes with which outcome.

Internally, each reduce follows Spark's `Aggregator` shape. The engine derives these from
the one reduce signature; users can override them with `merge="..."` / `fold="..."`:

| Spark | TracePress use |
|---|---|
| `reduce(buf, row)` | **fold**: incremental update for `emit="update"` |
| `merge(b1, b2)` | **merge**: combines partial outputs when tree-reducing large groups |
| `finish(buf)` | final pass on group close |

Groups that don't fit in context are chunked under a hard size budget, and the chunk
results are combined with the merge signature.

### Deferred

- **Joins** (`join` on keys, `lm_join` semantic predicate). Needed for two-pass patterns
  such as "build a failure taxonomy, then label each trial against it". The first real
  pipeline already wanted one: attaching each trial's reward to its steps currently takes
  a Python dict lookup inside `map`.

## Implementation status

Built and tested: the plan/runtime/event engine; `map`, `filter`, `select`, `explode`,
`group_by(...).reduce` (Python), `lm_map`, `lm_filter`, `lm_extract`, `lm_reduce` (with
tree merge and `emit="update"` / `tp.every(n)`), `lm_fold`; `collect` / `acollect` /
`to_pandas` / `tp.collect(*tables)` / `table.events()` / `from_rows` / `from_pandas`;
ATIF and Harbor adapters; placeholder adapters; `tp.replay`.

Batch-vs-replay equivalence (same keys and values) holds in the tests for reduces, folds
and provisional reduces under every fault setting. On the real Harbor sample (6 tasks,
546 LM calls) the replayed output is byte-identical to the batch output.

Not yet: sinks / `.start()`, time-based `tp.every("30s")`, user overrides for merge/fold
signatures, backpressure (queues are unbounded), retractions.

Streaming findings:

- **Duplicates are free.** Row-level LM operators skip a row already seen with the same
  key, version and values; groups do the same. Seen keys are forgotten when their prefix
  closes, so memory stays bounded by open groups.
- **No retractions.** If a provisional row explodes (`explode` / `lm_extract`) into more
  children than its final version, the extra children stay downstream. Explode only closes
  a row's children once the row is final.
- **Downstream LM operators run on every provisional row** they receive. That's the
  cascade cost noted in the design; `.filter(lambda r: r.final)` stops it.
- **One hung LM request stalls a whole fold**, because fold steps are sequential. See
  "LM latency and timeouts" below.

### LM latency and timeouts

Measured on GPT-6 Luna via OpenRouter, uncached, about 950 calls:

| Workload | p50 | p90 | p99 | max |
|---|---|---|---|---|
| per-step `lm_map` (concurrency 32) | 3.9s | 5.3s | 6.9s | 10.8s |
| fold steps (sequential per trial) | 5.7s | 8.1s | 14.5s | 14.5s |
| six-task failure analysis (maps + reduces) | 3.9s | 6.0s | 9.6s | 17.6s |

- Latency tracks output length: roughly 2s plus ~90 completion tokens per second. Luna
  spends 40–250 completion tokens on reasoning even for one-sentence answers.
- **Requests occasionally hang** (none in this sample, but earlier runs stalled for up to
  10 minutes, litellm's default timeout). A sequential fold over ~22 steps at ~6s/step takes
  ~2 minutes per trial even without hangs.
- **Timeouts are enforced by the executor** (`asyncio.wait_for`), not passed to the LM,
  because DSPy includes request kwargs in its cache key and changing a timeout would
  otherwise invalidate the cache. `request_timeout=30` (row calls, fold steps, running
  updates; ~2x the worst observed) and `reduce_timeout=120` (full reduces and merges,
  whose prompts can approach the chunk budget). A timed-out attempt is retried up to
  `timeout_retries=2` times; `tp.last_run()` counts `timeouts` and keeps a per-call
  `call_log` with `latency()` percentiles.
- Lower `reasoning_effort` was inconclusive: `minimal` medians of 1.8–4.0s vs 3.0–3.4s for
  the default across three runs, with provider variance as large as the effect. The default
  is unchanged; pass `tp.configure(lm=dspy.LM(..., reasoning_effort=...))` to try it.
- A close with no rows under it (e.g. the steps of a crashed trial) keeps its position
  relative to the source's preceding row, since there's no timestamp to place it by.
- `collect()` on a replay runs until the replay ends; on a truly live source it would never
  return, so live sources will need `.start()`.

Decisions made while building:

- **No default LM:** the user always picks one with `tp.configure(lm=...)`, a per-operator
  `lm=`, or `TRACEPRESS_LM`, and a run without one fails before any call. Model strings are
  `provider:model` (`openrouter:openai/gpt-6-luna`, the suggested one, or
  `portkey:<provider slug>/<model>`). A string without a provider prefix is passed to
  litellm as is. Keys are read from the environment or `.env`.
- **Output format:** DSPy's `JSONAdapter` by default; if a response can't be parsed, the
  call is retried with `XMLAdapter`. Without that fallback, a cached unparseable response
  fails the same call on every rerun.
- **Caching** uses DSPy's request cache; a fully cached rerun is byte-identical.
- **Errors:** LM operators default to `on_error="skip"` (logged and counted in
  `tp.last_run()`); `on_error="raise"` fails the run.

## Sources / adapters

Adapters convert source formats into the canonical trace schema and emit events.

- Many sources should eventually be supported (Harbor, generic agent traces, OpenTelemetry,
  webhooks, an in-process `tp.log_step(...)` hook, ...). Only Harbor and generic agent
  traces are needed initially.
- **Group closure is the adapter's job.** For now the source says when a group is
  complete (e.g. a Harbor adapter knows the expected trial count from the job config).
  A dedicated Harbor adapter should make loading traces easy and emit closures correctly.
- **OPEN:** fallback for traces that die without finishing (idle timeout?).

### Adapter interface

Every adapter implements the same small interface, so the engine never special-cases a format:

```python
class Adapter(Protocol):
    levels: list[str]                       # hierarchy, e.g. ["dataset", "task", "trial", "step"]
    def tables(self) -> dict[str, Schema]: ...       # named tables and their schemas
    def events(self, table: str) -> AsyncIterator[Event]: ...  # RowAdded / Closed
    bounded: bool                           # True = finite (batch), False = live
```

Readers live under `tp.read.*` (bounded) and `tp.watch.*` (live).

### Planned adapters

Placeholders exist so no single format shapes the design. Unimplemented adapters raise
`NotImplementedError` naming the format.

| Adapter | Source | Likely hierarchy | Status |
|---|---|---|---|
| `tp.read.atif` | ATIF trajectories | trajectory > step | v1 |
| `tp.read.harbor` | Harbor job / trials dir | dataset > task > trial > step | v1 |
| `tp.read.dspy` | DSPy program traces | run > module call (nested) > LM call | placeholder |
| `tp.read.otel` | OpenTelemetry / OpenInference spans | trace > span (variable-depth tree via `parent_id`) | placeholder |
| `tp.read.claude_code` | Claude Code / Agent SDK session `.jsonl` | session > message (+ subagent sidechains) | placeholder |
| `tp.read.chatgpt` | ChatGPT conversation exports | conversation > message (branching tree) | placeholder |

OpenTelemetry and ChatGPT exports are trees of variable depth, not fixed levels. They are
the test case for keeping hierarchy adapter-defined; how variable-depth trees map onto
keys is **OPEN**.

### ATIF and Harbor adapters

Based on the example job in `example_harbor_jobs/trials/` (gitignored).

- `tp.read.atif(path)` reads any ATIF trajectory (the generic "agent traces" format).
- `tp.read.harbor(dir)` builds on it: steps from `agent/trajectory.json` (ATIF), trial
  metadata from `result.json` / `config.json`, tests from `verifier/ctrf.json`.

**Hierarchy:** `dataset > task > trial > step`. `job_id`, `agent` and `model` are columns,
not levels, so `group_by("task")` can compare approaches across jobs.

| Table | Grain | Columns |
|---|---|---|
| `trials` | trial | task, dataset, job_id, agent, model, agent_kwargs, reward, exception_type, cost_usd, tokens, started/finished/duration, n_steps |
| `steps` | step | step_id, source, message, tool_calls, observations (paired with their calls), metrics, timestamp, is_sidechain |
| `tool_calls` | step > call | function_name, arguments, result |
| `tests` | trial > test | per-test pass/fail from `ctrf.json` |

Handling:

- **Images** in tool results are dropped and replaced with `[image omitted]`.
- **Trials with no trajectory** (agent crashed) still produce a `trials` row with zero
  steps and the exception recorded.
- **Rewards** are normalized to float.
- **Subagents:** `is_sidechain` is exposed as a column; nesting is untested (none in the sample).

Streaming:

- **OPEN:** group closure for Harbor jobs (the sample has no job-level expected trial count).
  v1 is batch only for Harbor.
- `trajectory.json` appears to be written once, when the agent finishes, so live step-level
  streaming would need per-agent log tailing (e.g. Claude Code session `.jsonl`). Later.

### Adapters define the tables

There is no universal "trial" or "step" primitive. The engine core knows only rows, keys
and closures. Each adapter declares:

- its **hierarchy levels** (e.g. Harbor: `job > task > trial > step`; a subagent-aware
  harness might also expose spans with `parent_id` / depth),
- the **named tables** it exposes (e.g. `src.table("trials")`, `src.table("steps")`), with schemas,
- **when each group closes**.

Adapters should follow a small set of optional column-name conventions (`role`, `content`,
`tool_calls`, `error`, `timestamp`, `reward`, ...) so pipelines port between sources where
the data allows, but nothing in the engine depends on them.

### Simulated streams (replay)

`tp.replay(source, ...)` turns any bounded source into a live one, for testing streaming
without a live system:

```python
live = tp.replay(tp.read.harbor("example_harbor_jobs/trials"), speed=100, seed=0)
```

- **Timing:** rows are emitted in timestamp order (ATIF step timestamps, trial
  `started_at`), so trials overlap the way they actually ran. `speed=100` is 100× real
  time; `speed=None` is as fast as possible but keeps the interleaving.
- **Closures:** a trial closes after its last step (its row, with reward, is emitted at
  close); a task closes after all its trials; the dataset closes at the end.
- **Fault injection** for testing edge cases: `drop_closes=0.05` (traces that never
  finish), `shuffle_within=2` (out-of-order steps), `duplicates=0.01`.
- **Equivalence check:** with the LLM cache warm, a pipeline run on `tp.replay(src)` must
  produce the same final rows as the same pipeline on `src`. This is the main
  correctness test for the streaming engine.

## Engine architecture

```
 Python API        Table objects (immutable, lazy)
 Logical plan      DAG of operator nodes + schemas, validated at build time
 Physical plan     concrete strategies (batch tree-reduce vs streaming fold, ...)
 Runtime           one async event-driven dataflow engine
 LM executor       shared DSPy calls: concurrency, rate limits, retries, cache
```

- **Rows**: `Row(key, values, final, version)`. `key` is a tuple of `(level, id)` pairs;
  a table's **grain** is its list of levels.
- **Schemas** are inferred from signatures at build time, so unknown columns and invalid
  grain changes fail before any LLM call.
- **The plan is a DAG**; shared upstream nodes run once (`tp.run([a, b], ...)`).

### Events and closure

`RowAdded(row)`, `RowUpdated(row)` (same key, higher version), `Closed(prefix)`.

- `Closed(prefix)` is a key-based watermark: no more rows will arrive under that prefix.
- Row-level operators forward a close only after their in-flight LLM calls under that
  prefix have finished (a close must never overtake its rows).
- Reduces buffer (or fold) per group and emit when the group closes.
- A group keyed on non-hierarchy fields closes when the smallest hierarchy prefix that
  contains it closes. A group that can never close (e.g. global `group_by("agent")` on an
  endless stream) triggers a warning.
- Folds process each group in order (reorder buffer by index); groups run concurrently.

### Batch vs streaming

One runtime, one set of semantics. Only the source and the physical plan differ:
batch reduces see whole groups (tree-reduce, `emit` ignored); streaming reduces buffer or fold.

- `.collect()` runs until all sources are exhausted and closed, and returns `list[Row]`
  containing only final rows. Rows support attribute access (`row.summary`) and keep
  their full `key` for lineage.
- `.to_pandas()` returns a DataFrame with key levels as columns.
- `tp.from_rows(rows)` / `tp.from_pandas(df)` turn results back into a static source, so
  expensive lower levels can be computed once and higher levels iterated on.
- `.start(sink)` runs in the background and returns a handle (`stop()`, `status()`, awaitable).

### Execution and LM calls

- **Parallelism = many concurrent LM calls** in one async process, bounded by per-provider /
  per-model rate limits (requests and tokens per minute, adaptive backoff on 429s).
  Multi-machine execution is not planned.
- Model and DSPy module are configurable per operator (`lm=...`, `module=...`).
- **Persistent LLM cache** keyed on (signature, inputs, model, module).
- **Fault tolerance = replay + cache**: re-run the source; completed calls are cache hits.
  No operator-state checkpointing in v1.
- Later: provider batch APIs for batch mode (cheaper, higher latency).

## Open questions

- **OPEN:** idle-timeout fallback for traces that never close.
- **OPEN:** streaming group closure for Harbor jobs.
- **OPEN:** mapping variable-depth span trees (OpenTelemetry, ChatGPT branches) onto keys.
- **OPEN:** how far provisional (running) reduce results drift from the full reduce on close.
- `requires-python` must be raised to >=3.10 for DSPy.
