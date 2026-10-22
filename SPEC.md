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
| `lm_fold` | running | Incremental per-group state |

`group_by` accepts any field (e.g. `agent`, `model`), not just hierarchy levels.

### Reduce emission

`lm_reduce(..., emit="close")` is the default: one final result per group.
`emit="update"` / `emit=tp.every(n | "30s")` emits provisional results.

- In batch mode the flag is a no-op (everything is already closed).
- Provisional emission is implemented internally as a fold, then a final pass on close.
- Rows carry `final: bool` so downstream operators can consume only final rows.

### Reduce input binding

Same by-name binding as `lm_map`, with one change inside a reduce:

- An input naming a **child column** receives a **list** of that column's values.
- An input naming a **group key** receives the **scalar** key value.
- A reserved name (e.g. `rows`) receives whole child records.
- A name that matches both a key and a column is an error.

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
  such as "build a failure taxonomy, then label each trial against it".

## Sources / adapters

Adapters convert source formats into the canonical trace schema and emit events.

- Many sources should eventually be supported (Harbor, generic agent traces, OpenTelemetry,
  webhooks, an in-process `tp.log_step(...)` hook, ...). Only Harbor and generic agent
  traces are needed initially.
- **Group closure is the adapter's job.** For now the source says when a group is
  complete (e.g. a Harbor adapter knows the expected trial count from the job config).
  A dedicated Harbor adapter should make loading traces easy and emit closures correctly.
- **OPEN:** fallback for traces that die without finishing (idle timeout?).
- **OPEN:** exact Harbor / agent-trace formats — waiting on example traces.

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
- **OPEN:** Harbor / agent-trace formats (waiting on example traces).
- **OPEN:** fold drift vs. batch reduce quality; whether the final pass on close is enough.
- `requires-python` must be raised to >=3.10 for DSPy.
