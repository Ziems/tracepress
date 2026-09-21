# Pipeline Runtime Spec

A config-driven map/filter/reduce runtime. Stages are plain Python functions imported from a module. The config file declares how they chain together. The central design problem this solves is variable-size groups, where a reduce must wait for all children of a parent before firing, and the number of children differs per parent and is not known to the config.

## 1. Design principles

Read these first. Most implementation questions resolve by referring back to them.

1. **Group size never appears in config.** A group carries its own size. The config declares that a group exists and what closes it. The runtime counts.
2. **Config owns the topology.** Reading the config alone tells you the full shape of the dataflow. No edge is created from inside Python.
3. **One barrier mechanism.** Every way of forming a group resolves to the same accounting rule described in section 6. There is not a separate code path per group type.
4. **Stages are ordinary functions.** No base classes, no decorators required, no framework imports in user code.
5. **Fail at load time.** Anything knowable before the first record is read is validated before the first record is read.

## MVP boundary

The MVP should prove the topology and barrier model without taking on the full runtime surface.

In scope for MVP:

- Config validation, path DAG ordering, stage resolution, and record identity.
- Linear JSONL pipelines with `map`, `filter`, `stage`, and `tee`.
- All three group constructors: `expand`, `by`, and `every`.
- Nested groups and deterministic reduce ordering by emit path.
- Sync and async stage functions using return values or `ctx.emit`.
- `on_child_error: fail`, `partial`, and `skip`, including the skip manifest.
- Basic diagnostics for stuck groups and accounting totals.

Post-MVP:

- Generator and async-generator stage shapes.
- Full concurrency and backpressure controls.
- Side channels.
- Timestamp-based resumability.

## 2. Config schema

```yaml
from: ./stages                  # module or package where stage functions are resolved

defaults:                       # optional, applied to every group unless overridden
  on_child_error: fail
  on_group_incomplete: fail
  ordered: true

id_field: id                    # optional, see section 11

pipelines:
  - in: traces.jsonl
    out: overviews.jsonl
    pipeline:
      - filter: keep
      - tee: kept.jsonl
      - group:
          expand: split_sessions
          pipeline:
            - map: label
            - map: score
          reduce: merge
          on_child_error: partial

```

### Top level keys


| Key           | Required | Meaning                                                                                       |
| ------------- | -------- | --------------------------------------------------------------------------------------------- |
| `from`        | yes      | Import path for stage resolution. Relative paths resolve against the config file's directory. |
| `defaults`    | no       | Group options applied when a group does not specify them.                                     |
| `concurrency` | post-MVP | Backpressure limits. See section 7.                                                           |
| `id_field`    | no       | Record field used for error messages. See section 11.                                         |
| `pipelines`   | yes      | List of pipeline blocks.                                                                      |


### Pipeline block keys


| Key        | Required | Meaning                                                                    |
| ---------- | -------- | -------------------------------------------------------------------------- |
| `in`       | yes      | Input file path, or a list of paths read in order.                         |
| `out`      | yes      | Output file path. Records reaching the end of `pipeline` are written here. |
| `pipeline` | yes      | Ordered list of steps.                                                     |


Pipelines execute in dependency order. A pipeline whose `in` matches another pipeline's `out` runs after it. This is what gives you a DAG without DAG syntax. Fan out is expressed by two pipelines sharing an `in`. See section 10 for the path matching rules, which are load-bearing and easy to get wrong.

### Step forms

A step is a single-key mapping. Six forms exist.

```yaml
- map: label            # stage that must emit exactly 1
- filter: keep          # stage that must emit 0 or 1
- stage: whatever       # stage with no emit-count constraint
- tee: path.jsonl       # side output, stream continues unchanged
- alert: flag_secret     # side alert, stream continues unchanged
- group: {...}          # see below

```

`map`, `filter`, and `stage` are the same mechanism. The difference is an assertion the runtime enforces on emit count, and the readability of the config. A `map` that emits twice is a config error surfaced at runtime with the stage name, the record identity, and the actual count. Treat these labels as executable documentation.

`tee` writes every record passing through it to the given path and passes it downstream unchanged. It exists so a long chain can expose an intermediate for debugging without being split into two pipeline blocks joined by a filename.

`alert` calls a stage and passes the record downstream unchanged. A Python stage returns a reason string to fire, or `None` to stay quiet. A blank string or the word `none` also stays quiet. Any other return value is a runtime error. The runtime appends an alert event and does not put it in `out`:

```json
{"name": "flag_secret", "reason": "ssn", "session_id": "s1"}
```

A DSPy `alert` stage has exactly one bool output. False stays quiet. True writes the module's outputs onto the event:

```json
{"name": "detect_frustration", "session_id": "s1", "has_frustration": true, "evidence": "This is taking forever"}
```

`name` is the stage name. `reason` is the string a Python stage returned. When `id_field` is set, that field is copied from the record. Inside a group it is copied from the parent when the child record does not have it. Named pipelines write `.trace-miner/alerts/<name>.jsonl`. A pipeline with no name writes `.trace-miner/alerts/<index>.jsonl`, using the pipeline's position in the file starting at 0. A batch run truncates that file at the start of the pipeline. A stream appends.

### Group block keys


| Key                       | Required    | Meaning                                                          |
| ------------------------- | ----------- | ---------------------------------------------------------------- |
| `expand` / `by` / `every` | exactly one | How the group is formed. See section 5.                          |
| `pipeline`                | no          | Steps applied to each child. Defaults to empty.                  |
| `reduce`                  | yes         | Stage called once per group with the list of surviving children. |
| `on_child_error`          | no          | `fail` (default), `partial`, or `skip`. See section 8.           |
| `on_group_incomplete`     | no          | `fail` (default) or `partial`. See section 8.                    |
| `ordered`                 | no          | Sort children before reduce. Default `true`. See section 9.      |
| `sidechannels`            | post-MVP    | Map of channel name to output path. See section 12.              |


Groups nest. A `group` step is legal inside a group's `pipeline`.

## 3. Stage contract

A stage is a module-level function taking the record, and optionally a context.

```python
def label(record, ctx):
    ctx.emit({**record, "label": classify(record)})

```

Resolution rules:

- Look up the name as an attribute of the module named in `from`. If `from` names a package, search its `__init__` exports.
- Determine arity by attempting `inspect.signature(fn).bind(record)` and then `bind(record, ctx)`. Use whichever binds, preferring the two-argument form. If neither binds, fail at load time naming the stage and its signature. Do not count positional parameters, because that silently accepts a required third argument and produces a confusing runtime `TypeError` on the first record instead of a clear error at startup.
- Resolve and validate every stage in the config before reading any input.

### Producing output

Output is produced one of two ways, and mixing them is an error.

- Call `ctx.emit(record)` zero or more times and return `None`.
- Return a non-`None` value, which is emitted as **exactly one record**.

A returned list is one record. Fanout requires `ctx.emit`. Post-MVP generator stages may also fan out with `yield`. If return could mean fanout, you could never emit a record that happens to be a list, and the ambiguity would be silent.

If a stage both calls `emit` and returns non-`None`, raise. Returning `None` after zero emits means the record is dropped, which is how a filter expresses rejection.

### Callable shapes

The runtime design supports four callable shapes, dispatched by inspection at load time. MVP
supports sync and async functions.


| Shape              | Phase    | Detected by                   | Output                             |
| ------------------ | -------- | ----------------------------- | ---------------------------------- |
| sync function      | MVP      | default                       | `return` or `ctx.emit`             |
| generator function | post-MVP | `inspect.isgeneratorfunction` | each `yield` is one emit           |
| async function     | MVP      | `inspect.iscoroutinefunction` | `return` or `ctx.emit`, awaited    |
| async generator    | post-MVP | `inspect.isasyncgenfunction`  | each `yield` is one emit, iterated |


Post-MVP, generators are sugar over the emit primitive rather than a parallel mechanism. A `yield` is an emit, and the barrier accounting is identical, with the sentinel dropping when the generator is exhausted rather than when a function returns. This makes streaming expand the natural way to write a producer.

```python
def split_sessions(trace, ctx):
    for chunk in stream_chunks(trace):
        yield chunk

```

`ctx.emit` remains the primitive. It is synchronous even inside async stages. Post-MVP
backpressure may require an awaitable emit path; do not make that API promise until global
backpressure is being built.

## 4. Context object

Pass a real object, not a dict, so fields can be added without breaking callers.

```python
ctx.emit(record)              # produce a downstream record
ctx.side(channel, record)     # post-MVP: write to a declared side channel
ctx.parent                    # the record this group was expanded from, or None
ctx.group_id                  # stable id of the enclosing group
ctx.key                       # grouping key when the group came from `by`
ctx.index                     # this record's emit index within its group
ctx.path                      # full emit path tuple, as in section 9
ctx.depth                     # group nesting depth, 0 at top level
ctx.failures                  # list of failures, populated for reduce under `partial`
ctx.log                       # logger scoped to stage and group

```

`ctx.parent` is the field that earns its place. Without it, a reduce that needs the record its children came from forces every map stage to copy a parent id into every child purely as plumbing.

## 5. Group formation

Three constructors. All three produce a group with the same accounting.

### `expand`

```yaml
- group:
    expand: split_sessions
    pipeline: [map: label]
    reduce: merge

```

`split_sessions` is a stage. Each record it receives opens one group, and every record it emits is a child of that group. The count is however many times it emitted.

This is the primary form and the one to build first. Completion is unambiguous, because the group closes when the producer finishes.

It supports streaming producers where the count is unknown at the start. Chunk 1 can be labeled while chunk 40 is still being produced. The barrier cannot fire early because of the sentinel rule in section 6. Do not implement this by computing N up front.

### `by`

```yaml
- group:
    by: session_id
    pipeline: [map: label]
    reduce: merge

```

Groups records by a field value. `by` may name a record field or a stage that returns a key.

This form has a sharp edge worth being explicit about. A key-based group has no natural close, so completeness is only known when the input is exhausted. Default behaviour is drain at end of input, meaning all groups fire after the last input record has been fully processed. State this in user-facing docs, because it means memory grows with the number of distinct keys, and it interacts badly with the backpressure bound in section 7.

Optionally support `complete_when` later, either a per-record declared total or a sentinel record, for streaming use. Do not build it in v1.

### `every`

```yaml
- group:
    every: 50
    pipeline: [map: label]
    reduce: merge

```

Fixed-size windows over the incoming stream. The final window may be short, and it fires at end of input rather than being discarded. This is the least interesting form and exists mainly for batching. Include it, but do not let it shape the design.

## 6. The barrier

This is the core of the implementation. Use credit accounting, not a precomputed count.

Each group holds an integer of **outstanding obligations**. The invariant is that the count equals the number of records that might still arrive at the barrier, plus one for any producer that might still emit.

Rules:

1. When a group opens, set count to 1. That 1 is a **sentinel** held by the producer.
2. Every emit into the group increments by 1.
3. When the producer finishes, whether by returning or by generator exhaustion, decrement the sentinel by 1.
4. When a record inside the group reaches the barrier, decrement by 1.
5. When a record inside the group is dropped by a filter, decrement by 1.
6. When a record inside the group enters a nested group, do not adjust the outer count. The record's obligation to the outer group is discharged when the nested reduce output reaches the outer barrier, and a nested reduce emitting N records adjusts the outer count by N minus 1 at that point.
7. When the count reaches 0, call `reduce` with the collected children.

The sentinel is what makes streaming expand correct. Without it, a group whose producer has emitted one child, and where that child completes before the second is emitted, would fire the barrier at count 0 prematurely.

This works under arbitrary concurrency. Guard the counter with a lock or run all accounting on a single event loop, and never let a decrement to 0 happen twice for the same group.

Assert at end of run that every opened group has fired or been explicitly handled. A group stuck above 0 at shutdown is a bug in the accounting and must be reported loudly, not silently dropped. See section 14 for the diagnostics that make this debuggable.

## 7. Concurrency and backpressure (post-MVP)

Without a bound, a fast producer feeding a slow map stage holds the entire input in memory. Streaming expand makes this worse rather than better, because it starts child work eagerly.

This section is intentionally post-MVP. The MVP runtime runs sequentially and uses the same barrier accounting, so the accounting model can be proven before capacity management is added.

Bound the number of records in flight. `concurrency.max_in_flight` caps records that have been emitted and not yet retired, where retiring means reaching a barrier, being dropped by a filter, or being written to an output. When the system is at capacity, `ctx.emit` blocks in sync stages and awaits in async stages.

### The deadlock to design around

This is the failure that will look like a random hang under load, so handle it deliberately.

If reduce draws from the same capacity budget as children, a group can hold every slot with completed children waiting on a barrier that cannot fire, because the reduce that would release them cannot acquire a slot. Everything stops and nothing looks wrong.

Give reduce reserved capacity. `concurrency.reduce_slots` is a separate pool that only reduce invocations draw from, and it must be non-zero. Children never draw from it. A simpler alternative is to run reduce inline on the thread that decremented the count to zero, which sidesteps the pool entirely at the cost of blocking that worker. Either is fine. Sharing one budget is not.

Nesting deepens this. A nested reduce whose output must reach an outer barrier needs a slot too, so size `reduce_slots` at least to the maximum group nesting depth, and document that.

## 8. Failure and incompleteness

These were one setting in an earlier draft and are now two, because "reduce over survivors" is a reasonable answer to a failed child and a bad answer to a group you never finished collecting.

### `on_child_error`, when a child raises


| Value     | Behaviour                                                                                                      |
| --------- | -------------------------------------------------------------------------------------------------------------- |
| `fail`    | Abort the run. Report the failing record, the stage, and the group id. Default.                                |
| `partial` | Drop the failing child, decrement the count, reduce over survivors. The failure is recorded in `ctx.failures`. |
| `skip`    | Drop the entire group. Do not call reduce. See cancellation below.                                             |


### `on_group_incomplete`, when a group cannot be completed

Applies when input is exhausted with a `by` group still open under a future `complete_when`, or at shutdown with groups still above zero.


| Value     | Behaviour                                                            |
| --------- | -------------------------------------------------------------------- |
| `fail`    | Abort and report every incomplete group. Default.                    |
| `partial` | Reduce over whatever arrived, marking the group incomplete in `ctx`. |


### Cancellation semantics for `skip`

There is no clean answer here and the spec should not pretend otherwise. Side effects that have already happened cannot be unwound, and adding rollback is a much larger project than this runtime deserves.

`skip` means precisely this. Reduce is not called. Nothing is emitted downstream. Siblings not yet started are not started. Siblings already running are allowed to finish and their output is discarded. Side channel writes and nested reduce side effects that already occurred **persist**.

Make it visible rather than silent. On skip, write a run manifest entry recording the group id, the parent identity, and the failing child. Once side channels exist, also record the count of side channel records the group had already written. The failure mode to avoid is someone finding orphaned anomaly records later with no idea where they came from.

## 9. Ordering

Children arrive at the barrier in completion order, which is nondeterministic under concurrency. With `ordered: true`, sort children before calling reduce so results are reproducible.

Sort by **emit path**, the tuple of emit indices from the pipeline root down to the record. Assign each record an index at emit time. A record emitted third by a producer whose parent was emitted first has path `(1, 3)`. Lexicographic comparison of paths gives a stable total order that survives nesting.

Do not sort by arrival time and do not sort by a record field.

## 10. Path matching and the DAG

Pipeline ordering depends entirely on path comparison, so define it exactly.

- Resolve every `in`, `out`, and `tee` path relative to the config file's directory.
- Normalize, resolving symlinks and `..`. Compare normalized absolute paths for exact equality. No globbing, no prefix matching.
- Two pipelines declaring the same `out` is a load error, not an implicit merge. If someone wants a merge they should say so with a list-valued `in` on a downstream pipeline.
- An `in` that no pipeline produces must exist on disk at load time.
- **A** `tee` **target read by another pipeline as** `in` **is a real DAG edge** and participates in ordering exactly like an `out`. Without this you get a race that only appears on a fast machine. The tee is declared in config, so this costs nothing in visibility.
- Detect cycles at load time and report the full cycle path in the error.

## 11. Record identity

Error messages need to name a record, but records are arbitrary JSON.

Use best-effort identity, in order:

1. `record[id_field]` if `id_field` is configured and the value is a scalar.
2. `record["id"]` if present and scalar.
3. The emit path, which is always available and always unique.

Never let identity lookup raise. A record with no usable id still needs a legible error.

## 12. Side channels (post-MVP)

A stage may write out of band, but only to a channel the config declared.

```yaml
sidechannels:
  anomalies: anomalies.jsonl

```

```python
ctx.side("anomalies", record)

```

Calling `ctx.side` with an undeclared channel name is an error caught at load time where possible and at runtime otherwise. Records written to a side channel do not count toward the barrier and do not continue downstream.

### Ordering

Side channel files are **append in arrival order, and unordered by contract**. Each written record is stamped with its emit path under a `_path` key.

Do not sort side channels the way children are sorted. Sorting requires buffering the channel until group close, and an anomalies channel on a bad run can be large. Stamping gives constant memory and leaves determinism available to anyone who wants it, without everyone paying for it. The stamp is the runtime's own field on a debug artifact, which is a different thing from making users carry plumbing fields on the main path.

## 13. Resumability (post-MVP)

Post-MVP, support the cheap version only. A pipeline block whose `out` exists and is newer than all of its inputs is skipped, and a `--force` flag disables this. Named jsonl intermediates make this nearly free and it turns a failure at 80% into a partial rerun.

Explicitly out of scope: resuming mid-pipeline with half-finished groups. That needs group state persisted to disk and is a deliberate no. If a pipeline block dies, that block reruns from its input in full.

## 14. Diagnostics

Given this design, the characteristic failure is a run that stops making progress because one group is stuck above zero. Build the tooling for it during step 3 of section 16. It is cheap then and painful to retrofit, and it is the difference between a five minute debug and an afternoon.

- On `SIGUSR1`, and after a configurable idle timeout with no record retired, dump every open group with its id, current count, parent identity, depth, how long it has been open, and which stage its outstanding children are sitting in.
- On normal shutdown with any group still open, dump the same and exit non-zero.
- Track and report at end of run: records read, emitted, retired, dropped by filter, failed. These should balance. When they do not, the accounting has a bug and the totals will find it faster than any test.

## 15. Explicit non-goals

Do not implement `ctx.emit_to("stage_name", record)` or any other way for a stage to name a downstream destination. It works for about a week, and then the config no longer describes the flow, because real edges live in Python and the pipeline shape cannot be seen without reading every stage. This is the exact failure mode the config format exists to prevent. Side channels cover the legitimate cases and stay visible in config.

Do not add node ids and edge lists. Named intermediates plus `tee` cover the cases that motivate DAG syntax.

Do not add rollback or compensating actions for `skip`. Section 8 defines what it does and the manifest makes the residue visible.

## 16. Build order

### MVP

1. Config parse and validation. Stage resolution, signature binding, path normalization, cycle detection, all at load time.
2. Linear pipelines, no groups. `map`, `filter`, `stage`, `tee`. Sync and async functions.
3. Group accounting with `expand`, single level, sequential execution. Build the section 14 diagnostics here.
4. Ordering by emit path.
5. Nesting.
6. `by` and `every`, reusing the same barrier accounting.
7. `on_child_error` and `on_group_incomplete` variants, plus `skip` and the skip manifest.

### Post-MVP

1. Generator stage shape, including streaming `expand`.
2. Concurrency and backpressure, including reserved reduce slots. Then run the determinism tests under pressure.
3. Async-generator stage shape.
4. Side channels.
5. Resumability.

The MVP is complete when the sequential barrier model is boringly correct across all three group constructors, nesting, ordering, and failure modes. The post-MVP work should not change the user-facing topology model; it should make the same model faster, more ergonomic, or easier to rerun.

## 17. Test cases

The implementation is correct when these pass.

### MVP acceptance tests

**Barrier correctness**

1. **Varying group sizes.** Parent A expands to 10 children, parent B to 30. Both reduces receive exactly their own children and neither fires early.
2. **Filter inside a group.** A group of 10 where a filter drops 4. Reduce receives 6 and fires rather than hanging.
3. **Empty group.** Every child filtered out. Reduce receives an empty list.
4. **Nested groups.** Docs to paragraphs, inner reduce per doc, outer reduce across docs. Both barriers fire the correct number of times.
5. **Nested reduce fan out.** An inner reduce that emits 2 records rather than 1. The outer count adjusts correctly per rule 6.
6. **Ordering under nesting.** Children of children sort by emit path.
7. **No leaks.** At end of run, zero groups remain open and the section 14 totals balance.

**Group constructors**

1. **`expand`.** One parent opens one group, emitted children enter that group's child pipeline, and reduce fires once for that parent.
2. **`by`.** Records with the same key reduce together at end of input, and distinct keys reduce separately.
3. **`every`.** Fixed-size windows reduce at the configured size, and the final short window reduces at end of input.

**Stage contract**

1. **Arity.** One-argument and two-argument sync stages both work.
2. **Bad signature.** A stage with a required third argument fails at load time, not on the first record.
3. **Async function.** Async one-argument and two-argument stages both work.
4. **Return a list.** Emitted as one record, not fanned out.
5. **Return and emit.** Raises.
6. **Emit-count assertions.** A `map` that emits twice fails with a useful message. A `filter` that emits twice fails.

**Failure**

1. `on_child_error: partial`**.** One child raises. Reduce fires over survivors and `ctx.failures` has one entry.
2. `on_child_error: fail`**.** Same input, run aborts with the failing record identified.
3. `on_child_error: skip`**.** Same input, reduce is not called, nothing is emitted downstream for that group, and the manifest records the skipped group.
4. **Record identity fallback.** A record with no id field produces an error message naming its emit path.

**Config**

1. **Load-time validation.** Unknown stage name, unknown step key, group with two of `expand`/`by`/`every`, group with none, and a pipeline cycle all fail before any record is read.
2. **Duplicate** `out`**.** Two pipelines writing the same normalized path is a load error.
3. **Tee ordering.** A pipeline reading another's `tee` target runs after it, verified by forcing the producer to be slow.
4. **Path normalization.** `./a.jsonl` and `sub/../a.jsonl` are recognized as the same file.

### Post-MVP acceptance tests

**Callable shapes**

1. **Generator shape.** A sync generator produces the same output as an equivalent `ctx.emit` stage.
2. **Streaming expand.** A producer that yields slowly. The barrier does not fire until the producer is exhausted, even when all previously emitted children have completed.
3. **Async-generator shape.** Async-generator stages produce identical output for equivalent `ctx.emit` logic.

**Concurrency**

1. **Determinism under pressure.** Same input twice at high concurrency. Byte-identical output.
2. **Backpressure holds.** A slow map stage with a large input. Peak in-flight never exceeds `max_in_flight` and memory stays flat.
3. **Reduce deadlock.** `max_in_flight` set below the size of a single group. The run completes rather than hanging. This is the test that proves reserved reduce capacity works.
4. **Deep nesting under pressure.** Nesting depth greater than 1 with `reduce_slots` at the documented minimum. No hang.

**Side channels**

1. **Undeclared channel.** Calling `ctx.side` with an undeclared side channel fails.
2. **Skip with prior side effects.** A group writes 3 side channel records, then a child fails. The 3 records persist, reduce is not called, and the manifest records the group with a side channel count of 3.

**Resume**

1. **Skip completed.** A pipeline whose `out` is newer than its `in` is skipped, and `--force` reruns it.
