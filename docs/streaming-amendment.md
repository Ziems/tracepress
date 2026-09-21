# Amendment: Streaming Runs

An amendment to [`pipeline-spec.md`](pipeline-spec.md). Read that first. This document adds a long-running
execution mode that follows append-only JSONL inputs and processes records as they are written.

The goal is not to turn Trace Miner into a queue, scheduler, or distributed stream processor.
The goal is a practical local command that can sit next to a growing `.jsonl` file and keep the
declared pipelines moving.

## S1. What this adds

Batch mode stays the default:

```bash
uv run trace-miner pipeline.yaml
```

Streaming mode is explicit:

```bash
uv run trace-miner stream pipeline.yaml
```

`stream` watches root input files, reads complete appended JSONL lines, runs the configured
pipelines, and appends outputs to their declared JSONL files.

The config topology remains the same. A pipeline is still a pipeline; a stage still cannot name
a downstream destination; `tee` still names an output path in config. Streaming changes the
runtime's read/write policy, not the shape of the graph.

## S2. MVP boundary

In scope for streaming MVP:

- A new CLI subcommand, `trace-miner stream CONFIG`.
- Follow append-only root input files.
- Treat a newline-terminated JSONL line as the unit of ingestion.
- Ignore partial trailing lines until the newline arrives.
- Append to pipeline `out` and `tee` files instead of overwriting them.
- Maintain an append-only execution journal and derive input checkpoints from it.
- Resume from the journal/checkpoint by default.
- Support `map`, `filter`, `stage`, `tee`, `alert`, and `expand` groups.
- Support `every` groups for complete windows.
- Flush incomplete `every` windows on graceful shutdown.
- Preserve the existing async stage and LM concurrency behavior.
- Reuse the existing compact progress reporting, but make it rate-oriented rather than
  finite-total-oriented where useful.

Explicitly deferred:

- `by` groups that close during an endless stream.
- Watermarks, event time, lateness, and window triggers.
- File rotation discovery.
- Multiple writers to the same input.
- Exactly-once delivery across crashes.
- Persisting open group state.
- Rebuilding output files from the journal.
- Backpressure across the whole pipeline DAG.
- Network sources, message queues, databases, webhooks, or sockets.

## S3. Why `by` is not in the streaming MVP

`by` has no natural close in an unbounded stream. In batch mode, end-of-file closes all `by`
groups. In streaming mode, there may never be an end-of-file, so a `by` group would either stay
open forever or require a new completion rule.

For the streaming MVP, reject configs that contain `by` groups when running `trace-miner stream`,
with a clear error:

```text
`by` groups need a close condition in streaming mode; use batch mode or wait for complete_when.
```

This is intentionally conservative. The later feature should probably be:

```yaml
- group:
    by: session_id
    complete_when:
      field: event_type
      equals: session_end
    pipeline:
      - map: label
    reduce: summarize_session
```

Other plausible later close conditions:

- `idle_for: 5m`
- `max_age: 1h`
- `watermark: timestamp`
- `count: 100`

Do not add these until the simple tailing runtime is solid.

## S4. Input semantics

Streaming inputs are append-only JSONL files.

The runtime records one byte offset per root input path. On startup:

- If a checkpoint exists, seek to the saved offset.
- If no checkpoint exists, default to `--from-start`.
- A `--from-end` flag starts at the current end of the file and only processes new lines.
- A `--reset-checkpoint` flag deletes the saved offsets before starting.

Only complete lines are processed. If the writer appends half a JSON object and flushes without
a newline, Trace Miner waits. When the newline appears, it parses the full line.

Invalid JSON is a run error by default. Later, we can add a dead-letter file, but for MVP a bad
line should stop loudly rather than silently corrupting the stream.

## S5. Execution Journal and Checkpoints

Use a small append-only execution journal, not just a mutable checkpoint file.

This is the bridge between offline search and streaming. A batch run over a fixed file and a
streaming run over a growing file should both be understandable as "admit JSONL records in
order, append runtime facts to the journal, and advance the durable cursor after effects
complete."

The journal is not a full database WAL. It does not promise to reconstruct every output file or
recover arbitrary in-flight group state. It is an audit and recovery log for the ingestion
boundary.

Default journal location:

```text
.trace-miner/runs/<run-id>/journal.jsonl
```

Each journal line is a small JSON object. MVP event types:

```json
{"type":"run_started","run_id":"...","config":"/abs/path/pipeline.yaml"}
{"type":"input_admitted","path":"/abs/path/events.jsonl","offset":0,"next_offset":82,"record_id":"abc"}
{"type":"input_committed","path":"/abs/path/events.jsonl","offset":0,"next_offset":82,"record_id":"abc"}
{"type":"run_stopped","reason":"sigint"}
```

Optional but useful soon after MVP:

```json
{"type":"stage_failed","stage":"label","record_id":"abc","error":"..."}
{"type":"group_reduced","group_id":"g12","stage":"summarize","children":30}
{"type":"output_written","path":"/abs/path/out.jsonl","record_id":"abc"}
```

The important distinction:

- `input_admitted` means the runtime has started processing the line.
- `input_committed` means every declared effect for that input line completed.

On restart, resume from the highest committed `next_offset` for each root input. If the journal
ends with admitted-but-uncommitted records, replay them. This gives explicit at-least-once
behavior instead of hiding the edge case inside a checkpoint file.

Checkpoint state lives next to the config by default:

```text
.trace-miner/checkpoints/<config-hash>.json
```

The checkpoint is a compact snapshot derived from the journal:

```json
{
  "version": 1,
  "config": "/abs/path/pipeline.yaml",
  "journal": "/abs/path/.trace-miner/runs/<run-id>/journal.jsonl",
  "inputs": {
    "/abs/path/events.jsonl": {
      "offset": 12345
    }
  }
}
```

Update the journal first, then update the checkpoint after all effects for a line have
completed: output writes, tee writes, manifest updates, and downstream pipeline writes. A crash
after journal commit but before checkpoint update can be repaired by scanning the journal tail.
A crash before journal commit may replay the last record.

Exactly-once behavior is explicitly out of scope. Users who need idempotence should include a
stable id in their output records and deduplicate downstream.

## S6. Output semantics

Batch mode overwrites pipeline output files. Streaming mode appends.

On startup, do not truncate outputs. This allows a stopped stream process to resume without
destroying previous results. It also makes the at-least-once checkpoint behavior honest: replayed
records may append duplicate outputs if a crash happened after writing but before checkpointing.

For MVP, each JSONL write should be line-buffered and flushed after each input record finishes.
This is slower than buffered batch output but much easier to reason about.

## S7. Pipeline DAG semantics

There are two reasonable ways to stream a multi-pipeline DAG:

1. Write each pipeline output to disk and have downstream pipelines tail those files.
2. Compile the config into an in-memory DAG and route emitted records directly to downstream
   pipeline nodes while still writing declared outputs.

Use option 2 for the eventual implementation. It preserves the config model while avoiding
file-tail races between stages of the same process.

For the streaming MVP, support one source-to-sink DAG inside one process:

- Root inputs are tailed from disk.
- Pipeline `out` and `tee` paths are still written to disk.
- `alert` steps append to the pipeline's alert log. That log is not a DAG edge.
- If another pipeline's `in` matches an upstream `out` or `tee`, records are routed in memory to
  that downstream pipeline after the upstream write succeeds.

This keeps streaming behavior consistent with batch DAG ordering without requiring users to run
one process per intermediate file.

## S8. Group semantics

### `expand`

`expand` is streaming-friendly because each parent record opens its own group. The expand stage
finishes, the sentinel drops, and the reduce can fire for that parent.

In the MVP, expand stages are still ordinary sync or async functions. Generator and async
generator expand stages remain a later improvement.

### `every`

`every` can emit complete windows during the stream.

If `every: 50`, the reduce fires after records 1-50, 51-100, and so on. The final short window
does not fire during normal operation because there is no natural final record in an endless
stream.

On graceful shutdown, flush incomplete `every` windows. On crash, open windows are lost and the
checkpoint should only advance through records whose effects completed.

### `by`

Rejected in streaming MVP. See S3.

## S9. Shutdown

Handle `SIGINT` and `SIGTERM` gracefully:

1. Stop reading new input.
2. Finish records already admitted into the runtime.
3. Flush incomplete `every` windows.
4. Write final outputs and manifest entries.
5. Update checkpoint offsets for completed records.
6. Exit zero.

A hard crash or `SIGKILL` may replay the last completed-but-uncheckpointed record on restart.
That is acceptable for MVP and should be documented.

## S10. Progress and logging

Finite progress bars do not fit an endless stream. In streaming mode, progress should show
ongoing counters rather than a percentage.

MVP display:

```text
[trace-miner] stream: read=128 produced=256 done=256 failed=0 waiting=0
```

Update the display periodically, not on every record. A one-second interval is enough.

If `--progress` is disabled, do not print periodic status. Errors still go to stderr.

## S11. Implementation plan

1. Split the batch runtime around a reusable record-processing core.
   - Keep `run_config` and `run_config_async` behavior unchanged.
   - Extract the parts that run a record through a pipeline so stream mode does not fake input
     by repeatedly calling the batch runner.

2. Add checkpoint support.
   - Define a small `ExecutionJournal` and `CheckpointStore`.
   - Append `input_admitted` before processing a line.
   - Append `input_committed` after all effects for that line complete.
   - Store offsets by normalized absolute input path.
   - Derive checkpoint offsets from committed journal entries.
   - Write checkpoints atomically with temp-file-and-rename.

3. Add a JSONL follower.
   - Seek to the chosen start offset.
   - Read complete lines.
   - Sleep briefly when no complete line is available.
   - Handle truncation by failing loudly in MVP.

4. Add `trace-miner stream`.
   - Parse `--from-start`, `--from-end`, `--reset-checkpoint`, `--checkpoint`, `--poll-interval`,
     `--progress`, and `--no-progress`.
   - Reuse config validation.
   - Add a stream-mode validation pass that rejects `by` groups.

5. Implement streaming DAG routing.
   - Build producer-to-consumer edges from normalized `out` and `tee` paths, just like batch.
   - Tail only root input files.
   - Route upstream outputs to downstream pipelines in memory.

6. Add streaming `every` state.
   - Keep open windows in memory.
   - Fire full windows immediately.
   - Flush short windows only during graceful shutdown.

7. Add tests.
   - Processes appended lines after startup.
   - Ignores partial line until newline arrives.
   - Resumes from checkpoint.
   - `--from-end` skips existing lines.
   - Appends outputs instead of truncating.
   - Routes through a two-pipeline DAG.
   - Rejects `by` groups in stream mode.
   - Flushes incomplete `every` group on graceful shutdown.
   - Replays at least once if a crash happens before `input_committed`.
   - Recovers committed offsets from journal if checkpoint is stale.

## S12. Later features

After the MVP works, the most useful next layer is explicit group closing:

```yaml
complete_when:
  field: event_type
  equals: session_end
```

That unlocks `by` groups for session-style streams without inventing watermarks too early.

After that, consider:

- Dead-letter files for invalid JSON or stage failures.
- File rotation support by inode tracking.
- Persisted open group state.
- Generator and async-generator stages.
- Global backpressure and bounded in-flight records.
- Optional dedupe helpers keyed by `id_field`.

## S13. Non-goals

Do not make the input source pluggable in the first streaming version. JSONL files are enough.

Do not add a daemon supervisor. Users can run the command under their own process manager if
they need restart behavior.

Do not promise exactly-once processing. It is a large feature disguised as a small phrase.

Do not make stages aware of streaming mode. A stage should receive records and context exactly
as it does in batch mode.
