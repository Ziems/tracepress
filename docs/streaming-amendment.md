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
- Remember one byte offset per root input, and resume from that checkpoint by default.
- Support `map`, `filter`, `stage`, `tee`, `alert`, and `expand` groups.
- Preserve the existing async stage and LM concurrency behavior.
- Reuse the existing compact progress reporting, but make it rate-oriented rather than
  finite-total-oriented where useful.

Explicitly deferred:

- Watermarks, event time, lateness, and window triggers.
- File rotation discovery.
- Multiple writers to the same input.
- Exactly-once delivery across crashes.
- Persisting open group state.
- Backpressure across the whole pipeline DAG.
- Network sources, message queues, databases, webhooks, or sockets.

## S3. Groups in a stream

A stream processes one complete JSONL line at a time, including that line's expand groups and any
downstream pipelines. The next line is read after the current line's outputs are written. `by` and
`every` are load errors in every mode, because a group has to close when its expand stage returns.

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

## S5. Checkpoints

Store one byte offset per root input. Write it after that line's outputs, tees, and downstream
pipelines have been written. Resume from those offsets.

Checkpoint state lives next to the config by default:

```text
.trace-miner/checkpoints/<config-hash>.json
```

```json
{
  "version": 1,
  "config": "/abs/path/pipeline.yaml",
  "inputs": {
    "/abs/path/events.jsonl": {
      "offset": 12345
    }
  }
}
```

Write the checkpoint atomically with temp-file-and-rename. A crash before that write replays the
last line. A crash after it does not.

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

`expand` closes when the expand stage returns, so a parent record's group finishes before the
stream reads the next line. Generator expand stages remain a later improvement.

## S9. Shutdown

Handle `SIGINT` and `SIGTERM` gracefully:

1. Stop reading new input.
2. Finish the line already being processed.
3. Exit zero. That line's checkpoint was written when its effects completed.

A hard crash or `SIGKILL` may replay the last line if the process died before the checkpoint
write. That is acceptable for MVP and should be documented.

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
   - Store offsets by normalized absolute input path.
   - Write the checkpoint after each line's effects complete.
   - Write checkpoints atomically with temp-file-and-rename.

3. Add a JSONL follower.
   - Seek to the chosen start offset.
   - Read complete lines.
   - Sleep briefly when no complete line is available.
   - Handle truncation by failing loudly in MVP.

4. Add `trace-miner stream`.
   - Parse `--from-start`, `--from-end`, `--reset-checkpoint`, `--checkpoint`, `--poll-interval`,
     `--progress`, and `--no-progress`.
   - Reuse config validation. `by` and `every` already fail at load time.

5. Implement streaming DAG routing.
   - Build producer-to-consumer edges from normalized `out` and `tee` paths, just like batch.
   - Tail only root input files.
   - Route upstream outputs to downstream pipelines in memory.

6. Add tests.
   - Processes appended lines after startup.
   - Ignores partial line until newline arrives.
   - Resumes from checkpoint.
   - `--from-end` skips existing lines.
   - Appends outputs instead of truncating.
   - Routes through a two-pipeline DAG.
   - Rejects `by` and `every` at load time.

## S12. Later features

After the MVP works, consider:

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
