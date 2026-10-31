---
name: tracepress-stream-traces
description: Monitor agent traces as they stream in with TracePress, using running per-trace state (lm_fold), provisional reduce results (emit="update"), and simulated streams replayed from stored traces (tp.replay). Use when asked to watch agents live, track progress or stuck detection step by step, or test that a pipeline behaves the same streamed as in batch.
---

# Streaming agent traces with TracePress

TracePress runs the same pipeline code over stored traces and live streams. Rows arrive as
events; a group (such as a trial) produces its final row when it closes. Live sources can also emit
provisional rows (`final=False`) as children arrive.

## When to use

- Watching agent runs live: a running summary per trial, updated every step
- Detecting stuck or looping agents while they run
- Testing streaming behavior offline by replaying stored traces as a stream
- Checking that a pipeline gives the same final results streamed and in batch

For one-off analysis of finished runs, use the `tracepress-analyze-traces` skill instead.

## Steps

1. **Get a stream.** For a real live system that exports OpenTelemetry, use
   `tp.watch.otlp(port=4318)` and point its exporter at `http://localhost:4318`
   (`OTEL_EXPORTER_OTLP_ENDPOINT`). It has `steps` keyed `trace > step`; group by `"trace"`.
   Steps stream as they complete, and a trace closes a couple of seconds after its root span
   ends. Otherwise, replay stored traces in timestamp order. Trials overlap the way they
   actually ran.

   ```python
   import tracepress as tp

   tp.configure(lm="openrouter:openai/gpt-6-luna")         # required; suggested model, or the user's choice
   src = tp.read.harbor("path/to/trials", tasks=["some-task"], limit=2)
   live = tp.replay(src, speed=60)      # 60x real time; speed=None = as fast as possible
   ```

   Fault injection (seeded with `seed=`) tests robustness: `shuffle_within=k` (out-of-order rows),
   `duplicates=p` (rows sent twice) and `drop_closes=p` (traces that never finish; those groups
   close at the end of the stream).

2. **Keep running state per group with `lm_fold`.** Fields on both sides of the arrow are state.
   The LM sees the old value as `previous_<name>` and writes the new one. Other inputs come from the
   newest child, or from group keys.

   ```python
   # Keep the user step (the task) so the note knows what the agent is meant to be doing.
   monitor = src_or_live.table("steps").group_by("trial").lm_fold(
       "progress, source, message, reasoning, tool_calls: list[dict], observation -> progress, stuck: bool, concern",
       instructions="Update your running note on the agent (max 3 sentences); flag stuck/looping.",
       emit="update",            # provisional row after every step (live sources only)
       init={"progress": ""},    # optional; defaults to "", [], 0 or False by type
   )
   ```

   - There's one LM call per child, sequential within a group; groups run concurrently. At about
     6 seconds per call, a 22-step trial takes about 2 minutes.
   - Children are folded in key order. A child that arrives late makes the fold recompute from
     that position, so the final state never depends on arrival order.
   - `emit=tp.every(n)` emits every n children instead of every one.
   - **Context:** keep the task step, and pass trial-level facts (model, agent) in as group keys.
     Pass the step columns, including `reasoning` (the agent's reasoning, when the trace has it),
     so the fold can flag a plan to skip checks or game the tests as it happens.
     Then the note can say what the agent is meant to be doing, not only what it just did.
   - **No Chain of Thought on per-step folds.** Every step is a sequential call, so it slows the
     whole stream. Save it for a final reduce over the finished notes, if there is one.
   - **Use Jev for live flags.** A fold's state is text, so the fold itself needs a generative
     model. But any per-step check where classification alone is enough (is this step stuck?
     risky? off-task?) should run on Jev by default: an `lm_map` with a `Noul` or `Choice` output
     and `lm="typesafe:jev-latest"`. It returns in a fraction of the time, which matters most
     when alerting as steps arrive. See the analyze-traces skill for decision signatures.

3. **Or keep a provisional reduce.** `lm_reduce(..., emit="update")` (or `tp.every(n)`) updates a
   running result as children arrive, and finishes with a full reduce when the group closes. The
   final row therefore equals the batch result.

4. **Consume events.**

   ```python
   async for event in monitor.events():          # inside asyncio / Jupyter
       if isinstance(event, tp.Closed):
           continue
       row = event.row                           # row.final, row.version, row.trial, row.progress...
       print(row.trial, "FINAL" if row.final else f"v{row.version}", row.progress)
   ```

   `collect()` returns only final rows (the newest version of each key). `examples/live_monitor.py`
   is a complete runnable version:
   `uv run python examples/live_monitor.py <trials_dir> --tasks <task> --limit 2 --speed 60`.

5. **Check that streaming matches batch.** Final keys and values must be identical. Compare
   `(r.key, r.values)`, not whole rows: live rows have higher `version`s because of their
   provisional updates.

   ```python
   def contents(rows):
       return [(r.key, r.values) for r in rows]

   batch = contents(pipeline(src).collect())
   for seed in range(3):
       stream = tp.replay(src, seed=seed, shuffle_within=4, duplicates=0.3, drop_closes=0.3)
       assert contents(pipeline(stream).collect()) == batch
   ```

   With a warm cache, replaying costs nothing: duplicates are skipped, and every other prompt
   matches the batch run.

## Pitfalls

- **Provisional rows cascade.** Any `lm_*` operator downstream of an `emit="update"` operator runs
  on every provisional row. Put `.filter(lambda r: r.final)` in between unless you want that.
- **No retractions.** If a provisional row is exploded (`explode` / `lm_extract`) into more children
  than its final version, the extra children stay downstream.
- **A group closes only when its source says so**, or at the end of the stream. Grouping on a
  column with no hierarchy level (e.g. `group_by("agent")`) closes only at the end of the stream.
  Harbor closure for truly live jobs isn't designed yet.
- **`tp.every("30s")` (time-based) is not implemented**; only counts work.
- **A hung LM request stalls a sequential fold** until it times out (`request_timeout`, 30s), and is
  then retried. Watch `tp.last_run().timeouts`.
- **`collect()` raises inside a running event loop.** Use `acollect()` / `events()` there.
