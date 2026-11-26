---
name: tracepress-stream-traces
description: Monitor agent traces as they stream in with TracePress, using running per-trace state (lm_fold), provisional reduce results (emit="update"), simulated streams replayed from stored traces (tp.replay), and a server that keeps pipelines running for other processes to listen to (tracepress serve, tp.connect). Use when asked to watch agents live, track progress or stuck detection step by step, keep a pipeline running continuously, alert (Slack, paging) or feed a dashboard as traces arrive, or test that a pipeline behaves the same streamed as in batch.
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
- Keeping pipelines running on a server, so other processes (a Slack bot, a dashboard, the agent
  runner) can act on their output as traces arrive

For one-off analysis of finished runs, use the `tracepress-analyze-traces` skill instead.

## Setup

- Check TracePress is installed with `python -c "import tracepress"`. If it isn't, add the
  `tracepress` package from PyPI with the project's own package manager, following the install
  steps at https://github.com/Ziems/tracepress#install. Inside the TracePress repo itself, use the
  repo's own environment.
- If something in this skill isn't supported, check you're on the latest version
  (`python -c "from importlib.metadata import version; print(version('tracepress'))"`) and, if a
  newer release is on PyPI, upgrade the package the same way.
- Configure a model before running LM operators; see Setup in the `tracepress-analyze-traces` skill.

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

6. **Keep pipelines running on a server** when something else should act on the results (a Slack
   bot, a dashboard, a process that stops runaway sessions), or several pipelines should share one
   stream. Any client can stream in or register, and any client can listen; they don't have to be
   the same process or machine.

   ```bash
   tracepress serve --port 7700 --lm openrouter:openai/gpt-6-luna
   ```

   ```python
   client = tp.connect("localhost:7700")
   client.stream(tp.watch.otlp())           # background; or tp.replay(src) to test, .wait() for it to finish
   live = client.source                     # the server's stream: build pipelines on this, not on src
   doom_loops = live.steps.group_by("trace").lm_fold(
       "notes, source, message, tool_calls -> notes, doom_loop: bool", emit="update")
   client.register(name="doom_loops", pipeline=doom_loops)

   # in any other process
   async for e in tp.connect("localhost:7700").listen("doom_loops"):
       if e.row is not None and e.row.doom_loop:
           stop_session(e.row.trace)
   ```

   - A pipeline registered after traces arrived catches up on the stream so far; registering the
     same name again replaces it. `client.pipelines()` shows each one's status and row count.
   - `listen(name, from_start=True)` sends the rows produced so far first, then new ones.
     Events are `added`, `updated` (a newer version of a row) or `closed`; `e.row` is a normal row.
   - `client.push(rows, table="steps")` sends rows directly, plus `tp.Closed(prefix)` to finish a
     group. Tables must be declared first: `client.stream(...)` does it, or `client.declare(schemas)`.
   - The server's own `--lm` (and its environment and `.env`) is the default LM for registered
     pipelines.

## Pitfalls

- **Registering ships code.** The pipeline is pickled with its lambdas, and the server runs it.
  Only run a server for clients you trust. It binds to localhost; any other `--host` needs
  `--token`, and clients pass it as `tp.connect(addr, token=...)`. Functions a pipeline imports
  from the user's own modules must be importable on the server too (lambdas and functions defined
  in the script travel with it).
- **Build registered pipelines on `client.source`.** A pipeline built on a local reader runs on
  the server against that reader, not the stream. Collecting a `client.source` pipeline locally
  raises; register it instead.
- **The server keeps every event in memory** to catch up late pipelines. Pass `--retain N` to
  cap events per table on a long-running server.

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
- **`events()` raises the run's error** once the events before it are delivered. Leaving the loop
  early stops the run and its LM calls. `tp.last_run()` reads live: `.status` is `running` until
  the run ends, and the counts are the totals so far.
