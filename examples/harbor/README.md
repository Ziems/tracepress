# Harbor Trajectory Example

The harbor example summarizes a trajectory by mapping over each step, producing one-sentence step
summaries, then reducing those into a failure diagnosis and a compact trajectory summary.

It has two modes:

| Mode | Config | Input | Output | Use it when |
| --- | --- | --- | --- | --- |
| Batch/offline | `pipeline.yaml` | `input/trajectories.jsonl` | `output/one_summary.jsonl` | You want a deterministic one-record run from the fixture. |
| Streaming | `stream.yaml` | `input/live.jsonl` | `output/stream_summaries.jsonl` | You want to watch new JSONL records get processed as they arrive. |

Both modes use the same Python stages. The batch config includes `filter: selected_harbor_trajectory`
so it only processes one fixture record. The streaming config omits that filter because every appended
live record is intended to be processed.

## Batch/offline run

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/harbor/pipeline.yaml --manifest examples/harbor/output/manifest.json
```

That pipeline keeps one selected JSONL line, expands the trajectory into steps, maps each step
through an LM for a one-sentence summary, then reduces those summaries into both a failure
diagnosis and a one-sentence trajectory summary. It writes
`examples/harbor/output/one_summary.jsonl`.

The harbor example sets `lm.max_concurrent: 16`, so the per-step LM summaries can run in
parallel without sending the whole trajectory at once.

## Streaming run

You can also simulate trajectories arriving over time. First prepare an empty live input file:

```bash
uv run python examples/harbor/simulate_stream.py --reset-only
```

Then, in one terminal, start the streaming runtime:

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace stream examples/harbor/stream.yaml --reset-checkpoint --progress
```

In another terminal, append trajectories from the fixture into the live input file:

```bash
uv run python examples/harbor/simulate_stream.py --limit 1 --interval 3
```

The stream reads `examples/harbor/input/live.jsonl` and appends summaries to
`examples/harbor/output/stream_summaries.jsonl`. Increase `--limit` to replay more
trajectories; use `--limit 0` to replay the whole fixture.

`--interval` belongs to the simulator, not the Mine Trace runtime. It is the number of seconds to
wait between appending fixture records into `live.jsonl`.

Streaming writes checkpoint and journal state under `.mine-trace/` beside the config. That state is
ignored by git. Use `--reset-checkpoint` when you want the stream to start reading from byte 0 again.
Outputs are append-only in stream mode, so remove `examples/harbor/output/stream_summaries.jsonl`
between experiments if you want a fresh output file.

The streaming progress line reports:

| Counter | Meaning |
| --- | --- |
| `records` | Complete input JSONL records read from root input files. |
| `emitted` | Records emitted by pipeline stages and reducers. |
| `retired` | Input records fully processed and safe to checkpoint past. |
| `failed` | Records or groups that failed during execution. |
| `groups_open` | Groups/windows still waiting to reduce or flush. |
