# LM Mine Trace Example

This example uses a DSPy-backed declarative LM stage. It calls OpenRouter, so set an API key
first:

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/lm/pipeline.yaml --manifest examples/lm/output/manifest.json
```

Then inspect:

- `examples/lm/output/trace_overviews.jsonl`
- `examples/lm/output/manifest.json`

The config spells out the MVP default provider and model explicitly:

```yaml
lm:
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
```

You can omit that block and get the same defaults, or override `lm.default` with another
OpenRouter model id. The stage itself stays in Python, and only prompt-relevant fields are LM
inputs:

```python
class LabelReading(dspy.Signature):
    """Classify one mine sensor reading using its value and nearby field notes."""

    position: int = dspy.InputField()
    reading: float = dspy.InputField()
    notes: str = dspy.InputField()

    material: str = dspy.OutputField()
    rationale: str = dspy.OutputField()

label_reading = dspy.Predict(LabelReading)
```

The surrounding pipeline still owns trace identity and grouping.

## Harbor trajectory example

The harbor example has two modes:

| Mode | Config | Input | Output | Use it when |
| --- | --- | --- | --- | --- |
| Batch/offline | `harbor_pipeline.yaml` | `input/harbor-trajectories.jsonl` | `output/harbor_one_summary.jsonl` | You want a deterministic one-record run from the fixture. |
| Streaming | `harbor_stream_pipeline.yaml` | `input/harbor-live.jsonl` | `output/harbor_stream_summaries.jsonl` | You want to watch new JSONL records get processed as they arrive. |

Both modes use the same Python stages. The batch config includes `filter: selected_harbor_trajectory`
so it only processes one fixture record. The streaming config omits that filter because every appended
live record is intended to be processed.

### Batch/offline harbor run

Run the one-record fixture pipeline:

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/lm/harbor_pipeline.yaml --manifest examples/lm/output/harbor_manifest.json
```

That pipeline keeps one selected JSONL line, expands the trajectory into steps, maps each step
through an LM for a one-sentence summary, then reduces those summaries into both a failure
diagnosis and a one-sentence trajectory summary. It writes
`examples/lm/output/harbor_one_summary.jsonl`.

The harbor example sets `lm.max_concurrent: 4`, so the per-step LM summaries can run in
parallel without sending the whole trajectory at once.

### Streaming harbor run

You can also simulate trajectories arriving over time. First prepare an empty live input file:

```bash
uv run python examples/lm/simulate_harbor_stream.py --reset-only
```

Then, in one terminal, start the streaming runtime:

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace stream examples/lm/harbor_stream_pipeline.yaml --reset-checkpoint --progress
```

In another terminal, append trajectories from the fixture into the live input file:

```bash
uv run python examples/lm/simulate_harbor_stream.py --limit 1 --interval 3
```

The stream reads `examples/lm/input/harbor-live.jsonl` and appends summaries to
`examples/lm/output/harbor_stream_summaries.jsonl`. Increase `--limit` to replay more
trajectories; use `--limit 0` to replay the whole fixture.

`--interval` belongs to the simulator, not the Mine Trace runtime. It is the number of seconds to
wait between appending fixture records into `harbor-live.jsonl`.

Streaming writes checkpoint and journal state under `.mine-trace/` beside the config. That state is
ignored by git. Use `--reset-checkpoint` when you want the stream to start reading from byte 0 again.
Outputs are append-only in stream mode, so remove `examples/lm/output/harbor_stream_summaries.jsonl`
between experiments if you want a fresh output file.

The streaming progress line reports:

| Counter | Meaning |
| --- | --- |
| `records` | Complete input JSONL records read from root input files. |
| `emitted` | Records emitted by pipeline stages and reducers. |
| `retired` | Input records fully processed and safe to checkpoint past. |
| `failed` | Records or groups that failed during execution. |
| `groups_open` | Groups/windows still waiting to reduce or flush. |

## PII detection demo

The PII example is batch-only. It reads three slim harbor copies in
`input/pii-trajectories.jsonl` (fake SSN, fake phone, clean), maps each step through an LM
detector, and reduces the labels into one finding record per trajectory.

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/lm/pii_pipeline.yaml --manifest examples/lm/output/pii_manifest.json
```

Inspect `examples/lm/output/pii_findings.jsonl`. Each line looks like:

```json
{
  "session_id": "...",
  "trajectory_path": "...",
  "has_pii": true,
  "findings": [
    {"step_index": 2, "pii_kind": "ssn", "evidence": "123-45-6789"}
  ]
}
```

## Harbor category demo

The category example is batch-only. It reads three slim harbor trajectories in
`input/category-trajectories.jsonl` (bioinformatics failed, systems failed, data-processing
passed), summarizes each step, and reduces those summaries into a domain, outcome, and
one-sentence trajectory summary.

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/lm/category_pipeline.yaml --manifest examples/lm/output/category_manifest.json
```

Inspect `examples/lm/output/category_findings.jsonl`. Each line looks like:

```json
{
  "session_id": "...",
  "trajectory_path": "...",
  "task_name": "...",
  "domain": "bioinformatics",
  "outcome": "wrong_output",
  "trajectory_summary": "..."
}
```

## Frustration detection demo

Terminal-bench traces rarely contain real user emotion, so this batch example uses three slim
copies in `input/frustration-trajectories.jsonl` (frustrated user turn, impatient user turn,
calm control). It maps each step through an LM detector and reduces the labels into one finding
record per trajectory.

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/lm/frustration_pipeline.yaml --manifest examples/lm/output/frustration_manifest.json
```

Inspect `examples/lm/output/frustration_findings.jsonl`. Each line looks like:

```json
{
  "session_id": "...",
  "trajectory_path": "...",
  "has_frustration": true,
  "findings": [
    {"step_index": 5, "frustration_kind": "frustrated", "evidence": "This is taking forever"}
  ]
}
```
