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

There is also a one-record trajectory example using `input/harbor-trajectories.jsonl`:

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
