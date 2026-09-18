# LM Trace Miner Example

This example uses a DSPy-backed declarative LM stage. It calls OpenRouter, so set an API key
first:

```bash
export OPENROUTER_API_KEY=...
uv run trace-miner examples/lm/pipeline.yaml --manifest examples/lm/output/manifest.json
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
