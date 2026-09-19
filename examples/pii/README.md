# PII Detection Demo

This batch example reads three slim harbor copies in `input/trajectories.jsonl` (fake SSN, fake
phone, clean), maps each step through an LM detector, and reduces the labels into one finding
record per trajectory. The detector output is a `list[PiiFinding]` Pydantic type so each step
can return several structured spans instead of one flat kind/evidence pair.

```bash
export OPENROUTER_API_KEY=...
uv run trace-miner examples/pii/pipeline.yaml
```

Inspect `examples/pii/output/findings.jsonl`. Each line looks like:

```json
{
  "session_id": "...",
  "trajectory_path": "...",
  "has_pii": true,
  "findings": [
    {"step_index": 2, "kind": "ssn", "evidence": "123-45-6789"}
  ]
}
```
