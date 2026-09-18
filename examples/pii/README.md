# PII Detection Demo

This batch example reads three slim harbor copies in `input/trajectories.jsonl` (fake SSN, fake
phone, clean), maps each step through an LM detector, and reduces the labels into one finding
record per trajectory.

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/pii/pipeline.yaml --manifest examples/pii/output/manifest.json
```

Inspect `examples/pii/output/findings.jsonl`. Each line looks like:

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
