# Frustration Detection Demo

Terminal-bench traces rarely contain real user emotion, so this batch example uses three slim
copies in `input/trajectories.jsonl` (frustrated user turn, impatient user turn, calm control).
It maps each step through an LM detector and reduces the labels into one finding record per
trajectory.

```bash
export OPENROUTER_API_KEY=...
uv run trace-miner examples/frustration/pipeline.yaml
```

Inspect `examples/frustration/output/findings.jsonl`. Each line looks like:

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
