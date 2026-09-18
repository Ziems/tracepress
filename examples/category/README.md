# Harbor Category Demo

This batch example reads three slim harbor trajectories in `input/trajectories.jsonl`
(bioinformatics failed, systems failed, data-processing passed), summarizes each step, and
reduces those summaries into a domain, outcome, and one-sentence trajectory summary.

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/category/pipeline.yaml --manifest examples/category/output/manifest.json
```

Inspect `examples/category/output/findings.jsonl`. Each line looks like:

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
