---
name: tracepress-explore
description: Explore the results of TracePress pipelines and answer questions from a TracePress workspace (a folder with runs/<run_id>/manifest.json and parquet artifacts). Use when the user asks what their pipeline found, wants counts, breakdowns, examples, or a report of failure modes from LLM-labeled data, wants to trace a summary back to the rows it came from, or wants a follow-up LLM step on part of the existing results.
---

# Exploring TracePress results

A TracePress workspace is a folder of pipeline runs. Each run is `<ws>/runs/<run_id>/` with a `manifest.json` and one parquet file per artifact. The manifest records, for every artifact, the op and instruction that produced it, the model, its parent artifacts, its shape, and its cost. Your job is to answer the user's question from these artifacts, exactly, and to show where each claim comes from.

Find the workspace first: look for a directory containing `runs/*/manifest.json`, and for the project's pipelines file (often `tracepress_pipelines.py`) to see how the artifacts were made.

## Orient

Reading artifacts is free: no model calls. Start with a short script or `uv run python -c`:

```python
import tracepress as tp

ws = tp.Workspace("tracepress_ws")
print(ws.runs())                       # run_id, pipeline, status, output, n_artifacts
print(ws.artifacts(latest=True))       # ref, name, op, n_rows, columns, parents, instruction, model
print(ws.lineage("failure_reasons"))   # the chain of steps back to the raw input
df = ws["failure_reasons"]             # latest artifact with that name
old = ws["20261020-101500-failure_reasons/failure_reasons"]  # a specific run's version
```

`ws.resolve(name)` returns an artifact's full manifest record (instruction, signature, output schema, usage). A run with status `errored` still has the artifacts it finished.

## Answer exactly

- Use pandas for anything exact: counts, shares, filters, joins, group-bys. Do not estimate a count by reading rows.
- Join an aggregate back to its inputs through the `by` columns (e.g. `session_id`) to get the underlying rows.
- LLM outputs can be wrong. When a claim matters, read the underlying rows (the parent artifact, or the raw input) and check it. Say so if a label looks wrong.
- Long cells: print the specific columns you need, truncated, rather than whole frames.

## When the artifacts can't answer

Only then spend model calls. Tell the user what you are about to run and roughly how many calls it costs, and keep it small:

- **One more operator on a subset.** Operators only save artifacts inside a run, so wrap ad-hoc steps in `tp.Run`:

```python
tp.configure(lm=tp.Model("openai/gpt-5-nano", temperature=0))
subset = ws["step_summaries"].query("task_name == 'flight-plan'").head(50)
with tp.budget(max_calls=100), tp.Run(ws, "adhoc-flight-plan") as run:
    subset.lm_filter("step_summary", "The agent skipped verifying its answer", explain=True, save_as="unverified_steps")
```

- **Re-run a registered pipeline on a subset.** Pipelines live in code, so import the project's pipelines file to register them, then call `ws.run` with a `.query()` / `.head()` subset under `tp.budget`. Use the same model the pipeline file configures.
- Harbor data: `tp.harbor_load(path)` reloads trials and steps, and `tp.harbor_agent_taxonomy` / `tp.harbor_command_taxonomy` classify them (see the `tracepress-pipelines` skill).

New artifacts land in the workspace like any others, so the next question can reuse them.

## Report

Answer in markdown. Be direct: lead with the answer, then the evidence. Cite evidence as artifact name plus row id, e.g. `failure_reasons` row 12 or `step_summaries` rows 40–44, so the reader can check it. Give exact counts with their denominator ("22 of 30 runs"). If you ran new model calls, say which artifacts you created and what they cost.
