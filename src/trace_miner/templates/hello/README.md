# Hello Trace Miner

This project is a small harbor-style workspace. One `stages.py` module, three
slim harbor traces, and four pipelines that reuse those stages.

The traces are real harbor runs (failed ATRX/VEP CRISPR, failed photonic
routing, passed FreeCAD spring clip) with planted signals so each pipeline has
something to find: a fake SSN, a fake phone number, and a frustrated user turn.

Language-model stages need OpenRouter:

```bash
export OPENROUTER_API_KEY=...
```

## Pipelines

| Config | What it does | Output |
| --- | --- | --- |
| `summarize.yaml` | Summarize each step, then diagnose the trajectory | `output/summaries.jsonl` |
| `pii.yaml` | Detect SSN / phone / email in each step | `output/pii.jsonl` |
| `frustration.yaml` | Detect user frustration or impatience | `output/frustration.jsonl` |
| `category.yaml` | Label domain and outcome | `output/categories.jsonl` |

## Run

```bash
trace-miner summarize.yaml
```

Then try the others on the same traces:

```bash
trace-miner pii.yaml
trace-miner frustration.yaml
trace-miner category.yaml
```

## What each file is

- The YAML files are topology only. They name stages and how they connect.
- `stages.py` is ordinary Python plus DSPy signatures. Changing a stage here
  changes every pipeline that names it.
- `input/trajectories.jsonl` is three slim harbor records.

## Change it

Edit a signature in `stages.py`, or copy `pii.yaml` to start a new pipeline
over the same traces.
