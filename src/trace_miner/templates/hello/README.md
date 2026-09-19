# Hello Trace Miner

This project is a small harbor-style workspace. One `stages.py` module, three
slim harbor traces, and five pipelines that reuse those stages.

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
| `stream.yaml` | Same as summarize, watching `input/live.jsonl` | `output/stream_summaries.jsonl` |

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

## Streaming

`stream.yaml` is the live-tail sibling of `summarize.yaml`. It uses the same
stages and traces, but reads records as they are appended to `input/live.jsonl`.

In one terminal:

```bash
trace-miner stream stream.yaml --reset-checkpoint --progress
```

In another, feed all three fixture traces, three seconds apart:

```bash
trace-miner replay input/trajectories.jsonl --into input/live.jsonl --reset
```

`--reset` truncates the live file first. `--interval` (default 3) is the delay
between appended records. Use `--limit 1` to send a single trace.

Stream outputs are append-only, so remove `output/stream_summaries.jsonl` when
you want a fresh result file. Checkpoint state lives under `.trace-miner/` and
is gitignored. `--reset-checkpoint` starts reading from byte 0 again.

## What each file is

- The YAML files are topology only. They name stages and how they connect.
- `stages.py` is ordinary Python plus DSPy signatures. Changing a stage here
  changes every pipeline that names it. `detect_pii_in_step` uses a Pydantic
  output type (`list[PiiFinding]`) so the model returns structured spans.
- `input/trajectories.jsonl` is three slim harbor records.
- `input/live.jsonl` starts empty so `stream.yaml` can start before replay.
  Gitignored. Recreated empty by `init --force`.
- `stream.yaml` watches `input/live.jsonl`; feed it with `trace-miner replay`.
- `AGENTS.md` is the working guide for coding agents: how to run, add a
  pipeline, and write Python or LM stages.

## Change it

Edit a signature in `stages.py`, or copy `pii.yaml` to start a new pipeline
over the same traces. Prefer a Pydantic model when an LM output is an object
or a list of objects.
