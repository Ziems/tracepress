# AGENTS.md

You are working in a Trace Miner project. Trace Miner is a config-driven JSONL
pipeline runtime for map/filter/reduce over language-model traces.

YAML owns topology. Python owns logic. Do not put prompts, routing, or stage
bodies in YAML. Do not invent a second stage module unless the user asks.

## First run

LM stages need OpenRouter:

```bash
export OPENROUTER_API_KEY=...
trace-miner summarize.yaml
```

Then inspect `output/summaries.jsonl`. A successful run prints each output path
and a one-line profile (wall time, plus LM calls / tokens / cost).

The other bundled pipelines reuse the same traces and `stages.py`:

```bash
trace-miner pii.yaml
trace-miner frustration.yaml
trace-miner category.yaml
```

`stream.yaml` is the live-tail sibling of `summarize.yaml`. In one terminal:

```bash
trace-miner stream stream.yaml --reset-checkpoint --progress
```

In another:

```bash
trace-miner replay input/trajectories.jsonl --into input/live.jsonl --reset
```

That replays all three traces. Use `--limit 1` to send a single trace.

Use `--progress` to show per-stage bars. There is no `--profile` flag; the
summary always prints.

## Layout

| Path | Role |
| --- | --- |
| `*.yaml` | Pipeline topology only: inputs, outputs, step order, model policy |
| `stream.yaml` | Live-tail copy of `summarize.yaml`; `in: input/live.jsonl` |
| `stages.py` | Shared Python functions and DSPy signatures |
| `input/trajectories.jsonl` | Three slim harbor traces |
| `input/live.jsonl` | Replay target. Starts empty. Gitignored. |
| `output/` | Generated JSONL. Gitignored. Safe to overwrite |
| `README.md` | Human walkthrough |

`from: ./stages` in every YAML resolves names against `stages.py`. A stage name
in config must be a module-level attribute there.

## What is already here

Three harbor traces with planted signals:

- ATRX/VEP CRISPR: failed run, fake SSN `123-45-6789`
- Photonic waveguide routing: failed run, frustrated user turn
- FreeCAD spring clip: passed run, fake phone `(555) 010-1234`

| Config | Pattern | Output |
| --- | --- | --- |
| `summarize.yaml` | prepare → expand steps → LM summarize → LM diagnose → compact | `output/summaries.jsonl` |
| `pii.yaml` | expand steps → LM detect → Python reduce | `output/pii.jsonl` |
| `frustration.yaml` | expand steps → LM detect → Python reduce | `output/frustration.jsonl` |
| `category.yaml` | prepare → expand steps → LM summarize → LM categorize → compact | `output/categories.jsonl` |
| `stream.yaml` | same as `summarize.yaml`, reading `input/live.jsonl` | `output/stream_summaries.jsonl` |

Prefer copying one of these over inventing a new shape. Feed `stream.yaml` with
`trace-miner replay`; do not write a helper script.

## How a pipeline is written

```yaml
from: ./stages
id_field: session_id

lm:
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
  max_concurrent: 64

pipelines:
  - in: input/trajectories.jsonl
    out: output/findings.jsonl
    pipeline:
      - map: prepare_harbor_trajectory
      - group:
          expand: split_harbor_steps
          pipeline:
            - map: detect_something
          reduce: aggregate_something
```

Rules:

- `in` and `out` are JSONL paths relative to the config file.
- One YAML file can hold multiple pipeline blocks. If one's `in` is another's
  `out`, they run in that order.
- `id_field` is for error messages, not grouping.
- `lm.max_concurrent` caps simultaneous OpenRouter calls. Keep it at least 64
  unless the user asks otherwise.
- `lm.cache` defaults to true and uses DSPy's request cache. Set
  `cache: false` to force a fresh call every time.
- Override a model on one step with `lm: {model: ...}` only on LM stages.

### Step forms

| Form | Meaning |
| --- | --- |
| `map: name` | Stage must emit exactly 1 |
| `filter: name` | Stage must emit 0 or 1 |
| `stage: name` | No emit-count assertion |
| `tee: path.jsonl` | Write the current records aside; stream continues unchanged |
| `group:` | Expand/by/every children, run a child pipeline, then reduce |

`map`, `filter`, and `stage` are the same mechanism. The label is an assertion.

### Groups

This project almost always uses `expand`: one parent record becomes many
children, then one reduce.

- `expand` names a stage. Each emit is a child. The group closes when expand
  finishes.
- `pipeline` runs on every child.
- `reduce` runs once per group with the surviving children.
- `by: field` groups by a key and only fires after all input is read. Avoid it
  unless the user wants a global aggregation. Streaming rejects `by` until there
  is a close condition for unbounded streams.
- `every: N` is a fixed-size window.

`on_child_error` is `fail` by default. Use `partial` to reduce over survivors,
or `skip` to drop the whole group.

## How a stage is written

Add new stages to `stages.py`. Keep helpers private with a leading underscore.

### Python stages

```python
def keep_failures(record):
    return record if record.get("failed") else None


def split_harbor_steps(record, ctx):
    for step in record.get("steps", []):
        ctx.emit({
            "step_index": step.get("step_id", 0),
            "message": step.get("message", ""),
        })


def aggregate_findings(steps, ctx):
    hits = [step for step in steps if step.get("has_hit")]
    return {
        "session_id": ctx.parent["session_id"],
        "has_hit": bool(hits),
        "findings": hits,
    }
```

- Take `record`, or `record, ctx`. No other parameters.
- Either `return` one record, or call `ctx.emit` and return `None`. Never both.
- Returning a list emits one record that is a list. Fan-out requires `ctx.emit`.
- Returning `None` with zero emits drops the record. That is how filters work.
- `ctx.parent` is the record the group was expanded from. Reducers that need
  `session_id` or `_path` should read it there instead of copying it through
  every child.

Useful `ctx` fields: `emit`, `parent`, `group_id`, `key`, `index`, `path`,
`depth`, `failures`, `log`. `ctx.side` is not supported.

### LM stages

A DSPy signature is an LM stage. Config still says `- map: detect_pii_in_step`.
Do not wrap `Predict` in a Python function unless you need `ctx`. The module
name in YAML is enough; DSPy executes it. Annotate outputs with Pydantic
models when you want nested structure.

```python
from typing import Literal

from pydantic import BaseModel, Field


class PiiFinding(BaseModel):
    kind: Literal["ssn", "phone", "email", "other"]
    evidence: str = Field(description="Short quote of the PII span.")


class DetectPiiInStep(dspy.Signature):
    """Detect personal identifiers in one trajectory step."""

    message: str = dspy.InputField()
    findings: list[PiiFinding] = dspy.OutputField(desc="Every distinct PII span. Empty if none.")


detect_pii_in_step = dspy.Predict(DetectPiiInStep)
```

- Input field names are record keys. Missing keys fail with the stage name and
  the keys the record actually had.
- Output fields merge into the record. Unmentioned keys pass through.
- Pydantic outputs are parsed by DSPy. JSONL writes dump models to dicts.
  See `PiiFinding` in `stages.py`.
- LM stages do not receive `ctx`. If you need `ctx`, wrap the module in a
  Python function; the wrapper is then a Python stage.
- Instructions live on the signature docstring, `Field(description=...)`, and
  `desc=`. Never in YAML.
- `filter` needs exactly one `bool` output.
- `expand` needs exactly one `list[...]` output.
- `reduce` needs exactly one `list[...]` input. That field is gathered from
  children; every other input comes from `ctx.parent`.
- A parent-level LM reduce (see `analyze_harbor_trajectory`) can take
  `step_summary: list[str]` plus parent metadata.

If a Python function needs `ctx` and an LM, wrap the Predict module. Do not
give the signature `ctx`.

## Adding a pipeline

1. Copy the closest YAML (`pii.yaml` for per-step detection, `summarize.yaml`
   for trajectory diagnosis, `stream.yaml` for a live-tail of summarize).
2. Add or reuse stages in `stages.py`. Name the YAML keys exactly after those
   attributes.
3. Point `out:` at a new file under `output/`.
4. Run `trace-miner new-pipeline.yaml`.
5. Read the output JSONL and the profile line. If cost or time looks wrong,
   change the signature or the expand, not the runtime.

When adding traces, append JSON objects to `input/trajectories.jsonl`. Each
line is one record. The bundled records look like harbor ATIF: `session_id`,
`agent`, `steps[]` (`step_id`, `source`, `message`, `tool_calls`,
`observation`), and `eval`.

Reuse `prepare_harbor_trajectory` and `split_harbor_steps` unless the new data
shape is different. Clip long strings before they reach an LM.

## Do not

- Do not put prompt text, few-shot examples, or model-chosen routing in YAML.
- Do not create a new `stages_*.py` for each pipeline.
- Do not treat a returned list as fan-out.
- Do not add `lm:` options to a Python stage.
- Do not change `from:` unless the user wants a different module.
- Do not edit files under `output/` as source. They are run products.
- Do not invent a `simulate_stream.py`. Use `trace-miner replay`.

## Debugging

- Load errors name the unknown stage or the invalid LM signature. Fix
  `stages.py` or the YAML name.
- Runtime errors name the stage, the record identity (`session_id` here), and
  the group. Missing LM input fields also print the record's actual keys.
- `tee: output/debug.jsonl` in the middle of a pipeline dumps intermediates.
- Reruns overwrite `output/*.jsonl`. There is no cache yet; LM calls cost money
  every time.
