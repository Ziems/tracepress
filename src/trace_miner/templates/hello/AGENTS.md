# AGENTS.md

You are working in a Trace Miner project. YAML pipelines over JSONL traces.
Stage logic is ordinary Python or a DSPy signature.

YAML owns topology. Python owns logic. Do not put prompts, routing, or stage
bodies in YAML. Do not invent a second stage module unless the user asks.

## Commands

```bash
trace-miner run <config.yaml> [--progress]
trace-miner serve <config.yaml> [--host 127.0.0.1] [--port 8765]
trace-miner stats
```

`run` executes a batch pipeline. `--progress` always shows stage bars.

`serve` accepts steps over HTTP and pushes alerts. `alerts.yaml` is the
bundled example. Bind beyond localhost with `--host`.

`stats` prints local usage totals from `~/.trace-miner/`.

## First run

LM stages need OpenRouter:

```bash
export OPENROUTER_API_KEY=...
trace-miner run summarize.yaml
```

Then inspect `output/summaries.jsonl`. A successful run prints each output path
and a one-line profile (wall time, plus LM calls / tokens / cost).

The other bundled pipelines reuse the same traces and `stages.py`:

```bash
trace-miner run pii.yaml
trace-miner run frustration.yaml
trace-miner run category.yaml
trace-miner run classify.yaml
```

`alerts.yaml` serves alerts. In one terminal:

```bash
trace-miner serve alerts.yaml
```

In another, from this directory:

```bash
python client_demo.py
```

That posts `input/trajectories.jsonl` one step at a time and exits when
`flag_sensitive_step` returns a reason. The first hit is the planted SSN in
the ATRX trace.

`doom_loop.yaml` alerts from a judgment over recent steps instead of one step.
In one terminal:

```bash
trace-miner serve doom_loop.yaml
```

In another:

```bash
python doom_loop_demo.py
```

The group's `expand` (`split_harbor_steps`) and per-child `map`
(`summarize_harbor_step`) are the same stages `summarize.yaml` uses. Only the
reduce and the two alerts are new:

```yaml
- group:
    expand: split_harbor_steps
    pipeline:
      - map: summarize_harbor_step
    reduce: detect_doom_loop
- alert: flag_doom_loop
- alert: flag_rollout_complete
```

```python
class DetectDoomLoopFlag(dspy.Signature):
    """Decide whether these steps show the agent stuck in a genuine doom loop.

    A doom loop is the agent repeating the same or near-identical action three
    or more times and getting the same unproductive result each time, with no
    new information gained and no change in approach.

    Judge the ACTION-RESULT PAIR, not the surface activity type. Do NOT flag
    varied exploration, verification after a real change, polling on a job
    whose state visibly changes, or a single transient error.
    """

    step_summary: list[str] = dspy.InputField()

    doom_loop: bool = dspy.OutputField(
        desc="True only for a clear, sustained doom loop with no progress."
    )


detect_doom_loop_flag = JevPredict(DetectDoomLoopFlag)


class ExplainDoomLoop(dspy.Signature):
    """A doom loop was already confirmed. Name the exact repeated action and result."""

    step_summary: list[str] = dspy.InputField()

    pattern: str = dspy.OutputField(desc="The exact repeated action and result.")


explain_doom_loop = dspy.Predict(ExplainDoomLoop)
_EXPLAIN_LM = dspy.LM("openrouter/openai/gpt-5.6-luna", api_key=os.environ.get("OPENROUTER_API_KEY"), cache=True, timeout=120)
_EXPLAIN_SLOTS = asyncio.Semaphore(8)


def _explain_doom_loop_sync(step_summary):
    with dspy.context(lm=_EXPLAIN_LM):
        return explain_doom_loop(step_summary=step_summary).pattern


async def detect_doom_loop(steps, ctx):
    step_summary = [step.get("step_summary", "") for step in steps]
    flag = await asyncio.to_thread(detect_doom_loop_flag, step_summary=step_summary)
    if not flag.doom_loop:
        return {**ctx.parent, "doom_loop": False, "pattern": ""}
    async with _EXPLAIN_SLOTS:
        pattern = await asyncio.to_thread(_explain_doom_loop_sync, step_summary)
    return {**ctx.parent, "doom_loop": True, "pattern": pattern}


def flag_doom_loop(record):
    if not record.get("doom_loop"):
        return None
    return record.get("pattern") or "stuck repeating itself"


def flag_rollout_complete(record):
    if not record.get("final"):
        return None
    verdict = "doom loop" if record.get("doom_loop") else "clean"
    return f"rollout complete ({verdict})"
```

`detect_doom_loop` is a **Python reduce**, not a DSPy module — that's why it has
to build `{**ctx.parent, ...}` itself. An LM-kind reduce gets that merge for
free from the runtime; a Python one does not. It exists to make the common
case (no loop) cheap: `detect_doom_loop_flag` is Jev, which answers the bool
in one fast System One call and is dispatched normally (bounded by
`lm.max_concurrent` like any other declared stage), but Jev cannot produce a
freeform `pattern` string. Only a confirmed hit pays for the second,
generative call — bounded by its own small `_EXPLAIN_SLOTS` semaphore, since
a hand-called `dspy.Predict` outside the runtime's own dispatch isn't covered
by `lm.max_concurrent` and won't show up in `trace-miner stats` either. Keep
that semaphore if you copy this pattern — an earlier version without it hit
OpenRouter's per-account in-flight spend cap once several expensive calls
landed at once.

The bool signature's prompt was tightened after a real false-positive hunt:
an earlier looser version ("stuck doing the same thing over and over") flagged
23 of 157 real failed trajectories, and manually reading the raw steps showed
almost all of those were varied, productive work (different files grepped,
different bugs fixed, tests re-run after real changes) that only *looked*
repetitive from a one-line-per-step summary. The tightened version above cut
that to 1 flag out of the same 157. Do not loosen this signature's language
back to "repeats itself" without re-running that kind of check — the LM will
happily pattern-match on "same tool called several times" unless told exactly
what to exclude.

The runtime has no memory of a group across separate `POST /v1/records` calls —
each post opens and closes its own `expand`→`reduce` group. So a reduce that
needs to see many steps needs the *client* to resend them: `doom_loop_demo.py`
keeps a sliding window of the last 8 raw steps and posts
`{"session_id", "steps": window, "final": ...}` on every step. `split_harbor_steps`
expands that `steps` list, so the group and reduce run fresh, over the current
window, at every post. Re-summarizing a step already seen in an earlier window
costs nothing — it hits DSPy's cache.

`final` is a plain field, not a runtime concept. Set `final: true` on a
posted record's last step for that rollout, and check `record.get("final")` in
whatever stage should act on it — that's the whole mechanism for "this rollout
has finished."

`step_summary` is gathered from the children. A client watches the named
pipeline and stops on `alerts.latest.reason`. A DSPy module used directly as
`alert:` needs exactly one bool output. False sends nothing. True sends the
module's other outputs on `alerts.latest.record`.

## Layout

| Path | Role |
| --- | --- |
| `*.yaml` | Pipeline topology only: inputs, outputs, step order, model policy |
| `alerts.yaml` | Named `alerts` pipeline; `in: input/alerts.jsonl` |
| `client_demo.py` | Posts trajectory steps to a running `serve` and stops on an abort |
| `doom_loop.yaml` | Named `doom_loop` pipeline; `in: input/doom_loop.jsonl` |
| `doom_loop_demo.py` | Posts a sliding window of recent steps per trace; watches for a doom loop or `final` |
| `classify.yaml` | One Jev map over a prepared trajectory; no per-step summarize |
| `stages.py` | Shared Python functions and DSPy signatures |
| `input/trajectories.jsonl` | Three slim harbor traces |
| `input/doom_loop_examples.jsonl` | Three real harbor traces: one confirmed doom loop, two confirmed look-alikes |
| `input/alerts.jsonl` | Serve target for `alerts.yaml`. Starts empty. Gitignored. |
| `input/doom_loop.jsonl` | Serve target for `doom_loop.yaml`. Starts empty. Gitignored. |
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
| `classify.yaml` | prepare → Jev categorize → compact | `output/classifications.jsonl` |
| `alerts.yaml` | `alert: flag_sensitive_step` on posted steps | `output/alerts.jsonl` |
| `doom_loop.yaml` | expand+LM summarize+LM reduce over a posted step window, on every post | `output/doom_loop.jsonl` |

Prefer copying one of these over inventing a new shape. Post live steps with
`trace-miner serve`; do not write a helper script.

## How a pipeline is written

```yaml
from: ./stages
id_field: session_id

lm:
  max_concurrent: 64
  default:
    provider: openrouter
    model: deepseek/deepseek-v4.1-flash
  map:
    provider: openrouter
    model: deepseek/deepseek-v4.1-flash
  reduce:
    provider: openrouter
    model: anthropic/claude-sonnet-5

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
- `lm.default`, `lm.map`, `lm.filter`, `lm.expand`, `lm.reduce`, and `lm.alert` each take `provider` and `model`. Providers are `openrouter`, `openai`, and `anthropic`.
- A missing role uses `default`. A missing `provider` uses the default provider. When `default` is omitted, reduce uses OpenRouter `anthropic/claude-sonnet-5` and other LM steps use OpenRouter `deepseek/deepseek-v4.1-flash`.
- A string such as `reduce: some-model` is still a model id on the inherited provider.
- `lm.max_concurrent` caps simultaneous LM calls. Keep it at least 64
  unless the user asks otherwise.
- `lm.cache` defaults to true and uses DSPy's request cache in
  `~/.trace-miner/cache`. Set `cache: false` to force a fresh call every time.
- Override one step with `lm: {provider: ..., model: ...}` only on generative LM stages. That wins over the role. A Jev stage cannot take `lm:` options.

### Step forms

| Form | Meaning |
| --- | --- |
| `map: name` | Stage must emit exactly 1 |
| `filter: name` | Stage must emit 0 or 1 |
| `stage: name` | No emit-count assertion |
| `tee: path.jsonl` | Write the current records aside; the pipeline continues unchanged |
| `alert: name` | Publish an alert; the record continues unchanged |
| `group:` | Expand children, run a child pipeline, then reduce |

`map`, `filter`, and `stage` are the same mechanism. The label is an assertion.

### Groups

This project almost always uses `expand`: one parent record becomes many
children, then one reduce.

- `expand` names a stage. Each emit is a child. The group closes when expand
  finishes.
- `pipeline` runs on every child.
- `reduce` runs once per group with the surviving children.
- `by` and `every` are load errors.

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

Useful `ctx` fields: `emit`, `parent`, `group_id`, `index`, `path`,
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
- A DSPy `alert` needs exactly one `bool` output. Other outputs are the alert body.
- `expand` needs exactly one `list[...]` output.
- `reduce` needs exactly one `list[...]` input. That field is gathered from
  children; every other input comes from `ctx.parent`.
- A parent-level LM reduce (see `analyze_harbor_trajectory`) can take
  `step_summary: list[str]` plus parent metadata.

If a Python function needs `ctx` and an LM, wrap the Predict module. Do not
give the signature `ctx`.

### Jev stages

A `JevPredict` module is still named from YAML (`- map: categorize_trajectory`).
It answers `bool`, `Literal`, and score fields in one OpenRouter Jev call and
does not use `dspy.LM`. Freeform strings stay on `dspy.Predict`. `classify.yaml`
is the bundled example: prepare the trajectory, classify it, compact. No expand.

```python
from typing import Literal

import dspy

from trace_miner.jev import JevPredict


class CategorizeTrajectory(dspy.Signature):
    """Categorize one trajectory."""

    task: str = dspy.InputField()
    reported_passed: bool = dspy.InputField()
    domain: Literal["bioinformatics", "systems", "other"] = dspy.OutputField(desc="Task domain.")
    failed: bool = dspy.OutputField()


categorize_trajectory = JevPredict(CategorizeTrajectory)
```

- Same `OPENROUTER_API_KEY` as other LM stages. No separate Jev key.
- Default model is `jev-1.13`. Pass `model=` to pin another release.
- A score field is an int or float plus `score_fields={"name": ["low", "high"]}`.
- Do not put `lm:` on a Jev stage.
- `filter` and `alert` still need exactly one bool output. `expand` is invalid.

## Adding a pipeline

1. Copy the closest YAML (`pii.yaml` for per-step detection, `summarize.yaml`
   for trajectory diagnosis, `classify.yaml` for a Jev classification,
   `alerts.yaml` for a served alert pipeline,
   `doom_loop.yaml` for a served pipeline that reduces over a step window).
2. Add or reuse stages in `stages.py`. Name the YAML keys exactly after those
   attributes.
3. Point `out:` at a new file under `output/`.
4. Run `trace-miner run new-pipeline.yaml`.
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
- Do not add `lm:` options to a Python stage or a Jev stage.
- Do not change `from:` unless the user wants a different module.
- Do not edit files under `output/` as source. They are run products.
- Do not invent a file-tail helper. Use `trace-miner serve` and `client_demo.py`.

## Debugging

- Load errors name the unknown stage or the invalid LM signature. Fix
  `stages.py` or the YAML name.
- Runtime errors name the stage, the record identity (`session_id` here), and
  the group. Missing LM input fields also print the record's actual keys.
- `tee: output/debug.jsonl` in the middle of a pipeline dumps intermediates.
- Reruns overwrite `output/*.jsonl`. Identical LM calls reuse
  `~/.trace-miner/cache` unless `lm.cache` is false.
