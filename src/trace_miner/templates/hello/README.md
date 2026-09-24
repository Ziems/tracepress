# Hello Trace Miner

Find a failure, planted PII, a frustrated user, or a doom loop in three example harbor traces.

Language-model stages need OpenRouter:

```bash
export OPENROUTER_API_KEY=...
trace-miner run pipeline.yaml --in input/trajectories.jsonl --out output/hello.jsonl
```

`pipeline.yaml` summarizes each step, then reduces those summaries to a label. Open `output/hello.jsonl` when it finishes.

```yaml
from: ./stages
id_field: session_id

lm:
  default:
    provider: openrouter
    model: deepseek/deepseek-v4.1-flash

pipelines:
  - pipeline:
      - group:
          expand: split_steps
          pipeline:
            - map: summarize_step
          reduce: label_trajectory
      - map: compact
```

```python
class SummarizeStep(dspy.Signature):
    """Name the concrete action in this step."""

    step_id: int = dspy.InputField()
    message: str = dspy.InputField()
    step_summary: str = dspy.OutputField(desc="One sentence, at most 15 words.")

summarize_step = dspy.Predict(SummarizeStep)


class LabelTrajectory(dspy.Signature):
    """Name what this agent trajectory was trying to do."""

    step_summary: list[str] = dspy.InputField()
    label: str = dspy.OutputField(desc="A few words naming the task.")

label_trajectory = dspy.Predict(LabelTrajectory)
```

`step_summary` on the reduce is a list gathered from the mapped steps. `compact` keeps `session_id`, `task_name`, and `label`.

## Cookbook

| Config | What it does | Output |
| --- | --- | --- |
| `cookbook/summarize.yaml` | Summarize each step, then diagnose the trajectory | `output/summaries.jsonl` |
| `cookbook/pii.yaml` | Detect SSN / phone / email in each step | `output/pii.jsonl` |
| `cookbook/frustration.yaml` | Detect user frustration or impatience | `output/frustration.jsonl` |
| `cookbook/category.yaml` | Label domain and outcome | `output/categories.jsonl` |
| `cookbook/classify.yaml` | Classify a trajectory with Jev, without summarizing each step | `output/classifications.jsonl` |
| `cookbook/alerts.yaml` | Alert when a step contains an SSN, a phone number, or a frustrated user turn | `output/alerts.jsonl` |
| `cookbook/doom_loop.yaml` | Alert when the agent is stuck repeating itself | `output/doom_loop.jsonl` |


The traces are real harbor runs with three planted signals: a fake SSN, a fake phone number, and a user turn that says "taking forever".

## Run

```bash
trace-miner run cookbook/summarize.yaml --in input/trajectories.jsonl --out output/summaries.jsonl
trace-miner run cookbook/pii.yaml --in input/trajectories.jsonl --out output/pii.jsonl
trace-miner run cookbook/frustration.yaml --in input/trajectories.jsonl --out output/frustration.jsonl
trace-miner run cookbook/category.yaml --in input/trajectories.jsonl --out output/categories.jsonl
trace-miner run cookbook/classify.yaml --in input/trajectories.jsonl --out output/classifications.jsonl
```

`cookbook/summarize.yaml` is the shape of a batch pipeline. YAML names the stages. `cookbook/stages.py` is ordinary Python or a DSPy signature.

```yaml
from: ./stages
id_field: session_id

lm:
  max_concurrent: 256
  default:
    provider: openrouter
    model: deepseek/deepseek-v4.1-flash
  reduce:
    provider: openrouter
    model: anthropic/claude-sonnet-5

pipelines:
  - pipeline:
      - map: prepare_harbor_trajectory
      - group:
          expand: split_harbor_steps
          pipeline:
            - map: summarize_harbor_step
          reduce: analyze_harbor_trajectory
      - map: compact_harbor_summary
```

A group expands one parent, runs a child pipeline, then reduces. A `list` input on the reduce is gathered from the children. The signatures below are the shape. The full versions are in `cookbook/stages.py`.

```python
def split_harbor_steps(record, ctx):
    for step in record.get("steps", []):
        ctx.emit({"step_index": step.get("step_id", 0), "message": step.get("message", "")})


class SummarizeHarborStep(dspy.Signature):
    """Name the concrete action in this step. Skip empty or no-op steps."""

    message: str = dspy.InputField()
    tool_calls: list[str] = dspy.InputField()
    step_summary: str = dspy.OutputField(desc="One sentence naming what the agent did.")

summarize_harbor_step = dspy.Predict(SummarizeHarborStep)
```



## Alerts

`cookbook/alerts.yaml` is one alert stage. `flag_sensitive_step` returns a reason for a planted SSN, phone number, or "taking forever". With `--out`, the step is written to that JSONL. The alert is written under `cookbook/.trace-miner/alerts/`.

```yaml
from: ./stages
id_field: session_id

pipelines:
  - name: alerts
    pipeline:
      - alert: flag_sensitive_step
```

```python
def flag_sensitive_step(record):
    text = f"{record.get('message') or ''} {record.get('observation') or ''}"
    if _SSN.search(text):
        return "ssn"
    if _PHONE.search(text):
        return "phone number"
    if "taking forever" in text.lower():
        return "frustrated user"
    return None
```

In one terminal:

```bash
trace-miner serve cookbook/alerts.yaml --out output/alerts.jsonl
```

In another, from this directory:

```bash
python cookbook/client_demo.py
```

The demo posts each step and stops on the planted SSN:

```text
aborted terminal-bench/atrx-vep-crispr step 2: ssn
```

The same loop, without the demo:

```python
from trace_miner.client import Client

client = Client("http://127.0.0.1:8765")
with client.watch_events(session_id="r1") as events:
    client.post_step(step)
    for alert in events.take():
        if alert.name == "flag_sensitive_step":
            raise SystemExit(alert.reason)
```

`take()` returns the alerts that have arrived since the last call. The phone number and the frustrated turn fire if you remove the earlier signal.

## Doom loop

To judge the whole rollout, expand the steps, reduce, then alert. The server remembers every step already submitted for that `session_id` and runs this pipeline on that list. No alert means keep going.

```yaml
from: ./stages
id_field: session_id

lm:
  reduce: openai/gpt-5.6-luna

pipelines:
  - name: doom_loop
    pipeline:
      - group:
          expand: split_harbor_steps
          pipeline:
            - map: summarize_harbor_step
          reduce: detect_doom_loop
      - alert: flag_doom_loop
```

```python
class DetectDoomLoop(dspy.Signature):
    """Decide if these summaries show an agent stuck repeating itself."""

    step_summary: list[str] = dspy.InputField()
    doom_loop: bool = dspy.OutputField()
    pattern: str = dspy.OutputField(desc="What it keeps repeating, or an empty string.")

detect_doom_loop = dspy.Predict(DetectDoomLoop)


def flag_doom_loop(record):
    if not record.get("doom_loop"):
        return None
    return record.get("pattern") or "stuck repeating itself"
```

When `doom_loop` is true, the event is `{"name": "flag_doom_loop", "reason": "retrying the same pytest command", "session_id": "r1"}`. The alert step returns a reason or `None`. The copy in `cookbook/stages.py` asks Jev for the bool, then writes `pattern` only when that bool is true.

In one terminal:

```bash
trace-miner serve cookbook/doom_loop.yaml --out output/doom_loop.jsonl
```

In another, from this directory:

```bash
python cookbook/doom_loop_demo.py
```

Use cases, and what each one should do, are in `cookbook/doom_loop/README.md`. The demo posts the three bundled traces plus three more from `input/doom_loop_examples.jsonl`. `coq-block-bound` should stop mid-trace. The two look-alikes should stay quiet.

The same check, without the demo:

```python
with client.watch_events(session_id="s1") as events:
    client.post_step({"session_id": "s1", **step})
    for alert in events.take():
        if alert.name == "flag_doom_loop":
            raise SystemExit(alert.reason)
```



## Change it

Edit a signature in `stages.py`, or copy a pipeline from `cookbook/` to start a new one over the same traces. `AGENTS.md` is the guide for adding a pipeline.