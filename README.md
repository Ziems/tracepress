# Trace Miner

Find failures, PII, frustration, or misbehavior in agent traces.

YAML names the pipeline. Stage logic is ordinary Python or a DSPy signature.

## Setup

```bash
uv tool install git+https://github.com/Ziems/trace-miner.git
```

The GitHub repo is private, so that command needs credentials that can clone it.

## Start a Project

```bash
export OPENROUTER_API_KEY=...
trace-miner init my-trace-pipelines
cd my-trace-pipelines
trace-miner summarize.yaml
```

That writes `stages.py`, three example traces, and six demo pipelines: `summarize.yaml`, `pii.yaml`, `frustration.yaml`, `category.yaml`, `stream.yaml`, and `alerts.yaml`. Open `output/summaries.jsonl` when it finishes.

## Commands

```bash
trace-miner init [directory] [--force]
trace-miner <config.yaml> [--progress]
trace-miner run <config.yaml> [--progress]
trace-miner stream <config.yaml> [--reset-checkpoint] [--from-end] [--progress]
trace-miner serve <config.yaml> [--host 127.0.0.1] [--port 8765]
trace-miner replay <source.jsonl> --into <live.jsonl> [--interval 3] [--limit N] [--reset]
trace-miner stats
```

`init` writes a hello-world project. Omit the directory to be prompted; the default is `hello-world`. `--force` overwrites.

`run` and a bare config path are the same batch command. `--progress` always shows stage bars.

`stream` processes each JSONL line as it is appended. `--reset-checkpoint` starts at byte 0. `--from-end` skips lines already in the file.

`serve` runs a stream and accepts steps over HTTP. A client watches a named pipeline for alerts. Bind beyond localhost with `--host`.

`replay` appends records into a live file. `--reset` truncates first. `--limit 1` sends one record.

`stats` prints local usage totals from `~/.trace-miner/`.

## Config

```yaml
from: ./stages                  # module where stage names are resolved
id_field: session_id            # optional; names the record in errors and alerts

lm:                             # optional LM defaults
  provider: openrouter          # openrouter, openai, or anthropic
  default: deepseek/deepseek-v4.1-flash  # must be a model for that provider
  max_concurrent: 64
  cache: true                   # LM cache under ~/.trace-miner/cache

pipelines:
  - in: input/trajectories.jsonl
    out: output/findings.jsonl
    pipeline:
      - filter: has_steps
      - group:
          expand: split_steps      # one parent -> many children
          pipeline:
            - map: detect_failure  # LM stage; once per child
          reduce: diagnose_trajectory
```

A stage name is a module-level function or DSPy module. YAML uses that attribute (`detect_failure`), not the class.

```python
def has_steps(record):
    return record if record.get("steps") else None


def split_steps(record, ctx):
    for step in record["steps"]:
        ctx.emit({"step_index": step["step_id"], "message": step["message"]})


class DetectFailure(dspy.Signature):
    """Flag a concrete failure mode in one agent step, or none."""
    step_index: int = dspy.InputField()
    message: str = dspy.InputField()
    failure_mode: str = dspy.OutputField()
    evidence: str = dspy.OutputField()

detect_failure = dspy.Predict(DetectFailure)
```

Steps are `map`, `filter`, `stage`, `tee`, `alert`, and `group`. Groups can `expand`, `by`, or `every` in batch. Streams support `expand` and `every`.

## Streaming

In one terminal:

```bash
trace-miner stream stream.yaml --reset-checkpoint --progress
```

In another:

```bash
trace-miner replay input/trajectories.jsonl --into input/live.jsonl --reset
```

Summaries append to `output/stream_summaries.jsonl`. Checkpoints live in `.trace-miner/` beside the config.

## Remote alerts

On the pipeline host:

```bash
trace-miner serve alerts.yaml --host 0.0.0.0 --port 8765
```

`alerts.yaml` names the pipeline `alerts`. `flag_sensitive_step` returns a reason for a planted SSN, phone number, or "taking forever". The step is written to `output/alerts.jsonl`. The alert is written under `.trace-miner/alerts/`.

On the eval host:

```python
from trace_miner.client import Client

client = Client("http://pipeline-host:8765")
with client.watch("alerts", session_id="r1") as alerts:
    client.post_step(step)
    if alerts.latest:
        raise SystemExit(alerts.latest.reason)
```

`alerts.latest` stays empty until an alert fires. The eval machine does not read the files.

`python client_demo.py` in the new project replays the bundled traces and stops on the planted SSN:

```text
aborted terminal-bench/atrx-vep-crispr step 2: ssn
```

To judge the whole rollout, expand the steps, reduce, then alert. A `list` input on the reduce is gathered from the children. Other reduce outputs merge onto the rollout. The alert event copies `id_field` (`session_id` here). The alert step returns a reason or `None`.

```yaml
- group:
    expand: split_steps
    pipeline:
      - map: summarize_step
    reduce: detect_doom_loop
- alert: flag_doom_loop
```

```python
class SummarizeStep(dspy.Signature):
    """Name the concrete action in this step."""

    message: str = dspy.InputField()
    tool_calls: list[str] = dspy.InputField()

    step_summary: str = dspy.OutputField(desc="One sentence naming what the agent did.")

summarize_step = dspy.Predict(SummarizeStep)


class DetectDoomLoop(dspy.Signature):
    """Decide if these summaries show an agent stuck repeating itself."""

    step_summary: list[str] = dspy.InputField()

    doom_loop: bool = dspy.OutputField(desc="True when the agent is stuck doing the same thing over and over.")
    pattern: str = dspy.OutputField(desc="What it keeps repeating, or an empty string.")

detect_doom_loop = dspy.Predict(DetectDoomLoop)


def flag_doom_loop(record):
    if not record.get("doom_loop"):
        return None
    return record.get("pattern") or "stuck repeating itself"
```

When `doom_loop` is true, the event is `{"name": "flag_doom_loop", "reason": "retrying the same pytest command", "session_id": "r1"}`.
