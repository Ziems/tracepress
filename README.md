# Trace Miner

YAML pipelines over JSONL traces. Stage logic is ordinary Python or a DSPy signature.

Use it to find failures, PII, frustration, or misbehavior in long agent runs.

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

That writes shared stages, four batch pipelines, a live-tail stream, and three example traces. Open `output/summaries.jsonl` when it finishes, then try `pii.yaml`, `frustration.yaml`, or `category.yaml` on the same traces.

## Commands

```bash
trace-miner init [directory] [--force]
trace-miner <config.yaml> [--progress]
trace-miner run <config.yaml> [--progress]
trace-miner stream <config.yaml> [--reset-checkpoint] [--progress]
trace-miner serve <config.yaml> [--host 127.0.0.1] [--port 8765]
trace-miner replay <source.jsonl> --into <live.jsonl> [--interval 3] [--reset]
trace-miner stats
```

`init` writes a hello-world project. Omit the directory to be prompted; the default is `hello-world`. `--force` overwrites.

`run` and a bare config path are the same batch command. `--progress` always shows stage bars.

`stream` processes each JSONL line as it is appended. `--reset-checkpoint` starts at byte 0. `--from-end` skips lines already in the file.

`serve` runs a stream and accepts steps over HTTP. A client watches a named pipeline for alerts. Bind beyond localhost with `--host` and `TRACE_MINER_TOKEN`.

`replay` appends records into a live file. `--reset` truncates first. `--limit 1` sends one record.

`stats` prints local usage totals from `~/.trace-miner/`.

## Config

```yaml
from: ./stages                  # module where stage names are resolved
id_field: session_id            # optional; used in error messages

lm:                             # optional LM defaults
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
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

A stage name is a Python function or a DSPy module:

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

Steps are `map`, `filter`, `stage`, `tee`, and `group`. Groups can `expand`, `by`, or `every` in batch. Streams support `expand` and `every`.

LM stages use OpenRouter. Cache and usage live in `~/.trace-miner/`. `trace-miner stats` prints totals.

## Streaming

In one terminal:

```bash
trace-miner stream stream.yaml --reset-checkpoint --progress
```

In another (simulating data streaming in):

```bash
trace-miner replay input/trajectories.jsonl --into input/live.jsonl --reset
```

Summaries append to `output/stream_summaries.jsonl`. Checkpoints live in `.trace-miner/` beside the config.

## Remote alerts

On the pipeline host:

```bash
TRACE_MINER_TOKEN=... trace-miner serve pipeline.yaml --host 0.0.0.0 --port 8765
```

On the eval host:

```python
from trace_miner.client import Client

client = Client("http://pipeline-host:8765")

with client.watch("alerts", rollout_id="r1") as alerts:
    for step in agent.run():
        if alerts.latest:
            raise EvalAborted(alerts.latest.reason)
        client.post_step(step)
```

`pipeline.yaml` names the pipeline. `alert` steps in it append to that pipeline's alert log. The eval machine does not read the files. `alerts.latest` stays empty until an alert fires.

Each posted rollout is split into steps, summarized, then reduced. The reduce decides whether the agent is stuck doing the same thing over and over. The alert step publishes that judgment:

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

`step_summary` on the reduce is the list gathered from the steps. The other reduce outputs merge onto the rollout, which still carries `id_field`. When `doom_loop` is true, the alert event is `{"name": "flag_doom_loop", "reason": "retrying the same pytest command", "rollout_id": "r1"}`.

`trace-miner init` writes this loop as `alerts.yaml` and `client_demo.py`. Serve with `trace-miner serve alerts.yaml`, then run `python client_demo.py` in the new project. The script replays the bundled traces and stops when `flag_sensitive_step` returns a reason.