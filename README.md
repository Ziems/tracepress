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
trace-miner run pipeline.yaml --in input/trajectories.jsonl --out output/hello.jsonl
```

That writes `pipeline.yaml`, `stages.py`, and three example traces. Open `output/hello.jsonl` when it finishes. More pipelines, including alerts and a doom loop, are in `cookbook/`.

## Commands

```bash
trace-miner init [directory] [--force]
trace-miner run <config.yaml> --in <file> --out <file> [--progress]
trace-miner serve <config.yaml> [--out <file>] [--host 127.0.0.1] [--port 8765]
trace-miner stats
```

`init` writes a hello-world project. Omit the directory to be prompted; the default is `hello-world`

`run` reads JSONL from `--in` and writes JSONL to `--out`. Repeat `--in` to read several files in order. `--progress` always shows stage bars.

`serve` accepts steps over HTTP and pushes alerts. It does not take `--in`. `--out` also writes the finished records to JSONL. A client watches a named pipeline. Bind beyond localhost with `--host`.

`stats` prints local usage totals from `~/.trace-miner/`.

## Config

```yaml
from: ./stages                  # module where stage names are resolved
id_field: session_id            # optional; names the record in errors and alerts

lm:                             # optional LM defaults
  max_concurrent: 256
  cache: true                   # LM cache under ~/.trace-miner/cache
  max_tokens: 4096               # optional cap on generated tokens per call
  default:
    provider: openrouter        # openrouter, openai, or anthropic
    model: deepseek/deepseek-v4.1-flash
  map:
    provider: openrouter
    model: deepseek/deepseek-v4.1-flash
  reduce:
    provider: openrouter
    model: anthropic/claude-sonnet-5

pipelines:
  - pipeline:
      - filter: has_steps
      - group:
          expand: split_steps      # one parent -> many children
          pipeline:
            - map: detect_failure  # LM stage; once per child
          reduce: diagnose_trajectory
```

`default`, `map`, `filter`, `expand`, `reduce`, and `alert` each take a `provider` and `model`. A missing role uses `default`. A missing `provider` inside a role uses the default provider. If `default` is omitted, reduce uses OpenRouter `anthropic/claude-sonnet-5` and every other LM step uses OpenRouter `deepseek/deepseek-v4.1-flash`. A string such as `reduce: some-model` is the model id on the inherited provider. A per-step `lm.provider` or `lm.model` overrides the role.

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

Steps are `map`, `filter`, `stage`, `tee`, `alert`, and `group`. A group expands one parent, runs a child pipeline, then reduces.

## Real Time Alerts

On the pipeline host:

```bash
trace-miner serve cookbook/alerts.yaml --out output/alerts.jsonl --host 0.0.0.0 --port 8765
```

`cookbook/alerts.yaml` names the pipeline `alerts`. `flag_sensitive_step` returns a reason for a planted SSN, phone number, or "taking forever". With `--out`, the step is written to that JSONL. The alert is written under `cookbook/.trace-miner/alerts/`.

On the eval host:

```python
from trace_miner.client import Client

client = Client("http://pipeline-host:8765")
with client.watch_events(session_id="r1") as events:
    client.post_step(step)
    for alert in events.take():
        if alert.name == "flag_sensitive_step":
            raise SystemExit(alert.reason)
```

`take()` returns the alerts that have arrived since the last call. The eval machine does not read the files.

`python cookbook/client_demo.py` in the new project posts the bundled traces and stops on the planted SSN:

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