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
trace-miner replay <source.jsonl> --into <live.jsonl> [--interval 3] [--reset]
trace-miner stats
```

`init` writes a hello-world project. Omit the directory to be prompted; the default is `hello-world`. `--force` overwrites.

`run` and a bare config path are the same batch command. `--progress` always shows stage bars.

`stream` processes each JSONL line as it is appended. `--reset-checkpoint` starts at byte 0. `--from-end` skips lines already in the file.

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
