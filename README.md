# Trace Miner

Trace Miner is a config-driven JSONL pipeline runtime for map/filter/reduce workflows over language model
traces. Pipeline topology lives in YAML; stage logic lives in ordinary Python functions.

It is built for things like:

- finding failure modes over very long agentic trajectories
- interrupting long running failed rollouts in real time (pending a notification mechanism)
- identifying reward hacking or misalignment
- identifying user frustration, impatience, or abandonment in a session
- spotting PII or other policy violations buried in tool traces
- labeling domain, outcome, and error type across a corpus of runs
- finding behavioral differences in model performance across agents or versions
- and much much more...



## Setup

```bash
uv tool install git+https://github.com/Ziems/trace-miner.git
trace-miner --help
```

The GitHub repo is private, so that command needs credentials that can clone it.

## Start a Project

`init` writes a hello-world project with shared LM stages, four batch pipelines,
and a live-tail stream config:

```bash
export OPENROUTER_API_KEY=...
trace-miner init my-traces
cd my-traces
trace-miner summarize.yaml
```

That scaffolds `stages.py`, three example harbor traces, `AGENTS.md`, and
`summarize.yaml` / `pii.yaml` / `frustration.yaml` / `category.yaml` /
`stream.yaml`. The summarize pipeline summarizes each step and diagnoses the
trajectory. The traces include a fake SSN, a fake phone number, and a
frustrated user turn so the other pipelines have something to find. Open
`output/summaries.jsonl` when it finishes, then read the generated `README.md`
and `AGENTS.md`.

`trace-miner init` without a name prompts for one and defaults to `hello-world`.
Use `--force` to replace an existing scaffold.

## Language-Model Pipelines

Language-model stages use OpenRouter:

```bash
export OPENROUTER_API_KEY=...
```

The default LM provider/model are:

```yaml
lm:
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
```

After `init`, the same project is the batch and stream demo. Summarize maps over
each step, produces one-sentence step summaries, then reduces those into a
failure diagnosis and a compact trajectory summary.

### Batch Run

```bash
trace-miner summarize.yaml
```

When it finishes, the CLI prints each output path and a one-line run profile
(wall time, plus LM calls, tokens, and cost when any language-model stages ran).

Inspect `output/summaries.jsonl`, then try `pii.yaml`, `frustration.yaml`, or
`category.yaml` on the same traces.

### Streaming Run

`stream.yaml` watches a live JSONL input file and processes each complete line
after it is appended. In one terminal:

```bash
trace-miner stream stream.yaml --reset-checkpoint --progress
```

In another, replay fixture trajectories into the live input file:

```bash
trace-miner replay input/trajectories.jsonl --into input/live.jsonl --reset
```

`--reset` truncates the live file first. That command replays all three fixture
traces. `--interval` is the delay between appended records. Use `--limit 1` to
send a single trace.

The stream reads `input/live.jsonl` and appends summaries to
`output/stream_summaries.jsonl`.

Stream checkpoint and journal state lives under `.trace-miner/` beside the
config and is ignored by git. Use `--reset-checkpoint` to start reading from
byte 0 again. Stream outputs are append-only, so remove the output JSONL
between experiments when you want a fresh result file.

The streaming progress line reports:


| Counter     | Meaning                                                    |
| ----------- | ---------------------------------------------------------- |
| `read`      | Complete input JSONL records read from root input files.   |
| `produced`  | Records created by pipeline stages and reducers.           |
| `done`      | Records fully processed and safe to checkpoint past.       |
| `failed`    | Records or groups that failed during execution.            |
| `waiting`   | Groups/windows still waiting to reduce or flush.           |




## Config Shape

A pipeline config names a stage module and a set of file-based pipelines. Comments below mark
each part:

```yaml
from: ./stages                  # module where stage names are resolved
id_field: session_id            # optional; used in error messages and identity

lm:                             # optional LM defaults
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
  max_concurrent: 64            # concurrent LM calls inside this run
  cache: true                   # DSPy request cache; set false to always hit the provider

pipelines:
  - in: input/trajectories.jsonl   # JSONL source
    out: output/findings.jsonl     # JSONL sink
    pipeline:
      - filter: has_steps          # Python stage: keep or drop a record
      - group:                     # one barrier: wait for all children, then reduce
          expand: split_steps      # Python stage: one parent -> many children
          pipeline:
            - map: detect_failure      # LM stage: fires for every item in the group
          reduce: diagnose_trajectory  # LM stage: fires once the group closes
```

Stage names resolve to either a Python function or a DSPy module. The config does not change.

```python
def has_steps(record):
    return record if record.get("steps") else None


def split_steps(record, ctx):
    for step in record["steps"]:
        ctx.emit({"step_index": step["step_id"], "message": step["message"]})


class DetectFailure(dspy.Signature):
    """Flag a concrete failure mode in one agent step, or none."""
    # Inputs
    step_index: int = dspy.InputField()
    message: str = dspy.InputField()
    # Outputs
    failure_mode: str = dspy.OutputField(
        desc="One of: tool_loop, hallucinated_path, stuck, none."
    )
    evidence: str = dspy.OutputField()

detect_failure = dspy.Predict(DetectFailure)


class DiagnoseTrajectory(dspy.Signature):
    """Diagnose the dominant failure mode from per-step flags."""
    # Inputs; child fields arrive as lists
    session_id: str = dspy.InputField()
    failure_mode: list[str] = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    # Outputs
    failed: bool = dspy.OutputField()
    diagnosis: str = dspy.OutputField()

diagnose_trajectory = dspy.Predict(DiagnoseTrajectory)
```

Supported MVP step forms are `map`, `filter`, `stage`, `tee`, and `group`. Groups support `expand`,
`by`, and `every` in batch mode. Streaming mode supports `expand` and `every`; `by` is rejected until
there is a close condition for unbounded streams.

## More Detail

Design specs live in `[docs/](docs/)`:

- `[docs/pipeline-spec.md](docs/pipeline-spec.md)` describes the core runtime model.
- `[docs/lm-pipeline.md](docs/lm-pipeline.md)` describes the DSPy/OpenRouter language-model stage support.
- `[docs/streaming-amendment.md](docs/streaming-amendment.md)` describes the stream-friendly WAL/checkpoint design.