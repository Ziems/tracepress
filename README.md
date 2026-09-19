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

`init` writes a hello-world project with shared LM stages and four pipelines:

```bash
export OPENROUTER_API_KEY=...
trace-miner init my-traces
cd my-traces
trace-miner summarize.yaml
```

That scaffolds `stages.py`, three example harbor traces, `AGENTS.md`, and `summarize.yaml` /
`pii.yaml` / `frustration.yaml` / `category.yaml`. The summarize pipeline summarizes each step and diagnoses the trajectory, same shape as `examples/harbor`. The traces include a fake SSN, a fake phone number, and a
frustrated user turn so the other pipelines have something to find. Open
`output/summaries.jsonl` when it finishes, then read the generated `README.md`
and `AGENTS.md`.

`trace-miner init` without a name prompts for one and defaults to `hello-world`.
Use `--force` to replace an existing scaffold.

## Language-Model Examples

Language-model examples use OpenRouter:

```bash
export OPENROUTER_API_KEY=...
```

The default LM provider/model are:

```yaml
lm:
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
```

Each directory under `examples/` is a self-contained demo (`pipeline.yaml`, `stages.py`, `input/`).


| Demo                                                     | What it shows                         | Run                                              |
| -------------------------------------------------------- | ------------------------------------- | ------------------------------------------------ |
| `[examples/harbor](examples/harbor/README.md)`           | Trajectory analysis, batch or stream  | `trace-miner examples/harbor/pipeline.yaml`      |
| `[examples/pii](examples/pii/README.md)`                 | PII detection over slim harbor traces | `trace-miner examples/pii/pipeline.yaml`         |
| `[examples/category](examples/category/README.md)`       | Domain/outcome labeling               | `trace-miner examples/category/pipeline.yaml`    |
| `[examples/frustration](examples/frustration/README.md)` | User-frustration detection            | `trace-miner examples/frustration/pipeline.yaml` |




## Run The Harbor LM Example

The harbor example summarizes a trajectory by mapping over each step, producing one-sentence step
summaries, then reducing those into a failure diagnosis and a compact trajectory summary. See
`[examples/harbor/README.md](examples/harbor/README.md)` for the full walkthrough.

### Batch Run

This reads a fixed fixture file and processes one selected trajectory:

```bash
trace-miner examples/harbor/pipeline.yaml
```

When it finishes, the CLI prints each output path and a one-line run profile (wall time, plus LM calls, tokens, and cost when any language-model stages ran).

Inspect:

- `examples/harbor/output/one_summary.jsonl`



### Streaming Run

The streaming example watches a live JSONL input file and processes each complete line after it is
appended.

First prepare an empty live input file:

```bash
uv run python examples/harbor/simulate_stream.py --reset-only
```

In one terminal, start the stream:

```bash
trace-miner stream examples/harbor/stream.yaml --reset-checkpoint --progress
```

In another terminal, replay fixture trajectories into the live input file:

```bash
uv run python examples/harbor/simulate_stream.py --limit 1 --interval 3
```

`--interval` is the simulator delay between appended records. Increase `--limit` to replay more
trajectories; use `--limit 0` to replay the whole fixture.

The stream reads `examples/harbor/input/live.jsonl` and appends summaries to:

- `examples/harbor/output/stream_summaries.jsonl`

Stream checkpoint and journal state lives under `.trace-miner/` beside the config and is ignored by
git. Use `--reset-checkpoint` to start reading from byte 0 again. Stream outputs are append-only, so
remove the output JSONL between experiments when you want a fresh result file.

The streaming progress line reports:


| Counter       | Meaning                                                    |
| ------------- | ---------------------------------------------------------- |
| `records`     | Complete input JSONL records read from root input files.   |
| `emitted`     | Records emitted by pipeline stages and reducers.           |
| `retired`     | Input records fully processed and safe to checkpoint past. |
| `failed`      | Records or groups that failed during execution.            |
| `groups_open` | Groups/windows still waiting to reduce or flush.           |




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

Example-specific notes live in each demo's `README.md`.