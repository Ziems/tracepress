# Trace Miner

Trace Miner is a config-driven JSONL pipeline runtime for map/filter/reduce workflows where groups
can have variable size. Pipeline topology lives in YAML; stage logic lives in ordinary Python
functions.

It is built for runs like:

- filter and transform JSONL records
- expand one record into many child records
- reduce child records back into one summary
- run the same pipeline in batch mode or against an appended JSONL stream
- call DSPy language-model stages when a pipeline needs semantic labeling or summarization

## Setup

From the repository root:

```bash
uv sync
uv run trace-miner --help
```

To install the CLI onto your PATH without publishing to PyPI:

```bash
uv tool install git+https://github.com/Ziems/trace-miner.git
trace-miner --help
```

The GitHub repo is private, so that command needs credentials that can clone it.
From a local checkout, `uv tool install .` works without GitHub access. After a
later PyPI publish, the public form is `uv tool install trace-miner`.

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

## Examples

Each directory under `examples/` is a self-contained demo (`pipeline.yaml`, `stages.py`, `input/`).

| Demo | What it shows | Run |
| --- | --- | --- |
| [`examples/lm`](examples/lm/README.md) | DSPy LM stage on mine-sensor readings | `uv run trace-miner examples/lm/pipeline.yaml` |
| [`examples/harbor`](examples/harbor/README.md) | Trajectory step summaries, batch or stream | `uv run trace-miner examples/harbor/pipeline.yaml` |
| [`examples/pii`](examples/pii/README.md) | PII detection over slim harbor traces | `uv run trace-miner examples/pii/pipeline.yaml` |
| [`examples/category`](examples/category/README.md) | Domain/outcome labeling | `uv run trace-miner examples/category/pipeline.yaml` |
| [`examples/frustration`](examples/frustration/README.md) | User-frustration detection | `uv run trace-miner examples/frustration/pipeline.yaml` |

## Run The Harbor LM Example

The harbor example summarizes a trajectory by mapping over each step, producing one-sentence step
summaries, then reducing those into a failure diagnosis and a compact trajectory summary. See
[`examples/harbor/README.md`](examples/harbor/README.md) for the full walkthrough.

### Batch Run

This reads a fixed fixture file and processes one selected trajectory:

```bash
export OPENROUTER_API_KEY=...
uv run trace-miner examples/harbor/pipeline.yaml
```

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
export OPENROUTER_API_KEY=...
uv run trace-miner stream examples/harbor/stream.yaml --reset-checkpoint --progress
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

A pipeline config names a stage module and a set of file-based pipelines:

```yaml
from: ./stages
id_field: id

pipelines:
  - in: input/traces.jsonl
    out: output/overviews.jsonl
    pipeline:
      - filter: enabled
      - tee: output/kept.jsonl
      - group:
          expand: split_samples
          pipeline:
            - filter: interesting
            - map: score
          reduce: summarize_trace
```

Stage functions are plain Python:

```python
def enabled(record):
    return record if record["enabled"] else None

def score(record):
    return {**record, "score": record["reading"] * 2}
```

Supported MVP step forms are `map`, `filter`, `stage`, `tee`, and `group`. Groups support `expand`,
`by`, and `every` in batch mode. Streaming mode supports `expand` and `every`; `by` is rejected until
there is a close condition for unbounded streams.

## More Detail

Design specs live in [`docs/`](docs/):

- [`docs/pipeline-spec.md`](docs/pipeline-spec.md) describes the core runtime model.
- [`docs/lm-pipeline.md`](docs/lm-pipeline.md) describes the DSPy/OpenRouter language-model stage support.
- [`docs/streaming-amendment.md`](docs/streaming-amendment.md) describes the stream-friendly WAL/checkpoint design.

Example-specific notes live in each demo's `README.md`.
