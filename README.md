# Mine Trace

Mine Trace is a config-driven JSONL pipeline runtime for map/filter/reduce workflows where groups
can have variable size. Pipeline topology lives in YAML; stage logic lives in ordinary Python
functions.

It is built for runs like:

- filter and transform JSONL records
- expand one record into many child records
- reduce child records back into one summary
- run the same pipeline in batch mode or against an appended JSONL stream
- call DSPy language-model stages when a pipeline needs semantic labeling or summarization

## Setup

Use `uv` from the repository root:

```bash
uv sync
uv run mine-trace --help
```

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

## Run A Basic Batch Pipeline

```bash
uv run mine-trace examples/basic/pipeline.yaml --manifest examples/basic/output/manifest.json
```

Inspect:

- `examples/basic/output/overviews.jsonl`
- `examples/basic/output/kept.jsonl`
- `examples/basic/output/manifest.json`

## Run The Harbor LM Example

The harbor example summarizes a trajectory by mapping over each step, producing one-sentence step
summaries, then reducing those into a failure diagnosis and a compact trajectory summary.

### Batch Run

This reads a fixed fixture file and processes one selected trajectory:

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace examples/lm/harbor_pipeline.yaml --manifest examples/lm/output/harbor_manifest.json
```

Inspect:

- `examples/lm/output/harbor_one_summary.jsonl`
- `examples/lm/output/harbor_manifest.json`

### Streaming Run

The streaming example watches a live JSONL input file and processes each complete line after it is
appended.

First prepare an empty live input file:

```bash
uv run python examples/lm/simulate_harbor_stream.py --reset-only
```

In one terminal, start the stream:

```bash
export OPENROUTER_API_KEY=...
uv run mine-trace stream examples/lm/harbor_stream_pipeline.yaml --reset-checkpoint --progress
```

In another terminal, replay fixture trajectories into the live input file:

```bash
uv run python examples/lm/simulate_harbor_stream.py --limit 1 --interval 3
```

`--interval` is the simulator delay between appended records. Increase `--limit` to replay more
trajectories; use `--limit 0` to replay the whole fixture.

The stream reads `examples/lm/input/harbor-live.jsonl` and appends summaries to:

- `examples/lm/output/harbor_stream_summaries.jsonl`

Stream checkpoint and journal state lives under `.mine-trace/` beside the config and is ignored by
git. Use `--reset-checkpoint` to start reading from byte 0 again. Stream outputs are append-only, so
remove the output JSONL between experiments when you want a fresh result file.

The streaming progress line reports:

| Counter | Meaning |
| --- | --- |
| `records` | Complete input JSONL records read from root input files. |
| `emitted` | Records emitted by pipeline stages and reducers. |
| `retired` | Input records fully processed and safe to checkpoint past. |
| `failed` | Records or groups that failed during execution. |
| `groups_open` | Groups/windows still waiting to reduce or flush. |

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

- `pipeline-spec.md` describes the core runtime model.
- `lm-pipeline.md` describes the DSPy/OpenRouter language-model stage support.
- `streaming-amendment.md` describes the stream-friendly WAL/checkpoint design.
- `examples/basic/README.md` and `examples/lm/README.md` contain example-specific notes.
