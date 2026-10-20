---
name: tracepress-pipelines
description: Build and run TracePress pipelines that use LLM operators (lm_map, lm_filter, lm_agg, lm_topk, lm_dedup) on pandas DataFrames to summarize, label, classify, rank, or dedupe rows. Use when the user wants to analyze agent traces, trajectories, logs, transcripts, reviews, or any text data with an LLM at row scale, find failure modes, PII, frustration, or misbehavior, or load and classify Harbor / Terminal-Bench jobs. Every operator output is saved to a workspace for later exploration (see the tracepress-explore skill).
---

# Building TracePress pipelines

TracePress adds LLM operators to pandas DataFrames. A **pipeline** is a Python function of those operators registered on a **workspace**; every operator output inside it is saved to disk as an artifact with its instruction, model, parents, and cost. You write the pipeline as a script, run it, and then explore the artifacts (the `tracepress-explore` skill).

## Setup

```bash
uv add tracepress   # or: pip install tracepress
```

Models are LiteLLM ids, optionally behind a gateway: `openai/gpt-5-nano`, `anthropic/claude-haiku-4-5`, `openrouter:openai/gpt-6-luna`, `portkey:openai/gpt-5-nano`. The matching key must be in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `PORTKEY_API_KEY`). Check which keys exist before picking a model, including in a `.env` file (load it with `set -a; . ./.env; set +a` before running, or `python-dotenv` in the script); if none does, ask the user. Never print key values. Operators make one call per row, so use a cheap, fast model unless the user names one.

## Workflow

1. **Look at the data first** with plain pandas: columns, row count, typical text length. Decide what one row is (a step, a run, a review).
2. **Write the pipeline to a file** in the user's project, e.g. `tracepress_pipelines.py`. Pipelines exist only in the Python process that defines them, so the file is what lets you (or the explore skill) re-run them later. Keep loading, pipeline definitions, and the run in that file, with the run under `if __name__ == "__main__":`.
3. **Try it on a few rows** (`df.head(5)`) and read the outputs. Fix the signature or instruction before spending on the full data.
4. **Run it on everything under a budget**, then report the run id, the artifact names, row counts, and `lm.usage` (calls and cost).

```python
# tracepress_pipelines.py
import pandas as pd
import tracepress as tp

lm = tp.Model("openai/gpt-5-nano", temperature=0, max_tokens=800, progress=False)
tp.configure(lm=lm)
ws = tp.Workspace("tracepress_ws")


@ws.pipeline(description="Summarizes each step of a failed run, then explains why the run failed.")
def failure_reasons(steps):
    summarized = steps.lm_map(
        "task_name, step_id, source, step -> step_summary",
        "Summarize what happened in this step in one sentence.",
        save_as="step_summaries",
    )
    return summarized.lm_agg(
        "task_name, failed_tests, step_summary -> failure_reason",
        "Explain in one sentence why the run failed.",
        by="session_id",
        save_as="failure_reasons",
    )


if __name__ == "__main__":
    import sys

    steps = pd.read_json("data/steps.jsonl", lines=True)
    if "--sample" in sys.argv:
        steps = steps[steps.session_id.isin(steps.session_id.unique()[:2])]
    with tp.budget(max_calls=2000, max_usd=5):
        run = ws.run("failure_reasons", steps)
    print(run.id, [a["name"] for a in run.manifest["artifacts"]])
    print(lm.usage)
```

`progress=False` turns off the tqdm bar, which floods non-interactive output; leave it on in notebooks. Run it with `uv run python tracepress_pipelines.py --sample` first, then without the flag. Responses are cached under `~/.tracepress/cache`, so rerunning an unchanged step is free.

Name a workspace once and reuse it; it is just a folder (`<ws>/runs/<run_id>/manifest.json` plus one parquet per artifact). Use `ws.save("name", df)` to store raw input as an artifact without running anything.

## Operators

Each takes a **signature** (which columns go in and, for map/agg, which come out) and a plain-prose **instruction**. Each returns a DataFrame, so they chain in any order.

| Call | What it does | Output |
|---|---|---|
| `df.lm_map("a, b -> c: bool, d", instr, examples=)` | one call per row | input rows plus output columns |
| `df.lm_filter("a, b", condition, explain=)` | keeps rows where the condition holds | subset, index preserved (`explanation` column if `explain=True`) |
| `df.lm_agg("a -> summary", instr, by="col")` | hierarchical reduce, any number of rows | one row per group, plus `n_rows` |
| `df.lm_topk("a, b", criterion, k=3)` | listwise ranking | `k` rows with a `rank` column |
| `df.lm_dedup("a, b", criterion)` | LLM duplicate clustering | one row per cluster, `n_duplicates`, `duplicate_ids` |

Signature rules:
- Outputs are `str` unless typed: `bool`, `int`, `float`, `list[str]`, `Literal['a', 'b']`. Use `Literal` for labels so the values are a closed set you can count.
- `lm_filter`, `lm_topk`, `lm_dedup` take inputs only: `"product, review"`.
- An input and an output cannot share a name (`"v -> v"` fails).
- Instructions are prose; there is no `{column}` templating.
- When fields need descriptions, use the class form; its docstring is the instruction:

```python
class SummarizeStep(tp.Signature):
    """What happened in this step of a failed agent run?"""
    step: str = tp.In(desc="the step's message, tool calls, and observation")
    step_summary: str = tp.Out(desc="one sentence")

steps.lm_map(SummarizeStep, save_as="step_summaries")
```

Inside a pipeline, always pass `save_as=` with a descriptive name; otherwise artifacts are named `step{n}_{op}`. Pass `lm=` to any operator to use a different model for that step, or wrap steps in `with tp.using(lm=...)`.

Give each pipeline a `description=` that says what it produces and when to use it: it is what someone exploring the workspace later reads.

## Cost discipline

- Estimate first: `lm_map`/`lm_filter` cost one call per row; `lm_agg`, `lm_topk`, `lm_dedup` cost fewer. Tell the user the row count and rough cost before a large run.
- `tp.budget(max_calls=, max_usd=)` refuses a batch before it starts if it would go over. Dollar caps estimate up front only when the model has `price=(input, output)` in $/M tokens; otherwise they are checked against actual spend.
- Filter or `lm_filter` down before an expensive `lm_map`. Truncate very long text columns in pandas before they reach a prompt.
- A run that fails keeps the artifacts it finished; fix and rerun rather than starting over.

## Harbor and Terminal-Bench jobs

When the input is a Harbor job directory, a trial directory, or an ATIF `trajectory.json`, load it instead of parsing files yourself:

```python
loaded = tp.harbor_load("jobs/my-job")   # or a list of paths
loaded.trials   # one row per trial: task_name, agent_name, model_name, reward, exception, failed_tests, ...
loaded.steps    # one row per step: trial columns plus step_id, source, message, reasoning, tool_calls, observation, is_subagent
```

`harbor_load` reads files, so call it outside the pipeline and pass the DataFrame in. It does not read JSONL: if you have one ATIF object per line, write each to its own `<id>/trajectory.json` and pass the list of directories. Bare trajectory files carry no `result.json`, so `task_name`, `reward`, and `failed_tests` come out empty; join them in from wherever the data keeps them. Two ready-made classifiers take the loaded result (or `loaded.steps`):

```python
@ws.pipeline(description="Labels each trial with the nine Terminal Agent Taxonomy modes.")
def agent_taxonomy(steps):
    return tp.harbor_agent_taxonomy(steps)       # saves agent_step_summaries, agent_taxonomy_labels

@ws.pipeline(description="Labels each failed shell/REPL/CLI command with a failure subcategory.")
def command_taxonomy(steps):
    return tp.harbor_command_taxonomy(steps)     # saves command_failures, command_taxonomy
```

Filter `loaded.steps` to failed trials (`reward` below 1, or a non-empty `exception`) first if the user only cares about failures. You can also chain your own operators on these outputs.

Full argument lists: `help(tp.Model)`, `help(pd.DataFrame.lm_agg)`, and the project's `REFERENCE.md`.
