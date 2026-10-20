# TracePress reference

What each user-facing piece is for. For how to use them together, see the [README](README.md) and [examples/quickstart.ipynb](examples/quickstart.ipynb). Full argument lists are in the docstrings: run `help(tp.Model)` and so on.

```python
import tracepress as tp
```

## Models and settings

| | Purpose |
|---|---|
| `tp.Model(model, ...)` | A language model to run operators with. Wraps a `dspy.LM` on the lm15 engine. It adds concurrent batches, a progress bar, token and cost totals (`model.usage`) and budget checks. A direct `model` is a LiteLLM id, like `"openai/gpt-5-nano"`. A gateway prefixes that id: `"openrouter:openai/gpt-6-luna"`, `"portkey:openai/gpt-5-nano"`. |
| `tp.configure(lm=)` | Sets the default model for operators. Set this once at the top of a notebook or script. |
| `tp.using(lm=)` | Overrides the default model inside a `with` block, for example to use a bigger model for one step. |
| `tp.budget(max_calls=, max_usd=)` | Caps the model calls made inside a `with` block. A batch that would go over the cap is refused before it starts, so it costs nothing. |
| `tp.BudgetExceeded` | The error raised when a budget refuses a batch. |
| `tp.Usage` | Totals of calls, cached calls, input and output tokens, and cost in dollars. Found on `model.usage` and in artifact manifests. |

## Operators

Every operator is a DataFrame method. Each takes a signature and a plain-language instruction, and returns a new DataFrame, so operators chain in any order. Inside a pipeline, `save_as="name"` saves the result as an artifact with that name.

| | Purpose |
|---|---|
| `df.lm_map(signature, instruction)` | Transforms or extracts, one call per row. The signature's outputs become new columns. Typed outputs (`bool`, `Literal[...]`, `list[str]`) make it an extractor or labeler. `examples=` adds few-shot rows. |
| `df.lm_filter(inputs, condition)` | Keeps the rows where a condition holds. `explain=True` also keeps a one-sentence reason per row. |
| `df.lm_agg(signature, instruction, by=)` | Summarizes many rows into one answer, or one per `by` group. Any number of rows works, because rows that don't fit in one call are reduced in stages. The result is one row per group with an `n_rows` column. |
| `df.lm_topk(inputs, criterion, k)` | Picks the `k` rows that best match a criterion, adding a `rank` column. |
| `df.lm_dedup(inputs, criterion)` | Merges rows that mean the same thing, keeping one row per cluster. `n_duplicates` and `duplicate_ids` record what was merged. |
| `df.lm_join(...)` | Not built yet (TODO). |

## Signatures

A signature names the columns a call reads and, for `lm_map` and `lm_agg`, the columns it writes. Signatures are DSPy signatures.

| | Purpose |
|---|---|
| `"a, b -> c: bool, d"` | The string form. Inputs go left of `->` and outputs to the right. Outputs are `str` unless typed. `lm_filter`, `lm_topk` and `lm_dedup` take only inputs: `"a, b"`. An input and an output can't share a name. |
| `tp.Signature` | The class form, for when fields need descriptions. Subclass it; the docstring is the instruction. It's `dspy.Signature`. |
| `tp.In(desc=)` / `tp.Out(desc=)` | Mark a class-signature field as an input column or an output column. They're `dspy.InputField` and `dspy.OutputField`. |

## Workspace and pipelines

| | Purpose |
|---|---|
| `tp.Workspace(path)` | A folder that keeps every pipeline run and its artifacts, plus the pipelines registered in this session. Point it at a Drive folder on Colab to keep results. |
| `@ws.pipeline(description=)` | Registers a function as a pipeline. Calling it then runs it in the workspace and saves every operator output. The description says what the pipeline is for, to whoever explores the workspace later. |
| `ws.run(name, *args)` | Runs a registered pipeline and returns its `Run`. Calling the pipeline directly does the same and returns just its output. |
| `ws.save(name, df)` | Adds a DataFrame, such as raw data, to the workspace as an artifact. |
| `ws[name]` / `ws.load(name)` | Loads an artifact: the latest with that name, or a specific one as `"run_id/name"`. |
| `ws.artifacts(latest=)` | A table of every artifact: which op and instruction made it, its parents, and its shape. |
| `ws.runs()` | A table of every run, with its status and output. |
| `ws.lineage(name)` | Traces an artifact back through the steps that produced it. |
| `tp.Pipeline` | What `@ws.pipeline` returns. Call it to run it. |
| `tp.Run` | One execution of a pipeline. `with tp.Run(ws, "name"):` also groups ad-hoc operator calls into a run so their outputs are saved. `run.manifest` holds its artifacts and their details, and `run.output` holds what the function returned. |

## Skills

[skills/](skills/) holds skills for coding agents. Install them with `npx skills add Ziems/tracepress`.

| Skill | Purpose |
|---|---|
| `tracepress-pipelines` | Writes a pipeline file in your project, tries it on a few rows, and runs it under a budget. Uses the Harbor helpers when the input is a Harbor job. |
| `tracepress-explore` | Answers questions from a workspace's artifacts with pandas, runs extra operators on subsets only when needed, and cites artifact names and row ids. |

## Harbor

Load a Harbor job directory, a trial directory (`agent/trajectory.json` or `trajectory.json`), an ATIF file, or a list of those. No Harbor install is required. `trial_key` identifies each trial; a list of paths is prefixed so two jobs do not collide.

| | Purpose |
|---|---|
| `tp.harbor_load(path)` | Returns a `HarborLoad` with `.trials` (one row per trial, including reward, exception, and failed tests from `result.json` and `verifier/ctrf.json`) and `.steps` (one row per ATIF step, including embedded subagents). `path` is a job directory, a trial directory, an ATIF file, or a list of those. Not a workspace pipeline. |
| `tp.harbor_agent_taxonomy(loaded, lm=)` | Takes a `HarborLoad` or a steps DataFrame, not a path. Summarizes each user and agent step, then labels the trial with the Terminal Agent Taxonomy: `modes` (any of the nine) and `classes` (`execution`, `coherence`, `verification`). System prompts are left out. Saved as `agent_step_summaries` and `agent_taxonomy_labels` inside a pipeline. Wrap with `@ws.pipeline` to save its artifacts. |
| `tp.harbor_command_taxonomy(loaded, lm=)` | Takes a `HarborLoad` or a steps DataFrame, not a path. Keeps agent steps whose shell, REPL, or CLI command failed, then labels each with one subcategory and a `failure_class` (`invocation`, `repl`, `runtime`, `filesystem`, `other`). Saved as `command_failures` and `command_taxonomy`. Wrap with `@ws.pipeline` to save its artifacts. |

## On disk

| Path | Contents |
|---|---|
| `<workspace>/runs/<run_id>/manifest.json` | The run's status and parameters, plus each artifact's op, instruction, signature, output schema, model, parents, shape and usage. |
| `<workspace>/runs/<run_id>/<name>.parquet` | One artifact. |
| `~/.tracepress/cache` | The response cache, so rerunning a cell doesn't pay twice. |
