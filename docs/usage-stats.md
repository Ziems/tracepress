# Usage stats

TracePress sends anonymous usage stats so we can see which operators and agent runs people actually use. They are on by default. A network failure does not change the result of your call.

Stats never include prompts, completions, instructions, signatures, questions, answers, row contents, artifact or pipeline names, workspace paths, budget or flag values, exception messages, or anything that identifies you.

## Opt out

Set `TRACEPRESS_TELEMETRY` to any of these (case does not matter): `0`, `false`, `no`, `off`, `disabled`.

```bash
export TRACEPRESS_TELEMETRY=off
```

Any other value, including leaving it unset, leaves stats on. The check happens when an event would be sent. An opted-out call does not build an event and does not send one.

## Install id

Each install gets a random id, stored at `~/.tracepress/telemetry_id` (or `$TRACEPRESS_HOME/telemetry_id` if that variable is set). It is not derived from your name, email, username, or a path. The same id is reused so events from one install can be grouped. PostHog person profiles and GeoIP are disabled.

## Events

`lm_op` — one event each time you call `df.lm_map`, `lm_filter`, `lm_agg`, `lm_topk`, `lm_dedup`, or `lm_join`, including when the call fails or is interrupted.

`agent_run` — one event each time you call `ws.agent_run`, including when it fails or is interrupted.

Both events include the TracePress version, how the package was installed, the Python version, operating system, architecture, whether the process looks like CI, and, when a known coding agent started the process, which one. They also include the model provider and model name, token and cost totals, and how long the call took. On failure, the exception type is included (the class name only).

`lm_op` adds the operator name, input and output row counts, and whether the call was inside a workspace or an agent run. Inside a workspace it adds whether that run is a pipeline, an agent run, or a save, plus how many artifacts and pipelines the workspace has. It does not add the pipeline's name.

`agent_run` adds how many turns the agent took, how many times it called each of its tools, how many operator calls those tools caused, whether it hit the step limit, and the workspace's artifact and pipeline counts.

## What is not collected

- instructions, signatures, questions, answers, prompts, and completions
- cell values and any other record contents
- artifact names, pipeline names, and workspace paths
- budget amounts and other parameter values
- exception messages and tracebacks
- environment variable values, usernames, and emails
