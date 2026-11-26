# Telemetry

TracePress sends anonymous usage events to PostHog. They help us see which models, operators,
pipelines and features people use, and what runs cost. The first time it sends, it prints a
one-time notice.

## Events

- `run_completed`: one per run. A run that only reads a source (no operators, no LM calls),
  such as counting a table's rows, sends nothing unless it fails.
- `optimize_completed`: one per `table.optimize(...)`. The runs it makes while trying prompts
  don't send their own events.
- `server_started`: when `tracepress serve` (or `serve_in_thread`) starts.
- `pipeline_registered`: each time a client registers a pipeline on a server. A registered
  pipeline also sends its `run_completed` when it stops.

## What is sent

- TracePress, DSPy and Python versions, and the OS name.
- Whether the run succeeded. On failure, the exception class name is sent, but only for
  built-in, TracePress, DSPy or litellm exceptions. Anything else is sent as `"other"`.
- Duration, and counts of LM calls, cache hits, errors, retries, tokens and estimated cost.
- For each model, the gateway and the model name, e.g. `openrouter` + `openai/gpt-6-luna`.
  Portkey provider slugs, endpoints and fine-tune ids are stripped.
- How many of each operator the plan has (`lm_map`, `filter`, `edit` for a `rerun`, …), the
  kinds of sources (`harbor`, `atif`, `server`, …), which built-in pipelines ran, and whether
  the run recorded lineage.
- For `optimize`: the optimizer, which kind of LM step was tuned, whether the metric is a
  column or a function, how many labels and held-out labels, the budget, how many prompts it
  tried and kept, LM calls, cost, duration and the held-out score before and after. Never the
  prompts.
- For the server: whether it listens only on this machine, whether it uses a token, keeps a
  limited history or has a default model, and each registered pipeline's operator counts.
  Never the address or the pipeline's code.
- A random install id, kept in `~/.config/tracepress/telemetry.json`. It isn't derived from
  your machine; delete the file to reset it.

Person profiles and GeoIP lookups are turned off for every event.

## What is never sent

Your data, prompts, signatures, instructions, column or task names, file paths, hostnames,
API keys and error messages.

## Turning it off

Any one of these turns it off:
- `TRACEPRESS_TELEMETRY=0`
- `DO_NOT_TRACK=1`
- `tp.configure(telemetry=False)`

It's also off in CI and under pytest.

To see exactly what would be sent, without sending anything, set
`TRACEPRESS_TELEMETRY=debug`.
