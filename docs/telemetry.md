# Telemetry

TracePress sends one anonymous usage event per run to PostHog. It helps us see which models,
operators and pipelines people use, and what runs cost. The first time it sends, it prints a
one-time notice.

## What is sent

- TracePress, DSPy and Python versions, and the OS name.
- Whether the run succeeded. On failure, the exception class name is sent, but only for
  built-in, TracePress, DSPy or litellm exceptions. Anything else is sent as `"other"`.
- Duration, and counts of LM calls, cache hits, errors, retries, tokens and estimated cost.
- For each model, the gateway and the model name, e.g. `openrouter` + `openai/gpt-6-luna`.
  Portkey provider slugs, endpoints and fine-tune ids are stripped.
- How many of each operator the plan has (`lm_map`, `filter`, …), the kinds of sources
  (`harbor`, `atif`, …), and which built-in pipelines ran.
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
