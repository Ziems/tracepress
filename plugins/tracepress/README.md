# TracePress plugin for Claude Code

Skills that teach Claude to build and run [TracePress](https://github.com/Ziems/tracepress)
pipelines: LLM-powered map/reduce over agent traces.

- **tracepress-analyze-traces:** failure modes, summaries and comparisons over finished runs.
- **tracepress-stream-traces:** live monitoring with running notes per trial.
- **tracepress-add-adapter:** support for a new trace format.

The plugin itself is only these skills: no hooks, servers or scripts. When a task needs it, Claude
uses the `tracepress` Python package from PyPI, installed into your project as described in the
[TracePress README](https://github.com/Ziems/tracepress#install).

## Telemetry

The `tracepress` package sends anonymous usage events to PostHog: one per finished run or
`optimize`, and one when a server starts or registers a pipeline, with versions, durations, LM
call and token counts, cost, model names and how many of each operator ran. It never sends your
data, prompts, signatures, column or task names, file paths, hostnames, API keys or error
messages. It prints a notice the first time it sends.

Turn it off with `TRACEPRESS_TELEMETRY=0`, `DO_NOT_TRACK=1` or `tp.configure(telemetry=False)`.
It's already off in CI and under pytest. `TRACEPRESS_TELEMETRY=debug` prints what would be sent
without sending it. [docs/telemetry.md](https://github.com/Ziems/tracepress/blob/main/docs/telemetry.md)
lists every field.

## License

Apache-2.0
