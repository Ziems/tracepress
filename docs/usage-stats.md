# Usage stats

Trace Miner sends anonymous usage stats by default. They never include prompts, completions, paths, or record contents.

Set `TRACE_MINER_TELEMETRY` to turn them off. Accepted values are `0`, `false`, `no`, `off`, and `disabled`.

`trace-miner stats` is separate. It reads local totals in `~/.trace-miner/` and does not send them.

## Install id

The install id is a random UUID stored at `~/.trace-miner/telemetry_id`. It is the analytics distinct id. It is not derived from a username, email, IP address, or path. `TRACE_MINER_HOME` changes the directory, the same way it changes the local stats directory.

## Delivery

After a pipeline command finishes, each event is posted by its own detached helper process. The timeout is 10 seconds. There is no retry. If the machine is offline, the event is dropped. A network failure does not change the command's result.

A batch run posts `pipeline_run` when it finishes. `serve` posts `pipeline_serve_start` when the process is ready, then a cumulative `pipeline_serving` snapshot every 5 minutes, because the process may never exit. `pipeline_served` is sent only when the process actually stops. `init` posts `pipeline_init`. `stats`, help, and unknown commands do not send an event.

Serve snapshots are cumulative. Keep the latest `snapshot` for a `run_id`, or use `pipeline_served` when the process has exited. Do not add the snapshots together.

Person profiles and GeoIP are disabled on every event.

The PostHog project key is a write-only key built into the package. It is not read from the environment.

## Events

Every event includes the app version, install source (`package`, `editable`, `source`, or `vcs`), Python version, OS, architecture, whether `CI` is set, and an optional `invoked_by` label. `invoked_by` is recorded only when a known agent variable is present (`CURSOR_AGENT`, `CLAUDECODE`, `CODEX_SANDBOX`, `GEMINI_CLI`). The variable's value is not read.

`command_flags` lists registered flag names only. Flag values are not sent.

### `pipeline_init`

Sent when `trace-miner init` finishes, including errors and interrupts.

- `schema`: `pipeline_init.v1`
- `command_flags`
- `status`: `completed`, `errored`, or `interrupted`
- `duration_seconds`
- `exit_code`
- `exception_type`: exception class name, when the invocation did not complete

### `pipeline_run` and `pipeline_served`

Sent when a batch run or a serve process finishes, including handled errors and interrupts. A run that fails before the runtime starts, such as a bad config path, still sends `pipeline_run` or `pipeline_served` with zero counts and no `run_id`.

- `schema`: `pipeline_run.v1` or `pipeline_served.v1`
- `run_id`, when a pipeline runtime started
- `command_flags`
- `status`: `completed`, `errored`, or `interrupted`
- `duration_seconds`: pipeline time when the runtime started, otherwise the whole command
- `exit_code`
- `exception_type`
- `pipeline_count`
- `stage_count`: every stage, of any kind
- `maps`, `filters`, `stages`, `alerts`, `expands`, `reduces`: how many stages of that kind the pipeline contains. `stages` counts steps whose kind is `stage`. A `tee` step is a file write and is not counted
- `depth`: deepest group nesting. `0` when the pipeline has no group
- `width`: most steps in any one step list, including nested groups. `tee` and `group` count as steps
- `lm_provider` and `lm_model`, when a runtime started
- `records_read`, `emitted`, `failed`
- `lm_calls`, `lm_prompt_tokens`, `lm_completion_tokens`, `lm_cost_usd`

Language-model fields are counts and totals for the whole run. They are not the text sent to or returned by the model.

### `pipeline_serve_start` and `pipeline_serving`

`pipeline_serve_start` is sent when `trace-miner serve` is ready. `pipeline_serving` follows every 5 minutes after that, including while the process is idle. `pipeline_served` is the separate event sent when the process stops.

The fields match `pipeline_served`, with these differences:

- `schema`: `pipeline_serve_start.v1` or `pipeline_serving.v1`
- `status`: `running`
- `snapshot`: `1` on `pipeline_serve_start`, then `2`, `3`, and so on for `pipeline_serving`
- `duration_seconds`: time since the process started
- `exit_code` and `exception_type` are omitted

Counts on `pipeline_serving` are totals since the process started, not the change since the previous snapshot.

## Not collected

- raw command arguments and flag values
- local paths, config filenames, stage names, and checkpoint paths
- host and port
- LM inputs and outputs: prompts, completions, signature text, stage messages, trace records, and cache entries
- exception messages and tracebacks
- environment variable names or values
- usernames, emails, and organization names
- IP address or GeoIP-derived location
