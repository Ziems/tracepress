# Usage stats

Trace Miner sends anonymous usage stats by default. They never include prompts, completions, paths, or record contents.

Set `TRACE_MINER_TELEMETRY` to turn them off. Accepted values are `0`, `false`, `no`, `off`, and `disabled`.

`trace-miner stats` is separate. It reads local totals in `~/.trace-miner/` and does not send them.

## Install id

The install id is a random UUID stored at `~/.trace-miner/telemetry_id`. It is the analytics distinct id. It is not derived from a username, email, IP address, or path. `TRACE_MINER_HOME` changes the directory, the same way it changes the local stats directory.

## Delivery

After a pipeline command finishes, each event is posted by its own detached helper process. The timeout is 10 seconds. There is no retry. If the machine is offline, the event is dropped. A network failure does not change the command's result.

A batch run posts one `pipeline_stage` event per configured stage, then `pipeline_run`. `serve` does the same with `pipeline_served` when the process stops. `init` posts `pipeline_init`. `stats`, help, and unknown commands do not send an event.

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
- `stage_count`
- `lm_provider` and `lm_model`, when a runtime started
- `records_read`, `emitted`, `failed`
- `lm_calls`, `lm_prompt_tokens`, `lm_completion_tokens`, `lm_cost_usd`

Language-model fields are counts and totals for the whole run. They are not the text sent to or returned by the model.

### `pipeline_stage`

Sent once per configured stage when a run or serve finishes. Totals cover every record that stage handled. A stage that never ran, because an earlier stage failed or no records reached it, is still sent with `calls: 0`.

`tee` steps are file writes and do not produce a stage event.

- `schema`: `pipeline_stage.v1`
- `run_id`
- `index`: position in config order, starting at 0
- `kind`: `map`, `filter`, `stage`, `alert`, `expand`, or `reduce`
- `implementation`: `python` or `lm`
- `async`: true when a Python stage is a coroutine
- `duration_seconds`: wall time spent in that stage
- `calls`: times the stage function ran
- `records_in`: records passed in. For `reduce`, this is the number of children
- `records_out`: records the stage produced. For `alert`, this is the number of alerts fired
- `failed`
- `status`: `completed`, `errored`, or `interrupted`
- `lm_calls`, `lm_prompt_tokens`, `lm_completion_tokens`, `lm_cost_usd`
- `lm_provider` and `lm_model`, for LM stages

## Not collected

- raw command arguments and flag values
- local paths, config filenames, stage names, and checkpoint paths
- host and port
- LM inputs and outputs: prompts, completions, signature text, stage messages, trace records, and cache entries
- exception messages and tracebacks
- environment variable names or values
- usernames, emails, and organization names
- IP address or GeoIP-derived location
