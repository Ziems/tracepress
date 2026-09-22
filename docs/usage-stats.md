# Usage stats

Trace Miner sends anonymous usage stats by default. They never include prompts, completions, paths, or record contents.

Set `TRACE_MINER_TELEMETRY` to turn them off. Accepted values are `0`, `false`, `no`, `off`, and `disabled`.

`trace-miner stats` is separate. It reads local totals in `~/.trace-miner/` and does not send them.

## Install id

The install id is a random UUID stored at `~/.trace-miner/telemetry_id`. It is the analytics distinct id. It is not derived from a username, email, IP address, or path. `TRACE_MINER_HOME` changes the directory, the same way it changes the local stats directory.

## Delivery

After a command finishes, a detached helper process posts one event. The timeout is 10 seconds. There is no retry. If the machine is offline, the event is dropped. A network failure does not change the command's result.

Person profiles and GeoIP are disabled on every event.

The PostHog project key is a write-only key built into the package. It is not read from the environment.

## Events

Both events include the app version, install source (`package`, `editable`, `source`, or `vcs`), Python version, OS, architecture, whether `CI` is set, and an optional `invoked_by` label. `invoked_by` is recorded only when a known agent variable is present (`CURSOR_AGENT`, `CLAUDECODE`, `CODEX_SANDBOX`, `GEMINI_CLI`). The variable's value is not read.

### `trace_miner.command_finished`

Sent when a CLI invocation finishes, including errors and interrupts.

- `schema`: `trace_miner.command_finished.v1`
- `command` and `command_path`: `init`, `run`, `stream`, `serve`, `replay`, `stats`, or empty for root help and no args
- `bare`: true when the config path was passed without `run`
- `command_flags`: registered flag names only
- `status`: `completed`, `errored`, or `interrupted`
- `duration_seconds`
- `exit_code`
- `exception_type`: exception class name, when the invocation did not complete
- `run_id`: present when this process started a pipeline

### `trace_miner.run_finished`

Sent when a batch run, stream, or serve finishes, including handled errors and interrupts.

- `schema`: `trace_miner.run_finished.v1`
- `run_id`
- `command`: `run`, `stream`, or `serve`
- `status`: `completed`, `errored`, or `interrupted`
- `duration_seconds`
- `exception_type`
- `pipeline_count`
- `lm_provider` and `lm_model`
- `records_read`, `emitted`, `failed`
- `lm_calls`, `lm_prompt_tokens`, `lm_completion_tokens`, `lm_cost_usd`

Language-model fields are counts and totals. They are not the text sent to or returned by the model.

## Not collected

- raw command arguments and flag values
- local paths, config filenames, stage names, and checkpoint paths
- host and port
- pipeline shape beyond the pipeline count
- LM inputs and outputs: prompts, completions, signature text, stage messages, trace records, and cache entries
- exception messages and tracebacks
- environment variable names or values
- usernames, emails, and organization names
- IP address or GeoIP-derived location
