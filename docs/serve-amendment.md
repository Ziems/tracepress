# Amendment: Serving Alerts

An amendment to [`pipeline-spec.md`](pipeline-spec.md) and [`streaming-amendment.md`](streaming-amendment.md). Read those first. This document adds a network boundary so an eval process on one machine can send steps to a pipeline on another and hear alerts.

JSONL paths and checkpoints stay on the pipeline host. The client only has a base URL.

## N1. What this adds

`serve` is the live path. It runs the follow runtime and an HTTP server in one process:

```bash
trace-miner serve pipeline.yaml --out steps.jsonl --host 0.0.0.0 --port 8765
```

It binds `127.0.0.1` unless `--host` is set. `--out` is optional.

- `POST /v1/records` appends one JSON object as a newline-terminated line to `.trace-miner/inbox/<config-stem>.jsonl` beside the config and returns `202` once that append is flushed. It does not wait for the pipeline.
- `GET /v1/events` is a Server-Sent Events response for one named pipeline's alert log.

`--out` writes records that finish the pipeline. Omit it and only the alert log is written.

## N2. Names

A pipeline may declare `name`. The name is optional, unique, and is the only public handle for that pipeline. A pipeline with no name is not subscribable. The client never sends a filesystem path.

```yaml
pipelines:
  - name: alerts
    pipeline:
      - group:
          expand: split_steps
          pipeline:
            - map: summarize_step
          reduce: detect_doom_loop
      - alert: flag_doom_loop
```

`serve` publishes that pipeline's alert log, `.trace-miner/alerts/<name>.jsonl`, under `name`. It does not publish `--out` or `tee` files. Subscribers receive alert events only.

`GET /v1/events?output=alerts&field=rollout_id&rollout_id=r1` keeps records whose named field matches. The field is whatever the caller is correlating, `rollout_id` in this example. When the config sets `id_field`, that field is the filter and the `field` query must match it. Otherwise `field` is required, and the value is the query parameter of that name.

SSE `id` is opaque. The server stores file offsets behind it. A reconnect sends `Last-Event-ID` and receives records written after that cursor. A connection with no `Last-Event-ID` receives only records written after it subscribes.

## N3. Alert events

Stages do not learn about the network. In this pipeline each rollout is split into steps, each step is summarized, and `detect_doom_loop` reduces those summaries. `step_summary: list[str]` is gathered from the children. The bool and the pattern merge back onto the rollout, which still carries `id_field`.

```python
class DetectDoomLoop(dspy.Signature):
    """Decide if these summaries show an agent stuck repeating itself."""

    step_summary: list[str] = dspy.InputField()

    doom_loop: bool = dspy.OutputField(desc="True when the agent is stuck doing the same thing over and over.")
    pattern: str = dspy.OutputField(desc="What it keeps repeating, or an empty string.")

detect_doom_loop = dspy.Predict(DetectDoomLoop)


def flag_doom_loop(record):
    if not record.get("doom_loop"):
        return None
    return record.get("pattern") or "stuck repeating itself"
```

```json
{"name": "flag_doom_loop", "reason": "retrying the same pytest command", "rollout_id": "r1"}
```

`name` is the stage name. `reason` is the string a Python `alert` stage returned. `rollout_id` here is `id_field`, copied from the record or from the group parent. The record that continues down the pipeline is unchanged, and it is what `--out` receives when that flag is set. A DSPy `alert` stage, used directly rather than as a reduce, has exactly one bool output. False stays quiet. True writes the module's outputs onto the event.

## N4. Client

`Client` is a sync helper. `post_step` only sends. `watch_events` reads the alert stream for the one pipeline this server is running. A background thread reads it. `events.take()` returns every alert that arrived since the last call, in arrival order. `alert.reason` is the string a Python stage returned. A DSPy alert leaves it empty and puts the module outputs on `alert.record`. `alert.name` is the stage name. The caller decides which names are worth acting on. The library does not raise.

```python
client = Client("http://pipeline-host:8765")

with client.watch_events(rollout_id="r1") as events:
    for step in agent.run():
        client.post_step(step)
        for alert in events.take():
            if alert.name == "flag_doom_loop":
                raise EvalAborted(alert.reason)
```

The keyword is the field name. `rollout_id` is just the field in this example; `watch_events(session_id="s1")` matches `session_id` instead. `watch_events` opens the SSE connection before the loop starts, reconnects with `Last-Event-ID` if the connection drops, and joins the listener on exit. `take()` without a timeout returns whatever is already queued. `take(timeout=1)` waits up to that many seconds for the next alert. The posted record already carries the same id.

Each POST is one step. The server keeps every step already accepted for that id and runs the pipeline on the session so far. The inbox line is the latest step plus `steps`, the list of submitted steps. Restarting the server forgets that list. `serve` refuses to start when `lm.cache` is false, because already-submitted steps are reused from that cache.

HTTP remains the contract. Another language can POST and read `text/event-stream` without this helper. Omit `output` on `GET /v1/events` when the server has one pipeline.

## N5. Non-goals

Do not add webhooks, sockets as a pipeline source, or a message broker. Do not add `ctx.side`. Do not make a stage name a subscriber or a host.
