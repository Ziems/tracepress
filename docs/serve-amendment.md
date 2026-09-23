# Amendment: Serving Alerts

An amendment to [`pipeline-spec.md`](pipeline-spec.md) and [`streaming-amendment.md`](streaming-amendment.md). Read those first. This document adds a network boundary so an eval process on one machine can send steps to a pipeline on another and hear alerts.

JSONL paths and checkpoints stay on the pipeline host. The client only has a base URL.

## N1. What this adds

`serve` is the live path. It runs the follow runtime and an HTTP server in one process:

```bash
trace-miner serve pipeline.yaml --host 0.0.0.0 --port 8765
```

It binds `127.0.0.1` unless `--host` is set.

- `POST /v1/records` appends one JSON object as a newline-terminated line to the single root input and returns `202` once that append is flushed. It does not wait for the pipeline.
- `GET /v1/events` is a Server-Sent Events response for one named pipeline's alert log.

A config with more than one root input is a startup error.

## N2. Names

A pipeline may declare `name`. The name is optional, unique, and is the only public handle for that pipeline. A pipeline with no name is not subscribable. The client never sends a filesystem path.

```yaml
pipelines:
  - name: alerts
    in: input/alerts.jsonl
    out: output/steps.jsonl
    pipeline:
      - group:
          expand: split_steps
          pipeline:
            - map: summarize_step
          reduce: detect_doom_loop
      - alert: flag_doom_loop
```

`serve` publishes that pipeline's alert log, `.trace-miner/alerts/<name>.jsonl`, under `name`. It does not publish `out` or `tee` files. Subscribers receive alert events only.

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

`name` is the stage name. `reason` is the string a Python `alert` stage returned. `rollout_id` here is `id_field`, copied from the record or from the group parent. The record that continues down the pipeline is unchanged, and it is what `out` receives. A DSPy `alert` stage, used directly rather than as a reduce, has exactly one bool output. False stays quiet. True writes the module's outputs onto the event.

## N4. Client

`Client` is a sync helper. `post_step` only sends. `watch` returns the alert stream for one named pipeline. A background thread reads it. `alerts.latest` stays `None` until an alert event arrives. A newer event replaces it. `alerts.latest.reason` is the string a Python stage returned. A DSPy alert leaves it empty and puts the module outputs on `alerts.latest.record`. `alerts.latest.name` is the stage name. Any alert is stop-worthy. The library does not raise.

```python
client = Client("http://pipeline-host:8765")

with client.watch("alerts", rollout_id="r1") as alerts:
    for step in agent.run():
        if alerts.latest:
            raise EvalAborted(alerts.latest.reason)
        client.post_step(step)
```

The keyword is the field name. `rollout_id` is just the field in this example; `watch("alerts", session_id="s1")` matches `session_id` instead. `watch` opens the SSE connection before the loop starts, reconnects with `Last-Event-ID` if the connection drops, and joins the listener on exit. One step may still run after the alert, because the check is at the top of the loop. The posted record already carries the same id.

HTTP remains the contract. Another language can POST and read `text/event-stream` without this helper.

## N5. Non-goals

Do not add webhooks, sockets as a pipeline source, or a message broker. Do not add `ctx.side`. Do not make a stage name a subscriber or a host.
