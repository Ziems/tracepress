# Hello Trace Miner

This project is a small harbor-style workspace. One `stages.py` module, three
slim harbor traces, and seven pipelines that reuse those stages.

The traces are real harbor runs (failed ATRX/VEP CRISPR, failed photonic
routing, passed FreeCAD spring clip) with planted signals so each pipeline has
something to find: a fake SSN, a fake phone number, and a frustrated user turn.
The doom-loop demo adds three more real traces with no planted signal: one
genuine doom loop and two that only look like one.

Language-model stages need OpenRouter:

```bash
export OPENROUTER_API_KEY=...
```

## Pipelines

| Config | What it does | Output |
| --- | --- | --- |
| `summarize.yaml` | Summarize each step, then diagnose the trajectory | `output/summaries.jsonl` |
| `pii.yaml` | Detect SSN / phone / email in each step | `output/pii.jsonl` |
| `frustration.yaml` | Detect user frustration or impatience | `output/frustration.jsonl` |
| `category.yaml` | Label domain and outcome | `output/categories.jsonl` |
| `classify.yaml` | Classify a trajectory with Jev, without summarizing each step | `output/classifications.jsonl` |
| `alerts.yaml` | Alert when a step contains an SSN, a phone number, or a frustrated user turn | `output/alerts.jsonl` |
| `doom_loop.yaml` | Reduce over the last few steps at every posted step; alert if the agent is stuck repeating itself | `output/doom_loop.jsonl` |

## Run

```bash
trace-miner run summarize.yaml --in input/trajectories.jsonl --out output/summaries.jsonl
```

Then try the others on the same traces:

```bash
trace-miner run pii.yaml --in input/trajectories.jsonl --out output/pii.jsonl
trace-miner run frustration.yaml --in input/trajectories.jsonl --out output/frustration.jsonl
trace-miner run category.yaml --in input/trajectories.jsonl --out output/categories.jsonl
trace-miner run classify.yaml --in input/trajectories.jsonl --out output/classifications.jsonl
```

## Alerts

`flag_sensitive_step` is an `alert` step. It returns a reason when a posted step contains an SSN, a phone number, or the phrase "taking forever". With `--out`, the posted step lands in that JSONL. The alert itself is written under `.trace-miner/alerts/`. Posted records are stored in `.trace-miner/inbox/alerts.jsonl`.

In one terminal:

```bash
trace-miner serve alerts.yaml --out output/alerts.jsonl
```

In another, from this directory:

```bash
python client_demo.py
```

`client_demo.py` posts each step of `input/trajectories.jsonl`. It stops at the first alert, which is the SSN planted in the ATRX trace:

```text
aborted terminal-bench/atrx-vep-crispr step 2: ssn
```

The later traces carry a frustrated user turn and a phone number. Those fire if you remove the earlier signal.

## Doom loop

`doom_loop.yaml` is served the same way as `alerts.yaml`, but its posted records carry a *window of recent
steps* instead of one step. Its `group` runs `detect_doom_loop`, an LM reduce,
over that window on every post, so the trace is judged again each time new
steps arrive rather than only once at the end. `flag_doom_loop` alerts when the
reduce says the agent is stuck repeating itself. `flag_rollout_complete` alerts
once, when a posted record carries `final: true` — that's the built-in signal
for "this rollout has finished and its last step was submitted."

There's no server-side memory of a trace across posts, so the *client* keeps a
sliding window of the last few steps and resends it on every post. DSPy's
request cache means re-summarizing a step already seen in an earlier window is
free; only the reduce judgment itself runs fresh each time.

In one terminal:

```bash
trace-miner serve doom_loop.yaml --out output/doom_loop.jsonl
```

In another, from this directory:

```bash
python doom_loop_demo.py
```

`doom_loop_demo.py` posts the three bundled traces plus three more from
`input/doom_loop_examples.jsonl`, none of them planted or fabricated:

- `coq-block-bound` really is a doom loop. The agent edits a Coq proof file,
  compiles, gets an error, edits again — 40+ compile attempts from step 10 to
  step 217 of a 218-step trace, never once succeeding. It reliably aborts
  mid-stream:

  ```text
  doom loop in terminal-bench/coq-block-bound step 18: Repeatedly edited Main.v with search_replace, then recompiled it; Coq continued to fail with proof/tactic/unification errors, without achieving a successful proof or changing approach.
  ```

  (the exact wording and step are the LM's, so both vary run to run — the
  pattern starts around step 10 and never lets up, so it should catch on
  early)

- `cargo-flight-dispatch` and `takens-embedding-lean` only *look* like doom
  loops from a one-line-per-step summary — the same commands recur, but each
  run follows a real change or targets a different hypothesis, and both end
  with the agent reporting an honest outcome (a fix it verified, or a proof it
  couldn't find) rather than repeating itself. Both should finish with
  `rollout complete (clean)`.

All three came from screening 157 real failed trajectories with this same
pipeline, then reading the raw steps by hand before trusting any verdict — the
LM's own "doom_loop: true" is not proof by itself. An earlier, looser version
of `detect_doom_loop`'s prompt flagged 23 of those 157 traces; most turned out
to be traces like the two look-alikes here. Tightening the signature's
criteria (see `AGENTS.md`) cut that to 1 false alarm out of 157.

The three original bundled traces should finish with `rollout complete
(clean)` too. A run of empty or no-op agent turns is idle, not a doom loop.

## What each file is

- The YAML files are topology only. They name stages and how they connect.
- `stages.py` is ordinary Python plus DSPy signatures. Changing a stage here
  changes every pipeline that names it. `detect_pii_in_step` uses a Pydantic
  output type (`list[PiiFinding]`) so the model returns structured spans.
- `input/trajectories.jsonl` is three slim harbor records.
- `input/doom_loop_examples.jsonl` is three real harbor records — one genuine
  doom loop, two confirmed look-alikes — their largest fields clipped for
  size. `doom_loop_demo.py` reads it.
- `alerts.yaml` is served with `trace-miner serve`. `client_demo.py` posts steps
  over HTTP. The inbox is `.trace-miner/inbox/alerts.jsonl`.
- `doom_loop.yaml` is served the same way. `doom_loop_demo.py` posts step
  windows and watches for `final: true` to know a rollout ended.
- `AGENTS.md` is the working guide for coding agents: how to run, add a
  pipeline, and write Python or LM stages.

## Change it

Edit a signature in `stages.py`, or copy `pii.yaml` to start a new pipeline
over the same traces. Prefer a Pydantic model when an LM output is an object
or a list of objects.
