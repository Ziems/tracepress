# Doom loop

Post one step at a time. The server remembers every step already submitted for that session, summarizes the new one, and judges the list. An alert means stop. Silence means keep going.

```bash
trace-miner serve cookbook/doom_loop.yaml --out output/doom_loop.jsonl
python cookbook/doom_loop_demo.py
```

The pipeline is `cookbook/doom_loop.yaml`.

```text
for step in session.steps:
    summary = summarize(step)            # map; a step already summarized is cached
verdict = detect_doom_loop(summaries)    # reduce; runs again because the list grew
if verdict.doom_loop:
    alert(verdict.pattern)               # flag_doom_loop
```

`detect_doom_loop` asks Jev for the bool. It writes `pattern` only when that bool is true.

Add a use case by appending a section here, then a trace under `input/` when you have one. `doom_loop_demo.py` requires the cases marked required.

## Repeated compile, same error

Required. Flag. Trace: `terminal-bench/coq-block-bound`.

```text
edit Main.v
coqc -> error
edit Main.v
coqc -> same error
# dozens of times, the approach never changes
```

## Same command after a real change

Required. Stay quiet. Trace: `terminal-bench/cargo-flight-dispatch`.

```text
edit the source
run the tests -> fail
edit the source
run the tests -> pass
# the command repeats; the result moves
```

## Same tool, different target

Required. Stay quiet. Trace: `terminal-bench/takens-embedding-lean`.

```text
grep lemma A
grep lemma B
build
# the command shape repeats; each step asks a different question
```

## Empty turns

Stay quiet. A run of these is idle, not a loop.

```text
# no tool, no message
# no tool, no message
```

## Polling that moves

Stay quiet. No trace yet.

```text
check the job -> 10%
check the job -> 40%
check the job -> done
```

## One error, then a new approach

Stay quiet. No trace yet.

```text
run the command -> error
read the error
try a different command -> progress
```
