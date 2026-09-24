"""Post each trace one step at a time into a served doom_loop pipeline.

The server remembers every step already submitted for the session. An alert
named flag_doom_loop means stop. Silence through the posted steps means clean.
"""

import json
import sys
from pathlib import Path

from trace_miner.client import Client

LOOP_TASK = "terminal-bench/coq-block-bound"

# terminal-bench/coq-block-bound really is a doom loop: the agent edits Main.v,
# runs coqc, gets an error, edits again — 40+ compile attempts from step 10 to
# step 217 of a 218-step trace, never once succeeding. The other two are traces
# that only *look* repetitive on the surface but are not:
#   - cargo-flight-dispatch: reruns the same dispatch/test commands, but each
#     run follows a real code change and confirms progress (bug found, fixed,
#     independently verified). Ends in a clean summary of what was fixed.
#   - takens-embedding-lean: greps and rebuilds repeatedly, but each search
#     targets a different Mathlib lemma. Ends with an honest "I could not prove
#     this" rather than a fabricated success.
# All three came from screening a larger real trace corpus with this same
# pipeline, then reading the raw steps by hand — the LM's own "doom_loop"
# verdict was not trusted without that. Kept in their own file rather than
# trajectories.jsonl, which the other pipelines share and check exact task
# sets for.
_EXAMPLES_FILE = "doom_loop_examples.jsonl"
FALSE_POSITIVE_TASKS = {
    "terminal-bench/cargo-flight-dispatch",
    "terminal-bench/takens-embedding-lean",
}


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    client = Client(url)
    here = Path(__file__).resolve().parent
    traces = [
        *_load_traces(here / "input" / "trajectories.jsonl"),
        *_load_traces(here / "input" / _EXAMPLES_FILE),
    ]
    outcomes = []
    for trace in traces:
        session_id = str(trace.get("session_id") or "")
        task = str((trace.get("eval") or {}).get("task_name") or session_id)
        outcome = _run_trace(client, session_id, task, trace.get("steps") or [])
        outcomes.append((task, outcome))
    clean = [task for task, outcome in outcomes if outcome != "doom_loop"]
    loops = [task for task, outcome in outcomes if outcome == "doom_loop"]
    print(f"{len(clean)} clean, {len(loops)} doom loop", flush=True)
    # The LM judge is only guaranteed to get two things right: catching the
    # confirmed real loop, and not crying wolf on the two confirmed look-alikes.
    # A marginal trace among the rest (e.g. one with a run of no-op steps) may
    # fairly go either way.
    problems = []
    if LOOP_TASK not in loops:
        problems.append(f"the real doom-loop trace was never flagged (loops={loops})")
    false_alarms = FALSE_POSITIVE_TASKS & set(loops)
    if false_alarms:
        problems.append(f"look-alike traces were wrongly flagged: {sorted(false_alarms)}")
    if problems:
        raise SystemExit("; ".join(problems))


def _run_trace(client: Client, session_id: str, task: str, steps: list[dict]) -> str:
    with client.watch_events(session_id=session_id) as events:
        for i, step in enumerate(steps, start=1):
            print(f"post {task} step {i}/{len(steps)}", flush=True)
            client.post_step({"session_id": session_id, **step})
            if _flagged(events.take(timeout=0.3), task, i):
                return "doom_loop"
        # Posts run ahead of the LM. Wait once for this session's judgment.
        if _flagged(events.take(timeout=300.0), task, len(steps)):
            return "doom_loop"
    print(f"{task}: no doom loop", flush=True)
    return "clean"


def _flagged(alerts, task: str, step: int) -> bool:
    for alert in alerts:
        if alert.name != "flag_doom_loop":
            continue
        print(f"doom loop in {task} at step {step}: {alert.reason}", flush=True)
        return True
    return False


def _load_traces(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    main()
