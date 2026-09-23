"""Post a trace's steps into a served doom_loop pipeline, one step at a time.

Each post carries a sliding window of the last WINDOW_SIZE steps rather than the
whole trace so far, so the reduce stays a constant-size judgment call at every
step. DSPy's cache means re-summarizing a step already in an earlier window costs
nothing. The last step of each trace is posted with `final: true`, which is the
signal `flag_rollout_complete` watches for.
"""

import json
import sys
import time
from pathlib import Path

from trace_miner.client import Client

WINDOW_SIZE = 8
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
    window: list[dict] = []
    with client.watch("doom_loop", session_id=session_id) as alerts:
        for i, step in enumerate(steps, start=1):
            window.append(step)
            if len(window) > WINDOW_SIZE:
                window.pop(0)
            record = {"session_id": session_id, "steps": window, "final": i == len(steps)}
            print(f"post {task} step {i}/{len(steps)}", flush=True)
            client.post_step(record)
            # Most steps never fire an alert, so this is a brief opportunistic
            # check, not a wait for the server to finish — `serve` processes
            # posts strictly in order, so racing ahead is safe; we only pay a
            # longer wait once, below, for whichever verdict is still in flight.
            _wait_for_alert(alerts, 0.3)
            outcome = _outcome_from_alert(alerts, task, i)
            if outcome is not None:
                return outcome
        # Posting outruns processing on a long trace, so give the server real
        # time to drain its backlog and deliver the last step's verdict. 60s
        # was measured too short here: six traces posted back to back build a
        # real backlog, and a trace's own verdict doesn't arrive until the
        # server has worked through everything queued ahead of it too.
        _wait_for_alert(alerts, 300.0)
        outcome = _outcome_from_alert(alerts, task, len(steps))
        if outcome is not None:
            return outcome
    print(f"{task}: no verdict posted", flush=True)
    return "clean"


def _outcome_from_alert(alerts, task: str, step: int) -> str | None:
    if not alerts.latest:
        return None
    reason = alerts.latest.reason or ""
    if reason.startswith("rollout complete"):
        print(f"{task}: {reason}", flush=True)
        return "doom_loop" if "doom loop" in reason else "clean"
    print(f"doom loop in {task} at step {step}: {reason}", flush=True)
    return "doom_loop"


def _load_traces(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _wait_for_alert(alerts, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while alerts.latest is None and time.monotonic() < deadline:
        time.sleep(0.02)


if __name__ == "__main__":
    main()
