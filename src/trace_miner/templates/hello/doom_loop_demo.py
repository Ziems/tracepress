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
LOOP_TASK = "terminal-bench/vf2-speedup-networkx"

# A real terminal-bench trajectory (same Harbor job as the other bundled traces)
# where the agent backgrounds a slow test, then checks on it with `sleep N; cat
# task.output` three times in a row and gets nothing back each time before it
# finally thinks to check whether the process is even still running. Kept in its
# own file rather than trajectories.jsonl, which the other pipelines share and
# check exact task sets for.
_LOOP_TRACE_FILE = "doom_loop_example.jsonl"


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    client = Client(url)
    here = Path(__file__).resolve().parent
    traces = [
        *_load_traces(here / "input" / "trajectories.jsonl"),
        *_load_traces(here / "input" / _LOOP_TRACE_FILE),
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
    # The LM judge is only guaranteed to catch the planted example; a marginal
    # real trace (e.g. one with a run of no-op steps) may fairly go either way.
    if LOOP_TASK not in loops:
        raise SystemExit(f"the real doom-loop trace was never flagged: clean={clean} loops={loops}")


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
        # time to drain its backlog and deliver the last step's verdict.
        _wait_for_alert(alerts, 60.0)
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
