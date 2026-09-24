"""Post the bundled traces into a served alerts pipeline and stop on the first abort."""

import json
import sys
from pathlib import Path

from trace_miner.client import Client


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    client = Client(url)
    path = Path(__file__).resolve().parent / "input" / "trajectories.jsonl"
    for trace in _load_traces(path):
        session_id = str(trace.get("session_id") or "")
        task = str((trace.get("eval") or {}).get("task_name") or session_id)
        with client.watch_events(session_id=session_id) as events:
            for step in trace.get("steps") or []:
                record = _step_record(session_id, step)
                print(f"post {task} step {record['step_id']}", flush=True)
                client.post_step(record)
                for alert in events.take(timeout=1.0):
                    if alert.name == "flag_sensitive_step":
                        raise SystemExit(f"aborted {task} step {record['step_id']}: {alert.reason}")
    print("posted every trace")


def _load_traces(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _step_record(session_id: str, step: dict) -> dict:
    names = []
    for call in step.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("function_name"):
            names.append(call["function_name"])
    return {
        "session_id": session_id,
        "step_id": step.get("step_id"),
        "source": step.get("source") or "",
        "message": step.get("message") or "",
        "tool_calls": names,
        "observation": _clip(_observation_text(step), 2000),
    }


def _observation_text(step: dict) -> str:
    observation = step.get("observation") or {}
    results = observation.get("results") or []
    chunks = []
    for result in results[:3]:
        content = result.get("content") if isinstance(result, dict) else None
        if content:
            chunks.append(str(content))
    return " ".join(chunks)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    main()
