import json
from pathlib import Path

import dspy
import pytest

import tracepress as tp

IMAGE = json.dumps({"type": "image", "source": {"type": "base64", "data": "AAAA"}})


@pytest.fixture(autouse=True)
def chat_adapter():
    """DSPy's DummyLM answers in the chat (text marker) format, not JSON."""
    previous = tp.configure().adapter
    tp.configure(adapter=dspy.ChatAdapter())
    yield
    tp.configure(adapter=previous)


def _step(step_id, source, message, calls=()):
    step = {"step_id": step_id, "timestamp": f"2026-01-01T00:00:0{step_id}Z", "source": source,
            "message": message, "extra": {"is_sidechain": False}}
    if calls:
        step["tool_calls"] = [
            {"tool_call_id": f"c{step_id}{i}", "function_name": name, "arguments": args}
            for i, (name, args, _) in enumerate(calls)
        ]
        step["observation"] = {"results": [
            {"source_call_id": f"c{step_id}{i}", "content": out} for i, (_, _, out) in enumerate(calls)
        ]}
    return step


def write_trial(trials: Path, task: str, trial: str, reward, steps=None, tests=None, exception=None):
    d = trials / trial
    (d / "agent").mkdir(parents=True)
    (d / "verifier").mkdir()
    config = {"trial_name": trial, "job_id": "job-1",
              "agent": {"name": "claude-code", "model_name": "anthropic/claude-opus-5", "kwargs": {}}}
    (d / "config.json").write_text(json.dumps(config))
    result = {
        "trial_name": trial, "task_name": f"bench/{task}", "source": "bench/bench", "config": config,
        "agent_result": {"n_input_tokens": 100, "n_output_tokens": 10, "n_cache_tokens": 0, "cost_usd": 0.01},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": {"exception_type": exception, "exception_message": "boom"} if exception else None,
        "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:01:00Z",
    }
    (d / "result.json").write_text(json.dumps(result))
    if steps is not None:
        (d / "agent" / "trajectory.json").write_text(json.dumps(
            {"schema_version": "ATIF-v1.7", "session_id": trial,
             "agent": {"name": "claude-code", "model_name": "claude-opus-5"}, "steps": steps}))
    if tests is not None:
        (d / "verifier" / "ctrf.json").write_text(json.dumps(
            {"results": {"tests": [{"name": n, "status": s, "duration": 1} for n, s in tests]}}))


@pytest.fixture
def harbor_dir(tmp_path):
    """Two tasks x two trials. One trial crashed without a trajectory."""
    trials = tmp_path / "job" / "trials"
    trials.mkdir(parents=True)
    for task in ("alpha", "beta"):
        for n in (1, 2):
            name = f"{task}__{n}"
            if task == "beta" and n == 2:
                write_trial(trials, task, name, 0, exception="NonZeroAgentExitCodeError")
                continue
            steps = [
                _step(1, "user", f"Please solve {task}."),
                _step(2, "agent", "Looking around.", [("Bash", {"command": "ls"}, "a.txt"),
                                                     ("Read", {"file_path": "x.png"}, IMAGE)]),
                _step(3, "agent", "Done."),
            ]
            write_trial(trials, task, name, 1 if n == 1 else 0.0, steps=steps,
                        tests=[("test_a", "passed"), ("test_b", "failed" if n == 2 else "passed")])
    return tmp_path / "job"
