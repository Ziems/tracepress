import json

import pytest

import tracepress as tp
from tracepress.adapters.atif import IMAGE_PLACEHOLDER


def test_trials_table(harbor_dir):
    rows = tp.read.harbor(harbor_dir).table("trials").collect()
    assert [r.trial for r in rows] == ["alpha__1", "alpha__2", "beta__1", "beta__2"]
    assert [r.reward for r in rows] == [1.0, 0.0, 1.0, 0.0]
    assert all(isinstance(r.reward, float) for r in rows)
    assert rows[0].key[:2] == (("dataset", "bench/bench"), ("task", "bench/alpha"))
    assert rows[3].exception_type == "NonZeroAgentExitCodeError"
    assert rows[0].duration_s == 60.0


def test_steps_and_tool_calls(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    steps = src.table("steps").collect()
    # The crashed trial has no trajectory, so no steps.
    assert len(steps) == 9
    step = next(s for s in steps if s.trial == "alpha__1" and s.step == 2)
    assert step.agent == "claude-code"
    assert [c["result"] for c in step.tool_calls] == ["a.txt", IMAGE_PLACEHOLDER]

    calls = src.table("tool_calls").collect()
    assert len(calls) == 6
    assert calls[0].key[-1] == ("call", 0) and calls[0].name == "Bash"


def test_tests_table(harbor_dir):
    tests = tp.read.harbor(harbor_dir).table("tests").collect()
    assert len(tests) == 6
    assert {t.status for t in tests if t.trial == "alpha__2"} == {"passed", "failed"}


def test_filters(harbor_dir):
    assert len(tp.read.harbor(harbor_dir, tasks=["beta"]).trial_dirs) == 2
    assert len(tp.read.harbor(harbor_dir, tasks=["bench/alpha"]).trial_dirs) == 2
    assert len(tp.read.harbor(harbor_dir, limit=3).trial_dirs) == 3


def test_group_by_task(harbor_dir):
    trials = tp.read.harbor(harbor_dir).table("trials")
    rows = trials.group_by("task").reduce(lambda rs: {"pass_rate": sum(r.reward for r in rs) / len(rs)}).collect()
    assert [(r.task, r.pass_rate) for r in rows] == [("bench/alpha", 0.5), ("bench/beta", 0.5)]


def test_atif_reader(harbor_dir):
    src = tp.read.atif(harbor_dir / "trials" / "alpha__1" / "agent" / "trajectory.json")
    assert [r.n_steps for r in src.table("trajectories").collect()] == [3]
    assert len(src.table("steps").collect()) == 3


def test_missing_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        tp.read.harbor(tmp_path)


@pytest.mark.parametrize("reader", ["dspy", "chatgpt"])
def test_placeholder_adapters_raise(reader):
    with pytest.raises(NotImplementedError, match="isn't implemented yet"):
        getattr(tp.read, reader)("anything")


def test_atif_reader_jsonl(harbor_dir, tmp_path):
    trajectories = [harbor_dir / "trials" / name / "agent" / "trajectory.json" for name in ("alpha__1", "beta__1")]
    (tmp_path / "bundles").mkdir()
    bundle = tmp_path / "bundles" / "bundle.jsonl"
    bundle.write_text("\n".join(p.read_text() for p in trajectories) + "\n")
    src = tp.read.atif(bundle)
    assert [r.trajectory for r in src.table("trajectories").collect()] == ["alpha__1", "beta__1"]
    assert len(src.table("steps").collect()) == 6
    assert len(tp.read.atif(tmp_path / "bundles").table("trajectories").collect()) == 2


def test_atif_jsonl_trajectories_without_session_ids_stay_apart(tmp_path):
    bundle = tmp_path / "bundle.jsonl"
    bundle.write_text(
        "".join(
            json.dumps({"steps": [{"step_id": 1, "source": "user", "message": f"task {i}"}]}) + "\n" for i in range(3)
        )
    )
    src = tp.read.atif(bundle)
    assert [r.trajectory for r in src.table("trajectories").collect()] == [f"{bundle}:{i}" for i in (1, 2, 3)]
    assert [r.message for r in src.table("steps").collect()] == ["task 0", "task 1", "task 2"]


def test_atif_observation_keeps_every_result_without_a_call_id(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "session_id": "s",
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "observation": {"results": [{"content": "output of ls"}, {"content": "output of pwd"}]},
                    }
                ],
            }
        )
    )
    (step,) = tp.read.atif(path).table("steps").collect()
    assert step.observation == "output of ls\noutput of pwd"


def test_model_from_agent_info(tmp_path):
    """A regraded job runs a placeholder agent with no model_name; agent_info still names the model."""
    import json

    from conftest import write_trial

    trials = tmp_path / "job" / "trials"
    trials.mkdir(parents=True)
    write_trial(trials, "alpha", "alpha__1", 1.0)
    result_path = trials / "alpha__1" / "result.json"
    result = json.loads(result_path.read_text())
    result["config"]["agent"] = {"name": "nop", "model_name": None}
    result["agent_info"] = {"name": "nop", "model_info": {"name": "claude-opus-5-5", "provider": "anthropic"}}
    result_path.write_text(json.dumps(result))
    (row,) = tp.read.harbor(tmp_path / "job").table("trials").collect()
    assert row.model == "claude-opus-5-5"


def test_dataset_from_regrade_lock(tmp_path):
    """A regraded trial's result has no source; lock.json keeps the original trial's."""
    import json

    from conftest import write_trial

    trials = tmp_path / "job" / "trials"
    trials.mkdir(parents=True)
    write_trial(trials, "alpha", "alpha__1", 1.0)
    result_path = trials / "alpha__1" / "result.json"
    result = json.loads(result_path.read_text())
    del result["source"]
    result_path.write_text(json.dumps(result))
    (trials / "alpha__1" / "lock.json").write_text(json.dumps({"source_trial": {"task": {"source": "bench/bench"}}}))
    (row,) = tp.read.harbor(tmp_path / "job").table("trials").collect()
    assert row.key[0] == ("dataset", "bench/bench")
