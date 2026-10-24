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
    assert len(tp.read.harbor(harbor_dir, tasks=["beta"]).trials) == 2
    assert len(tp.read.harbor(harbor_dir, tasks=["bench/alpha"]).trials) == 2
    assert len(tp.read.harbor(harbor_dir, limit=3).trials) == 3


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


@pytest.mark.parametrize("reader", ["dspy", "otel", "claude_code", "chatgpt"])
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
