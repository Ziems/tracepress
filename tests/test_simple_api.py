"""Short pipelines: `src.steps`, whole-row inputs, joins, group constants, whole-table reduces."""

import asyncio
import json

import pytest
from fakes import FuncLM

import tracepress as tp
from tracepress.adapters.base import Adapter
from tracepress.rows import Closed, Row, RowAdded
from tracepress.schema import Schema


def echo(**outputs):
    """A fake LM that records each call's inputs and answers with fixed outputs."""
    seen = []

    def fn(fields):
        seen.append(fields)
        return outputs

    lm = FuncLM(fn)
    lm.seen = seen
    return lm


def test_tables_as_attributes(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    assert src.steps.schema == src.table("steps").schema
    with pytest.raises(AttributeError, match="no attribute or table 'stepz'"):
        src.stepz


def test_step_input_is_the_whole_step(harbor_dir):
    lm = echo(summary="did a thing")
    rows = tp.read.harbor(harbor_dir).steps.lm_map("step -> summary", lm=lm).collect()
    assert len(rows) == 9 and rows[0].summary == "did a thing"
    step = next(s for s in map(json.loads, (f["step"] for f in lm.seen)) if s["message"] == "Looking around.")
    assert step["source"] == "agent"
    assert step["tool_calls"][0]["name"] == "Bash"
    assert "timestamp" not in step and "model" not in step  # bookkeeping columns stay out


def test_trials_have_instruction(harbor_dir):
    trials = {r.trial: r for r in tp.read.harbor(harbor_dir).trials.collect()}
    assert trials["alpha__1"].instruction == "Please solve alpha."
    assert trials["beta__2"].instruction is None  # crashed before writing a trajectory


def test_join_adds_parent_columns(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    rows = src.steps.join(src.trials).collect()
    assert len(rows) == 9
    first = rows[0]
    assert first.instruction == "Please solve alpha." and first.reward == 1.0
    assert first.message == "Please solve alpha."  # the step's own columns are untouched
    assert {r.trial: r.reward for r in rows}["alpha__2"] == 0.0


def test_join_needs_a_prefix(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    with pytest.raises(ValueError, match="prefix"):
        src.trials.join(src.steps)


def test_reduce_lifts_group_constants_and_carries_them(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    lm = echo(what_went_wrong="nothing")
    per_trial = (src.steps.join(src.trials).group_by("trial")
                 .lm_reduce("instruction, reward, message -> what_went_wrong", lm=lm))
    rows = per_trial.collect()
    assert len(rows) == 3
    call = next(f for f in lm.seen if f["instruction"] == "Please solve alpha.")
    assert call["reward"] == "1.0"  # one scalar, not a value per step
    assert '"reward"' not in call["steps"] and '"message"' in call["steps"]
    # Trial-level columns ride along onto the per-trial rows, for the next level up.
    assert {r.trial: r.reward for r in rows} == {"alpha__1": 1.0, "alpha__2": 0.0, "beta__1": 1.0}
    assert rows[0].model == "anthropic/claude-opus-5"
    assert "reward" in per_trial.schema.columns


def test_hierarchy_compares_trials_by_model(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    summaries = src.steps.lm_map("step -> summary", lm=echo(summary="s"))
    trajectories = summaries.join(src.trials).group_by("trial").lm_reduce(
        "instruction, reward, summary -> trajectory_summary", lm=echo(trajectory_summary="t"))
    lm = echo(where_they_diverged="here")
    contrast = trajectories.group_by("task").lm_reduce(
        "model, reward, trajectory_summary -> where_they_diverged", lm=lm)
    rows = contrast.collect()
    assert [r.task for r in rows] == ["bench/alpha", "bench/beta"]
    trials = json.loads(next(f["trials"] for f in lm.seen if "1.0" in f["trials"] and "0.0" in f["trials"]))
    assert [t["reward"] for t in trials] == [1.0, 0.0]
    assert trials[0]["model"] == "anthropic/claude-opus-5"


def test_reduce_over_whole_steps(harbor_dir):
    lm = echo(doom_loop="False")
    rows = tp.read.harbor(harbor_dir).steps.group_by("trial").lm_reduce("step -> doom_loop: bool", lm=lm).collect()
    assert [r.doom_loop for r in rows] == [False] * 3
    steps = json.loads(lm.seen[0]["steps"])
    assert steps[0] == {"source": "user", "message": "Please solve alpha."}


def test_fold_reads_whole_step_and_constants(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    lm = echo(progress="p")
    rows = (src.steps.join(src.trials).group_by("trial")
            .lm_fold("progress, instruction, step -> progress", lm=lm).collect())
    assert len(rows) == 3
    assert lm.seen[0]["instruction"] == "Please solve alpha."
    assert json.loads(lm.seen[0]["step"])["message"] == "Please solve alpha."


def test_group_by_nothing_reduces_the_whole_table(harbor_dir):
    lm = echo(overall="all good")
    rows = tp.read.harbor(harbor_dir).trials.group_by().lm_reduce("reward -> overall", lm=lm).collect()
    assert len(rows) == 1 and rows[0].key == () and rows[0].overall == "all good"
    assert lm.calls == 1


# --- Join on a stream: rows that arrive before their match wait for it --------------------


class Slow(Adapter):
    """Steps arrive at once; their trials arrive later, one with no trial row at all."""

    levels = ("trial", "step")

    def tables(self):
        return {"steps": Schema(self.levels, {"msg": str}), "trials": Schema(self.levels[:1], {"reward": float})}

    async def events(self, table):
        if table == "steps":
            for t in ("t1", "t2", "t3"):
                for i in (1, 2):
                    yield RowAdded(Row((("trial", t), ("step", i)), {"msg": f"{t}-{i}"}))
                yield Closed((("trial", t),))
        else:
            await asyncio.sleep(0.01)
            for t, reward in (("t2", 0.0), ("t1", 1.0)):
                yield RowAdded(Row((("trial", t),), {"reward": reward}))
                yield Closed((("trial", t),))


def test_join_waits_for_late_matches():
    src = Slow()
    joined = src.steps.join(src.trials)
    rows = joined.collect()
    assert [(r.msg, r.get("reward", None)) for r in rows] == [
        ("t1-1", 1.0), ("t1-2", 1.0), ("t2-1", 0.0), ("t2-2", 0.0), ("t3-1", None), ("t3-2", None)]
    # Each trial's close comes after its joined rows, so reduces downstream see every step.
    counts = joined.group_by("trial").reduce(lambda rs: {"n": len(rs), "r": rs[0].get("reward", None)}).collect()
    assert [(r.trial, r.n, r.r) for r in counts] == [("t1", 2, 1.0), ("t2", 2, 0.0), ("t3", 2, None)]


def test_join_stops_cleanly_when_a_run_fails():
    """A billing error downstream of a join (still waiting on its right side) ends the run."""
    import dspy

    def broke(fields):
        raise dspy.LMBillingError("exceeds your available credits (402)", model="m")

    src = Slow()
    table = src.steps.join(src.trials).lm_map("msg -> topic", lm=FuncLM(broke))
    with pytest.raises(dspy.LMBillingError):
        table.collect()
    assert tp.last_run().status == "failed"
