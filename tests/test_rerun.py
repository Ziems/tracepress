"""What-if reruns (`Table.rerun`) and lineage (`collect(lineage=True)`, `Row.inputs()`)."""

import pytest

import tracepress as tp


def passed_per_task(trials):
    return trials.group_by("task").reduce(
        lambda rs: {"passed": sum((r.reward or 0) >= 1 for r in rs)}, columns={"passed": int}
    )


def by_task(rows, col):
    return {r.task: r.get(col) for r in rows}


def test_rerun_with_new_values(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    per_task = passed_per_task(src.trials)
    what_if = per_task.rerun(edit={("trial", "alpha__2"): {"reward": 1.0}})
    assert by_task(what_if.collect(), "passed")["bench/alpha"] == 2
    assert by_task(per_task.collect(), "passed")["bench/alpha"] == 1  # the original plan is untouched


def test_rerun_as_another_row(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    what_if = passed_per_task(src.trials).rerun(edit={("trial", "alpha__2"): ("trial", "alpha__1")})
    assert by_task(what_if.collect(), "passed")["bench/alpha"] == 2


def test_rerun_edits_the_nearest_table_at_that_level(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    notes = src.steps.group_by("trial").reduce(lambda rs: {"note": "as run"}, columns={"note": str})
    per_task = (
        notes.join(src.trials, on="trial")
        .group_by("task")
        .reduce(lambda rs: {"notes": sorted(r.note for r in rs)}, columns={"notes": list})
    )
    rows = per_task.rerun(edit={("trial", "alpha__1"): {"note": "edited"}}).collect()
    assert by_task(rows, "notes")["bench/alpha"] == ["as run", "edited"]


def test_rerun_errors(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    per_task = passed_per_task(src.trials)
    with pytest.raises(ValueError, match="keyed by 'step'"):
        per_task.rerun(edit={("step", 1): {"reward": 1.0}})
    with pytest.raises(ValueError, match="no row with key"):
        per_task.rerun(edit={("trial", "nope"): {"reward": 1.0}}).collect()
    with pytest.raises(ValueError, match="to copy from"):
        per_task.rerun(edit={("trial", "alpha__2"): ("trial", "nope")})
    with pytest.raises(ValueError, match="pair"):
        per_task.rerun(edit={"alpha__2": {"reward": 1.0}})


def test_inputs_walk_down_the_hierarchy(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    per_trial = (
        src.steps.join(src.trials, on="trial").group_by("trial").reduce(lambda rs: {"n": len(rs)}, columns={"n": int})
    )
    per_task = per_trial.group_by("task").reduce(lambda rs: {"runs": len(rs)}, columns={"runs": int})
    alpha = next(r for r in per_task.collect(lineage=True) if r.task == "bench/alpha")
    trials = alpha.inputs()
    assert sorted(r.trial for r in trials) == ["alpha__1", "alpha__2"]
    steps = trials[0].inputs()
    assert len(steps) == trials[0].n and steps[0].instruction == "Please solve alpha."  # joined columns kept
    assert steps[0].inputs() == []  # a source row


def test_inputs_need_lineage(harbor_dir):
    row = passed_per_task(tp.read.harbor(harbor_dir).trials).collect()[0]
    with pytest.raises(ValueError, match="lineage=True"):
        row.inputs()
