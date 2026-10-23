import asyncio

import pytest
from fakes import FuncLM

import tracepress as tp
from tracepress.rows import Closed, Row

TRIALS = ("alpha__1", "alpha__2", "beta__1")
FAULTS = [
    {},
    {"shuffle_within": 4},
    {"drop_closes": 0.5},
    {"duplicates": 0.5},
    {"shuffle_within": 4, "drop_closes": 0.3, "duplicates": 0.3},
]


def trail_lm():
    """Appends the newest step to the trail, so the result records the order steps were folded in."""
    return FuncLM(lambda f: {"trail": f"{f['previous_trail']}|{f['text']}", "count": f["previous_trail"].count("|") + 1})


def count_lm():
    """Running updates say 'partial'; the full reduce counts the steps it was given."""
    def answer(f):
        if "current" in f:
            return {"summary": f"partial+{f['new_steps'].count('text')}"}
        return {"summary": f"full:{f['steps'].count('text')}"}

    return FuncLM(answer)


def steps_of(src):
    return src.table("steps").map(lambda r: {"text": str(r.step)}, columns={"text": str})


def contents(rows):
    """What batch and stream must agree on: keys and values. Versions record update history."""
    return [(r.key, r.values) for r in rows]


def collect_events(table):
    async def run():
        return [e async for e in table.events()]

    return asyncio.run(run())


# --- lm_fold -----------------------------------------------------------------------------


def test_fold_runs_children_in_order(harbor_dir):
    lm = trail_lm()
    rows = steps_of(tp.read.harbor(harbor_dir)).group_by("trial").lm_fold("trail, text -> trail, count: int", lm=lm).collect()
    assert [(r.trial, r.trail, r.count) for r in rows] == [(t, "|1|2|3", 3) for t in TRIALS]
    assert lm.calls == 9


def test_fold_init_and_validation(harbor_dir):
    steps = steps_of(tp.read.harbor(harbor_dir))
    rows = steps.group_by("trial").lm_fold("trail, text -> trail, count: int", init={"trail": "start"}, lm=trail_lm()).collect()
    assert rows[0].trail == "start|1|2|3"
    with pytest.raises(ValueError, match="state field"):
        steps.group_by("trial").lm_fold("text -> trail")
    with pytest.raises(ValueError, match="non-state"):
        steps.group_by("trial").lm_fold("trail, text -> trail", init={"text": "x"})
    with pytest.raises(ValueError, match="nope"):
        steps.group_by("trial").lm_fold("trail, nope -> trail")


@pytest.mark.parametrize("faults", FAULTS)
def test_fold_replay_matches_batch(harbor_dir, faults):
    src = tp.read.harbor(harbor_dir)
    fold = lambda s: steps_of(s).group_by("trial").lm_fold("trail, text -> trail, count: int", emit="update", lm=trail_lm())
    batch = contents(fold(src).collect())
    for seed in range(3):
        assert contents(fold(tp.replay(src, seed=seed, **faults)).collect()) == batch


def test_fold_emits_provisional_rows_only_when_live(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    fold = lambda s: steps_of(s).group_by("trial").lm_fold("trail, text -> trail, count: int", emit="update", lm=trail_lm())

    batch_rows = [e.row for e in collect_events(fold(src)) if not isinstance(e, Closed)]
    assert all(r.final for r in batch_rows) and len(batch_rows) == 3

    live_rows = [e.row for e in collect_events(fold(tp.replay(src))) if not isinstance(e, Closed)]
    alpha = [r for r in live_rows if r.trial == "alpha__1"]
    assert [(r.trail, r.final, r.version) for r in alpha] == [
        ("|1", False, 1), ("|1|2", False, 2), ("|1|2|3", False, 3), ("|1|2|3", True, 4),
    ]


def test_fold_rewinds_when_a_child_arrives_late():
    """Step 2 arrives after step 3: the fold recomputes from step 2 onward."""
    rows = [Row((("trial", "t"), ("step", i)), {"text": str(i)}) for i in (1, 3, 2)]

    class Unsorted(tp.adapters.base.RowsAdapter):
        async def events(self, table):
            for row in rows:  # deliberately out of order, and no closes until the end
                yield tp.RowAdded(row)
                await asyncio.sleep(0.05)  # let the fold catch up before the next row

    src = Unsorted(rows, tp.Schema(("trial", "step"), {"text": str}))
    lm = trail_lm()
    (row,) = src.table("rows").group_by("trial").lm_fold("trail, text -> trail, count: int", lm=lm).collect()
    assert row.trail == "|1|2|3"
    # Folded 1, then 3; step 2 arrived late, so 2 and 3 were redone.
    assert lm.calls == 4


def test_every_n_batches_provisional_updates(harbor_dir):
    src = tp.replay(tp.read.harbor(harbor_dir))
    fold = steps_of(src).group_by("trial").lm_fold("trail, text -> trail, count: int", emit=tp.every(2), lm=trail_lm())
    alpha = [e.row for e in collect_events(fold) if not isinstance(e, Closed) and e.row.trial == "alpha__1"]
    assert [(r.trail, r.final) for r in alpha] == [("|1|2", False), ("|1|2|3", True)]


# --- lm_reduce emit="update" -------------------------------------------------------------


def update_reduce(src, emit="update", lm=None):
    return steps_of(src).group_by("trial").lm_reduce("text -> summary", emit=emit, lm=lm or count_lm())


def test_update_reduce_is_a_plain_reduce_in_batch(harbor_dir):
    lm = count_lm()
    rows = update_reduce(tp.read.harbor(harbor_dir), lm=lm).collect()
    assert [r.summary for r in rows] == ["full:3"] * 3
    assert lm.calls == 3  # no running updates: one full reduce per trial


def test_update_reduce_streams_provisional_then_full(harbor_dir):
    lm = count_lm()
    events = collect_events(update_reduce(tp.replay(tp.read.harbor(harbor_dir)), lm=lm))
    alpha = [e.row for e in events if not isinstance(e, Closed) and e.row.trial == "alpha__1"]
    assert [(r.summary, r.final) for r in alpha] == [
        ("partial+1", False), ("partial+1", False), ("partial+1", False), ("full:3", True),
    ]
    assert lm.calls == 3 * 3 + 3  # a running update per step, then a full reduce per trial


def test_update_reduce_every_n(harbor_dir):
    events = collect_events(update_reduce(tp.replay(tp.read.harbor(harbor_dir)), emit=tp.every(2)))
    alpha = [e.row for e in events if not isinstance(e, Closed) and e.row.trial == "alpha__1"]
    assert [(r.summary, r.final) for r in alpha] == [("partial+2", False), ("full:3", True)]


@pytest.mark.parametrize("faults", FAULTS)
def test_update_reduce_replay_matches_batch(harbor_dir, faults):
    src = tp.read.harbor(harbor_dir)
    batch = contents(update_reduce(src).collect())
    for seed in range(3):
        assert contents(update_reduce(tp.replay(src, seed=seed, **faults)).collect()) == batch


def test_emit_validation(harbor_dir):
    steps = steps_of(tp.read.harbor(harbor_dir)).group_by("trial")
    with pytest.raises(ValueError, match="emit must be"):
        steps.lm_reduce("text -> summary", emit="sometimes")
    with pytest.raises(NotImplementedError, match="time-based"):
        tp.every("30s")
    with pytest.raises(TypeError, match="string signature"):
        steps.lm_fold(object())


# --- Downstream of provisional rows ------------------------------------------------------


def test_downstream_reduce_uses_final_rows(harbor_dir):
    """A task-level reduce over provisional trial rows still sees only the final versions."""
    src = tp.replay(tp.read.harbor(harbor_dir), shuffle_within=4, seed=1)
    per_trial = steps_of(src).group_by("trial").lm_fold("trail, text -> trail, count: int", emit="update", lm=trail_lm())
    per_task = per_trial.group_by("task").reduce(lambda rows: {"trails": [r.trail for r in rows]})
    assert [r.trails for r in per_task.collect()] == [["|1|2|3", "|1|2|3"], ["|1|2|3"]]


def test_duplicates_do_not_cost_extra_lm_calls(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    plain, dup = trail_lm(), trail_lm()
    steps_of(tp.replay(src)).lm_map("text -> trail", lm=plain).collect()
    steps_of(tp.replay(src, duplicates=1.0)).lm_map("text -> trail", lm=dup).collect()
    assert dup.calls == plain.calls == 9
