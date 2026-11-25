import asyncio
import time

import pytest
from dspy.utils import DummyLM

import tracepress as tp
from tracepress.adapters.base import RowsAdapter
from tracepress.rows import Closed, Row, RowAdded, is_prefix
from tracepress.schema import Schema

TRIALS = ("alpha__1", "alpha__2", "beta__1")  # beta__2 crashed: no steps


def replay_events(adapter, table):
    async def run():
        return [e async for e in adapter.events(table)]

    return asyncio.run(run())


def assert_closes_follow_rows(events):
    closed = []
    for event in events:
        if isinstance(event, Closed):
            closed.append(event.prefix)
        else:
            late = [p for p in closed if is_prefix(p, event.row.key)]
            assert not late, f"{event.row.key} arrived after Closed({late[0]})"


def test_replay_interleaves_trials_by_time(harbor_dir):
    events = replay_events(tp.replay(tp.read.harbor(harbor_dir)), "steps")
    rows = [e.row for e in events if isinstance(e, RowAdded)]
    # Every trial's step 1 happened at the same time, so they all come before any step 2.
    assert [(r.trial, r.step) for r in rows[:3]] == [(t, 1) for t in TRIALS]
    assert_closes_follow_rows(events)
    # Every close from the batch source is still sent, including the crashed trial's.
    assert sum(isinstance(e, Closed) for e in events) == 4 + 2 + 1


@pytest.mark.parametrize("faults", [
    {"shuffle_within": 4},
    {"drop_closes": 0.5},
    {"duplicates": 0.5},
    {"shuffle_within": 4, "drop_closes": 0.3, "duplicates": 0.3},
])
def test_faults_keep_closes_after_rows(harbor_dir, faults):
    for seed in range(5):
        events = replay_events(tp.replay(tp.read.harbor(harbor_dir), seed=seed, **faults), "steps")
        assert_closes_follow_rows(events)


def test_duplicates_and_drops_are_injected(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    plain = replay_events(tp.replay(src), "steps")
    dup = replay_events(tp.replay(src, duplicates=1.0), "steps")
    dropped = replay_events(tp.replay(src, drop_closes=1.0), "steps")
    n_rows = sum(isinstance(e, RowAdded) for e in plain)
    assert sum(isinstance(e, RowAdded) for e in dup) == 2 * n_rows
    assert not any(isinstance(e, Closed) for e in dropped)


def test_speed_follows_timestamps(harbor_dir):
    # Steps are one second apart (0:01 to 0:03); at 10x that's at least 0.2s.
    start = time.monotonic()
    tp.replay(tp.read.harbor(harbor_dir), speed=10).table("steps").collect()
    assert time.monotonic() - start >= 0.18


def test_tables_of_one_replay_share_its_clock(harbor_dir):
    """A trial's row (placed when it finished, at 0:01:00) comes after its steps (0:01 to 0:03)."""
    live = tp.replay(tp.read.harbor(harbor_dir), speed=200)
    arrived = []

    async def watch(table):
        async for event in live.events(table):
            if isinstance(event, RowAdded):
                arrived.append(table)

    async def run():
        await asyncio.gather(watch("trials"), watch("steps"))

    asyncio.run(run())
    assert arrived == ["steps"] * 9 + ["trials"] * 4


def test_untimed_rows_before_the_first_timed_one_take_its_time():
    rows = [Row((("trajectory", "t"), ("step", 1)), {"timestamp": None}),
            Row((("trajectory", "t"), ("step", 2)), {"timestamp": "2026-01-01T00:00:00Z"})]
    src = RowsAdapter(rows, Schema(("trajectory", "step"), {"timestamp": str}))
    times = [t for t, _ in asyncio.run(tp.replay(src).timeline("rows"))]
    assert times[0] == times[1]


def test_a_close_with_no_rows_comes_before_its_parents_close(harbor_dir):
    """beta__2 crashed with no steps; task beta ran before alpha, so its close must still follow beta__2's."""
    trials = harbor_dir / "trials"
    for d in trials.glob("alpha__*/agent/trajectory.json"):
        d.write_text(d.read_text().replace("2026-01-01T", "2026-01-02T"))
    events = replay_events(tp.replay(tp.read.harbor(harbor_dir)), "steps")
    closes = [dict(e.prefix) for e in events if isinstance(e, Closed)]
    crashed = next(i for i, c in enumerate(closes) if c.get("trial") == "beta__2")
    task = next(i for i, c in enumerate(closes) if c.get("task") == "bench/beta" and "trial" not in c)
    assert crashed < task


def test_replay_validates_arguments(harbor_dir):
    src = tp.read.harbor(harbor_dir)
    with pytest.raises(ValueError, match="bounded"):
        tp.replay(tp.replay(src))
    with pytest.raises(ValueError, match="between 0 and 1"):
        tp.replay(src, duplicates=2)


# --- Batch vs stream equivalence ---------------------------------------------------------


def make_lm():
    """Answers depend on the input, so a misrouted row or group changes the output."""
    answers = {}
    for trial in TRIALS:
        answers[f"note-{trial}"] = {"summary": f"summary of {trial}"}  # reduce prompts (checked first)
        answers[f"{trial}:"] = {"note": f"note-{trial}"}  # step prompts
    return DummyLM(answers)


def pipeline(src):
    steps = src.table("steps").map(lambda r: {"text": f"{r.trial}: {r.message}"}, columns={"text": str})
    notes = steps.lm_map("text -> note", lm=make_lm())
    per_trial = notes.group_by("trial").lm_reduce("note -> summary", lm=make_lm())
    per_task = per_trial.group_by("task").reduce(
        lambda rows: {"trials": [r.trial for r in rows], "summaries": [r.summary for r in rows]}
    )
    step_counts = steps.group_by("trial").reduce(lambda rows: {"steps": [r.step for r in rows]})
    return per_trial, per_task, step_counts


@pytest.mark.parametrize("faults", [
    {},
    {"shuffle_within": 4},
    {"drop_closes": 0.5},
    {"duplicates": 0.5},
    {"shuffle_within": 4, "drop_closes": 0.3, "duplicates": 0.3},
])
def test_replay_matches_batch(harbor_dir, faults):
    src = tp.read.harbor(harbor_dir)
    batch = tp.collect(*pipeline(src))
    for seed in range(3):
        stream = tp.collect(*pipeline(tp.replay(src, seed=seed, **faults)))
        assert stream == batch

    per_trial, per_task, step_counts = batch
    assert [r.summary for r in per_trial] == [f"summary of {t}" for t in TRIALS]
    assert [r.trials for r in per_task] == [["alpha__1", "alpha__2"], ["beta__1"]]
    assert all(r.steps == [1, 2, 3] for r in step_counts)
