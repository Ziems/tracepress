import pandas as pd
import dspy
import pytest
from dspy.utils import DummyLM

import tracepress as tp
from tracepress.rows import Row


def rows_table():
    rows = [
        Row((("task", t), ("trial", f"{t}{i}")), {"summary": f"{t} trial {i}", "ok": i % 2 == 0, "agent": a})
        for t in ("a", "b")
        for i, a in enumerate(("x", "y", "x"))
    ]
    return tp.from_rows(rows)


# --- Plain operators ---------------------------------------------------------------------


def test_map_filter_select():
    table = rows_table().map(lambda r: {"n": len(r.summary)}).filter(lambda r: r.ok).select("summary")
    rows = table.collect()
    assert [r.trial for r in rows] == ["a0", "a2", "b0", "b2"]
    assert list(rows[0].values) == ["summary"]


def test_group_by_level_and_column():
    by_task_agent = rows_table().group_by("task", "agent").reduce(lambda rs: {"n": len(rs)})
    rows = by_task_agent.collect()
    assert [(r.task, r.agent, r.n) for r in rows] == [("a", "x", 2), ("a", "y", 1), ("b", "x", 2), ("b", "y", 1)]


def test_global_group_by_column():
    rows = rows_table().group_by("agent").reduce(lambda rs: {"n": len(rs)}).collect()
    assert [(r.agent, r.n) for r in rows] == [("x", 4), ("y", 2)]


def test_explode():
    table = tp.from_rows([Row((("doc", 1),), {"words": ["a", "b"]})]).explode("words", keep=())
    rows = table.collect()
    assert [(r.key, r.word) for r in rows] == [((("doc", 1), ("word", 0)), "a"), ((("doc", 1), ("word", 1)), "b")]


def test_shared_upstream_runs_once():
    calls = []
    base = rows_table().map(lambda r: calls.append(r.key) or {})
    tp.collect(base.filter(lambda r: r.ok), base.group_by("task").reduce(lambda rs: {"n": len(rs)}))
    assert len(calls) == 6


def test_from_pandas_roundtrip():
    df = rows_table().to_pandas()
    assert list(df.columns[:2]) == ["task", "trial"]
    again = tp.from_pandas(df, levels=["task", "trial"]).group_by("task").reduce(lambda rs: {"n": len(rs)})
    assert [r.n for r in again.collect()] == [3, 3]


# --- Build-time validation ---------------------------------------------------------------


def test_unknown_input_fails_at_build_time():
    with pytest.raises(ValueError, match="summry"):
        rows_table().lm_map("summry -> topic")


def test_output_clash_fails():
    with pytest.raises(ValueError, match="already exist"):
        rows_table().lm_map("summary -> ok: bool")


def test_lm_filter_needs_one_bool():
    with pytest.raises(ValueError, match="one bool"):
        rows_table().lm_filter("summary -> reason")


def test_lm_reduce_unknown_column():
    with pytest.raises(ValueError, match="nope"):
        rows_table().group_by("task").lm_reduce("nope -> x")


# --- LM operators (DummyLM) --------------------------------------------------------------


def test_lm_map():
    lm = DummyLM([{"topic": "testing"}] * 6)
    rows = rows_table().lm_map("summary -> topic", lm=lm).collect()
    assert [r.topic for r in rows] == ["testing"] * 6
    assert tp.last_run().calls == 6


def test_lm_filter():
    lm = DummyLM({"a trial 1": {"keep": True}, "trial": {"keep": False}})
    rows = rows_table().lm_filter("summary -> keep: bool", lm=lm).collect()
    assert [r.trial for r in rows] == ["a1"]


def test_lm_extract():
    lm = DummyLM([{"failures": ["timeout", "wrong file"]}] * 6)
    rows = rows_table().lm_extract("summary -> failures: list[str]", keep=("agent",), lm=lm).collect()
    assert len(rows) == 12
    assert rows[0].key[-1] == ("failure", 0)
    assert rows[0].failure == "timeout" and rows[0].agent == "x"


def test_lm_reduce_binds_keys_and_aligned_children():
    lm = DummyLM([{"common": "they all ran"}] * 2)
    table = rows_table().group_by("task").lm_reduce("task, summary, ok -> common", lm=lm)
    rows = table.collect()
    assert [(r.task, r.common) for r in rows] == [("a", "they all ran"), ("b", "they all ran")]
    prompt = lm.history[0]["messages"][-1]["content"]
    # Each child is one record holding its own summary and ok.
    assert '"summary": "' in prompt and '"ok": ' in prompt
    assert "[[ ## trials ## ]]" in prompt


def test_lm_reduce_tree_merges_large_groups():
    lm = DummyLM([{"common": "partial"}] * 50)
    table = rows_table().group_by("task").lm_reduce("summary -> common", lm=lm, chunk_chars=40)
    rows = table.collect()
    assert [r.common for r in rows] == ["partial", "partial"]
    # Each task: one reduce call per chunk, then merge calls.
    assert tp.last_run().calls > 2
    assert any("partials" in h["messages"][-1]["content"] for h in lm.history)


def test_lm_errors_are_skipped_by_default():
    class Broken(DummyLM):
        def forward(self, *args, **kwargs):
            raise RuntimeError("provider down")

        async def aforward(self, *args, **kwargs):
            raise RuntimeError("provider down")

    lm = Broken([])
    assert rows_table().lm_map("summary -> topic", lm=lm).collect() == []
    assert tp.last_run().errors == 6
    with pytest.raises(RuntimeError):
        rows_table().lm_map("summary -> topic", lm=lm, on_error="raise").collect()


def test_collect_rejects_running_loop():
    import asyncio

    async def inner():
        with pytest.raises(RuntimeError, match="acollect"):
            rows_table().collect()
        return await rows_table().acollect()

    assert len(asyncio.run(inner())) == 6


def test_pandas_columns():
    df = rows_table().to_pandas()
    assert isinstance(df, pd.DataFrame) and len(df) == 6


def test_unparseable_response_falls_back_to_next_adapter():
    import dspy

    # DummyLM answers in the chat format, which the XML adapter can't parse.
    tp.configure(adapter=dspy.XMLAdapter(), fallback_adapters=[dspy.ChatAdapter()])
    try:
        rows = rows_table().lm_map("summary -> topic", lm=DummyLM([{"topic": "t"}] * 30)).collect()
    finally:
        tp.configure(fallback_adapters=[dspy.XMLAdapter()])
    assert [r.topic for r in rows] == ["t"] * 6
    assert tp.last_run().fallbacks == 6 and tp.last_run().errors == 0


def test_hung_calls_time_out_and_retry():
    from fakes import HangingLM

    tp.configure(request_timeout=0.1, timeout_retries=2)
    try:
        lm = HangingLM(lambda f: {"topic": "t"}, hangs=1)
        rows = rows_table().lm_map("summary -> topic", lm=lm).collect(max_concurrency=1)
        assert [r.topic for r in rows] == ["t"] * 6
        assert tp.last_run().timeouts == 1 and tp.last_run().errors == 0

        stuck = HangingLM(lambda f: {"topic": "t"}, hangs=100)
        assert rows_table().lm_map("summary -> topic", lm=stuck).collect() == []
        assert tp.last_run().errors == 6 and tp.last_run().timeouts == 6 * 3
    finally:
        tp.configure(request_timeout=30.0, timeout_retries=2)


def test_call_log_records_latency():
    rows_table().lm_map("summary -> topic", lm=DummyLM([{"topic": "t"}] * 6)).collect()
    stats = tp.last_run()
    assert len(stats.call_log) == 6 and all(ok for *_, ok in stats.call_log)
    assert set(stats.latency()) == {"p50", "p90", "p99", "max"}


def test_map_rejects_columns_shadowing_key_levels():
    with pytest.raises(ValueError, match="shadow"):
        rows_table().map(lambda r: {"trial": "x"}, columns={"trial": str})


def test_billing_errors_stop_the_run_instead_of_skipping_rows():
    class OutOfCredits(DummyLM):
        calls = 0

        async def aforward(self, *args, **kwargs):
            OutOfCredits.calls += 1
            raise dspy.LMBillingError("exceeds your available credits (402)", model="m")

    rows = [Row((("trial", i),), {"summary": f"trial {i}"}) for i in range(200)]
    with pytest.raises(dspy.LMBillingError):
        tp.from_rows(rows).lm_map("summary -> topic", lm=OutOfCredits([])).collect(max_concurrency=4)
    # Only the calls already in flight ran; the other rows were never tried.
    assert OutOfCredits.calls <= 8
