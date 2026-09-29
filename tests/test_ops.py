from __future__ import annotations

import re

import pandas as pd
import pytest

import tracepress

from .conftest import field_value, output_fields, prompt_of


def test_map_writes_outputs_as_columns(make_lm):
    lm, engine = make_lm(lambda r: {"loud": field_value(r, "text").upper()})
    df = pd.DataFrame({"text": ["a", "b", "c"], "other": [1, 2, 3]})
    out = df.lm_map("text -> loud", "Shout it", lm=lm)
    assert list(out["loud"]) == ["A", "B", "C"]
    assert list(df.columns) == ["text", "other"]  # input untouched

    request = engine.requests[0]
    assert "Shout it" in request.system
    assert "[[ ## text ## ]]" in prompt_of(request) and "other" not in prompt_of(request)


def test_map_typed_outputs(make_lm):
    def fn(r):
        n = int(field_value(r, "n"))
        return {"even": n % 2 == 0, "label": "big" if n > 1 else "small"}

    lm, engine = make_lm(fn)
    out = pd.DataFrame({"n": [1, 2]}).lm_map("n -> even: bool, label: Literal['small', 'big']", "Describe", lm=lm)
    assert out["even"].tolist() == [False, True] and out["label"].tolist() == ["small", "big"]
    assert output_fields(engine.requests[0]) == ["even", "label"]


def test_map_class_signature(make_lm):
    class Shout(tracepress.Signature):
        """Shout the text."""

        text: str = tracepress.In()
        loud: str = tracepress.Out(desc="uppercase")

    lm, engine = make_lm(lambda r: {"loud": "A"})
    out = pd.DataFrame({"text": ["a"]}).lm_map(Shout, lm=lm)
    assert out["loud"].tolist() == ["A"]
    assert "Shout the text." in engine.requests[0].system


def test_map_examples_become_demos(make_lm):
    lm, engine = make_lm(lambda r: {"o": "x"})
    pd.DataFrame({"t": ["real"]}).lm_map("t -> o", "Do it", examples=[{"t": "demo", "o": "DEMO"}], lm=lm)
    messages = engine.requests[0].messages
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert "demo" in messages[0].parts[0].text and "DEMO" in messages[1].parts[0].text


def test_map_missing_input_column(make_lm):
    lm, engine = make_lm(lambda r: "{}")
    with pytest.raises(KeyError, match="nope"):
        pd.DataFrame({"a": [1]}).lm_map("nope -> x", lm=lm)
    assert engine.requests == []


def test_filter_keeps_matching_rows_and_index(make_lm):
    lm, engine = make_lm(lambda r: {"keep": int(field_value(r, "n")) % 2 == 0})
    df = pd.DataFrame({"n": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
    out = df.lm_filter("n", "The number is even", lm=lm)
    assert list(out["n"]) == [2, 4] and list(out.index) == [11, 13]
    assert "Condition: The number is even" in engine.requests[0].system


def test_filter_explain(make_lm):
    lm, engine = make_lm(lambda r: {"explanation": "because", "keep": True})
    out = pd.DataFrame({"n": [1]}).lm_filter("n", "Positive", explain=True, lm=lm)
    assert list(out["explanation"]) == ["because"]
    assert output_fields(engine.requests[0]) == ["explanation", "keep"]


def test_filter_rejects_outputs(make_lm):
    lm, _ = make_lm(lambda r: "{}")
    with pytest.raises(ValueError, match="only reads rows"):
        pd.DataFrame({"n": [1]}).lm_filter("n -> keep", "x", lm=lm)


def summing_agg(request):
    """Answers an aggregation by adding up the `v` values (or the partial answers) it sees."""
    records = field_value(request, "records")
    if "[Partial answer" in records:
        values = re.findall(r"\[Partial answer \d+\]\n(-?\d+)", records)
    else:
        values = re.findall(r"^\w+: (-?\d+)$", records, re.M)
    total = str(sum(int(v) for v in values))
    return {output_fields(request)[0]: total}


def test_agg_groups(make_lm):
    lm, _ = make_lm(summing_agg)
    df = pd.DataFrame({"g": ["a", "b", "a", "b", "a"], "v": [1, 2, 3, 4, 5]})
    out = df.lm_agg("v -> total", "Add them up", by="g", lm=lm)
    assert out.to_dict("records") == [
        {"g": "a", "total": "9", "n_rows": 3},
        {"g": "b", "total": "6", "n_rows": 2},
    ]


def test_agg_is_hierarchical_when_rows_dont_fit(make_lm):
    lm, engine = make_lm(summing_agg)
    df = pd.DataFrame({"v": list(range(1, 41))})
    out = df.lm_agg("v -> total", "Add them up", max_chars=30, lm=lm)
    assert out["total"].item() == str(sum(range(1, 41)))
    scopes = [field_value(r, "scope") for r in engine.requests]
    assert any("part 1 of" in s for s in scopes)
    assert any("partial answers" in s for s in scopes)
    # Only the last call writes the signature's outputs; the rest write partial answers.
    assert sum(1 for r in engine.requests if output_fields(r) == ["total"]) == 1


def test_agg_chains(make_lm):
    lm, _ = make_lm(summing_agg)
    df = pd.DataFrame({"task": ["t1", "t1", "t2"], "run": ["r1", "r2", "r3"], "v": [1, 2, 3]})
    per_run = df.lm_agg("v -> run_total", "Add", by=["task", "run"], lm=lm)
    per_task = per_run.lm_agg("run_total -> task_total", "Add", by="task", lm=lm)
    overall = per_task.lm_agg("task_total -> total", "Add", lm=lm)
    assert list(per_task["task_total"]) == ["3", "3"] and overall["total"].item() == "6"


def test_agg_needs_an_instruction_and_real_columns(make_lm):
    lm, _ = make_lm(summing_agg)
    with pytest.raises(ValueError, match="instruction"):
        pd.DataFrame({"v": [1]}).lm_agg("v -> total", lm=lm)
    with pytest.raises(KeyError):
        pd.DataFrame({"v": [1]}).lm_agg("v -> total", "Add", by="nope", lm=lm)


def rank_by_value(request):
    """Ranks records by their `v`, highest first."""
    values = [int(v) for v in re.findall(r"^v: (-?\d+)$", field_value(request, "records"), re.M)]
    return {"ranking": sorted(range(1, len(values) + 1), key=lambda i: -values[i - 1])}


def test_topk_across_rounds(make_lm):
    lm, engine = make_lm(rank_by_value)
    df = pd.DataFrame({"v": [5, 17, 3, 42, 8, 23, 1, 16, 4, 30, 11, 2]})
    out = df.lm_topk("v", "Highest value", k=3, batch_size=4, lm=lm)
    assert list(out["v"]) == [42, 30, 23] and list(out["rank"]) == [1, 2, 3]
    assert len(engine.requests) > 3  # more than one round
    assert field_value(engine.requests[0], "criterion") == "Highest value"


def test_topk_repairs_bad_rankings(make_lm):
    lm, _ = make_lm(lambda r: {"ranking": [2, 2, 99]})
    out = pd.DataFrame({"v": [1, 2, 3]}).lm_topk("v", "Best", 3, lm=lm)
    assert list(out["v"]) == [2, 1, 3]


def cluster_by_topic(request):
    """Clusters records with the same topic."""
    topics = re.findall(r"^topic: (\w+)$", field_value(request, "records"), re.M)
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(topics, 1):
        groups.setdefault(t, []).append(i)
    return {"clusters": [g for g in groups.values() if len(g) > 1]}


def test_dedup_merges_across_batches(make_lm):
    lm, _ = make_lm(cluster_by_topic)
    topics = ["cat", "dog", "cow", "pig", "cat", "dog", "hen", "cat"]
    df = pd.DataFrame({"topic": topics}, index=[f"r{i}" for i in range(len(topics))])
    out = df.lm_dedup("topic", "Same animal", batch_size=3, passes=4, seed=1, lm=lm)
    assert sorted(out["topic"]) == ["cat", "cow", "dog", "hen", "pig"]
    cat = out[out["topic"] == "cat"].iloc[0]
    assert cat.name == "r0" and cat["n_duplicates"] == 3 and sorted(cat["duplicate_ids"]) == ["r4", "r7"]


def test_join_is_a_todo():
    with pytest.raises(NotImplementedError):
        pd.DataFrame().lm_join(pd.DataFrame(), "a", "x")


def test_signature_is_recorded_for_the_agent(make_lm, tmp_path):
    lm, _ = make_lm(lambda r: {"label": "a"})
    ws = tracepress.Workspace(tmp_path / "ws")

    @ws.pipeline
    def label(rows):
        return rows.lm_map("t -> label: Literal['a', 'b']", "Label it", lm=lm, save_as="labeled")

    label(pd.DataFrame({"t": ["x"]}))
    rec = ws.resolve("labeled")
    assert rec["instruction"] == "Label it"
    assert rec["params"]["signature"] == "t -> label: Literal['a', 'b']"
    assert rec["params"]["json_schema"]["properties"]["label"]["enum"] == ["a", "b"]
    assert rec["usage"]["calls"] == 1
