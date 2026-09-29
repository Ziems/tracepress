from __future__ import annotations

import json

import pandas as pd
import pytest

import tracepress

from .test_ops import summing_agg


@pytest.fixture
def ws(tmp_path, make_lm):
    lm, _ = make_lm(summing_agg)
    tracepress.configure(lm=lm)
    return tracepress.Workspace(tmp_path / "ws")


def data():
    return pd.DataFrame({"task": ["t1", "t1", "t2"], "run": ["r1", "r2", "r3"], "v": [1, 2, 3]})


def test_pipeline_saves_every_step_with_lineage(ws):
    @ws.pipeline(description="sums")
    def sums(rows):
        per_task = rows.lm_agg("v -> total", "Add them up", by="task", save_as="per_task")
        return per_task.lm_agg("total -> total_all", "Add them up", save_as="overall")

    out = sums(data())
    assert out["total_all"].item() == "6"

    arts = ws.artifacts()
    assert list(arts["name"]) == ["rows", "per_task", "overall"]
    assert list(arts["op"]) == ["input", "agg", "agg"]
    assert ws["per_task"]["total"].tolist() == ["3", "3"]

    run = ws.runs().iloc[-1]
    assert run["status"] == "ok" and run["output"] == "overall"
    chain = [a["ref"].split("/")[1] for a in ws.lineage("overall")]
    assert chain == ["overall", "per_task", "rows"]
    manifest = json.loads(next((ws.path / "runs").glob("*/manifest.json")).read_text())
    assert manifest["artifacts"][1]["instruction"] == "Add them up"
    assert manifest["artifacts"][1]["usage"]["calls"] == 2


def test_unnamed_steps_and_repeated_names(ws):
    @ws.pipeline
    def twice(rows):
        a = rows.lm_agg("v -> total", "Add")
        return a.lm_agg("total -> total_all", "Add", save_as="step1_agg")

    twice(data())
    assert list(ws.artifacts()["name"]) == ["rows", "step2_agg", "step1_agg"]


def test_plain_frames_returned_are_saved_as_output(ws):
    @ws.pipeline
    def passthrough(rows):
        return rows[rows["v"] > 1]

    run = ws.run("passthrough", data())
    assert run.manifest["output"] == "output"
    assert len(ws["output"]) == 2
    assert ws.resolve("output")["parents"] == [f"{run.id}/rows"]


def test_failed_run_keeps_partial_artifacts(ws):
    @ws.pipeline
    def broken(rows):
        rows.lm_agg("v -> total", "Add", by="task", save_as="first")
        raise ValueError("boom")

    with pytest.raises(ValueError):
        broken(data())
    run = ws.runs().iloc[-1]
    assert run["status"] == "error" and "boom" in run["error"]
    assert len(ws["first"]) == 2


def test_latest_version_wins_and_refs_address_old_ones(ws):
    ws.save("raw", data())
    first = ws.resolve("raw")["ref"]
    ws.save("raw", data().head(1))
    assert len(ws["raw"]) == 1
    assert len(ws[first]) == 3
    assert len(ws.artifacts(latest=True)) == 1


def test_nested_values_survive_parquet(ws):
    df = pd.DataFrame({"mixed": [{"a": 1}, "text", [1, 2]]})
    ws.save("odd", df)
    assert list(ws["odd"]["mixed"]) == ['{"a": 1}', "text", "[1, 2]"]
