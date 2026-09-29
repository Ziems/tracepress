from __future__ import annotations

import json

import pandas as pd
import pytest

import tracepress

from .conftest import ScriptedEngine, tool_call
from .test_ops import summing_agg


def agent_model(script):
    engine = ScriptedEngine(script)
    return tracepress.Model("fake/agent", engine=engine, cache=False), engine


@pytest.fixture
def ws(tmp_path, make_lm):
    lm, client = make_lm(summing_agg)
    tracepress.configure(lm=lm)
    ws = tracepress.Workspace(tmp_path / "ws")
    ws.save("runs", pd.DataFrame({"task": ["t1", "t1", "t2"], "v": [1, 2, 3]}))

    @ws.pipeline(description="Sums v per task")
    def per_task(rows):
        return rows.lm_agg("v -> total", "Add them up", by="task", save_as="per_task")

    ws.op_client = client
    return ws


def test_agent_reads_computes_and_runs_ops(ws):
    model, agent = agent_model(
        [
            tool_call("describe_artifact", name="runs"),
            tool_call("python", code="df = load('runs')\nlen(df[df.task == 't1'])"),
            tool_call("run_op", op="agg", artifact="runs", signature="v -> total", instruction="Add them up", by=["task"], save_as="sums"),
            tool_call("run_pipeline", pipeline="per_task", inputs={"rows": "runs"}, where={"rows": "task == 't2'"}),
            "t1 has 2 runs (`runs` rows 0, 1).",
        ]
    )
    answer = ws.agent_run("How many runs does t1 have?", model=model, verbose=False)

    assert answer.text.startswith("t1 has 2 runs")
    outputs = [s["output"] for s in answer.steps if s["type"] == "tool"]
    assert '"value_counts"' in outputs[0]
    assert outputs[1].strip() == "2"
    assert "Saved as artifact" in outputs[2] and '"total": "3"' in outputs[2]
    assert json.loads(outputs[3])["status"] == "ok"
    assert ws["per_task"]["task"].tolist() == ["t2"]

    # The ad-hoc op was saved under the agent run, with lineage to its input.
    sums = ws.resolve("sums")
    assert sums["pipeline"] == "agent_run" and sums["parents"] == [ws.resolve("runs")["ref"]]
    assert answer.op_calls == 3  # 2 groups for run_op + 1 for the pipeline

    # The agent saw the workspace overview up front, and each tool result went back.
    first = agent.requests[0]
    system = first.system[0].text if isinstance(first.system, tuple) else first.system
    assert "per_task" in system and '"name": "runs"' in system
    assert len(agent.requests[-1].messages) == 9
    assert json.loads(open(answer.transcript_path).read())["answer"] == answer.text


def test_budget_refusals_go_back_to_the_agent(ws):
    model, _ = agent_model(
        [
            tool_call("run_op", op="agg", artifact="runs", signature="v -> total", instruction="Add them up", by=["task"]),
            "Could not run it.",
        ]
    )
    answer = ws.agent_run("q", model=model, max_calls=1, verbose=False)
    step = [s for s in answer.steps if s["type"] == "tool"][0]
    assert step["error"] and "Refused by the budget" in step["output"]
    assert ws.op_client.requests == []


def test_tool_errors_are_reported_not_raised(ws):
    model, _ = agent_model([tool_call("read_rows", name="nope"), tool_call("python", code="1/0"), "done"])
    answer = ws.agent_run("q", model=model, verbose=False)
    outs = [s["output"] for s in answer.steps if s["type"] == "tool"]
    assert "no artifact 'nope'" in outs[0] and "ZeroDivisionError" in outs[1]


def test_step_limit_forces_an_answer(ws):
    model, agent = agent_model([tool_call("list_artifacts"), "final"])
    answer = ws.agent_run("q", model=model, max_steps=1, verbose=False)
    assert answer.text == "final"
    last = agent.requests[-1]
    assert last.config.tool_choice.mode == "none"
