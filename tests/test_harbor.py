"""Load Harbor jobs and classify them with the Terminal-Bench taxonomies."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import tracepress
from tracepress.harbor.load import STEP_COLUMNS, TRIAL_COLUMNS

from .conftest import field_value, output_fields


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _job(root: Path) -> Path:
    job = root / "job"
    _write(job / "result.json", {"trial_name": "JOB", "task_name": "not-a-trial"})
    observation = "START" + ("x" * 6000) + "TAIL_MARKER"
    _write(
        job / "trial-a" / "result.json",
        {
            "trial_name": "trial-a",
            "task_name": "task-a",
            "verifier_result": {"rewards": {"partial": 0.5, "reward": 0.0}},
            "exception_info": {"exception_type": "TimeoutError", "exception_message": "timed out"},
        },
    )
    _write(
        job / "trial-a" / "verifier" / "ctrf.json",
        {
            "results": {
                "tests": [
                    {"name": "test_out", "status": "failed", "message": "expected 3"},
                    {"name": "test_ok", "status": "passed", "message": "fine"},
                ]
            }
        },
    )
    _write(
        job / "trial-a" / "agent" / "trajectory.json",
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "sess-a",
            "trajectory_id": "traj-a",
            "agent": {"name": "terminus", "version": "2.0.0", "model_name": "gpt-test"},
            "final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 4, "total_cost_usd": 0.25},
            "steps": [
                {"step_id": 1, "source": "system", "message": "SYSTEM_SECRET"},
                {"step_id": 2, "source": "user", "message": "USER_TASK"},
                {
                    "step_id": 3,
                    "source": "agent",
                    "message": "I will search",
                    "model_name": "gpt-step",
                    "tool_calls": [
                        {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": "rg foo"}}
                    ],
                    "observation": {"results": [{"source_call_id": "c1", "content": observation}]},
                    "metrics": {"prompt_tokens": 8, "completion_tokens": 2, "cost_usd": 0.1},
                },
                {
                    "step_id": 4,
                    "source": "agent",
                    "message": [
                        {"type": "text", "text": "look"},
                        {"type": "image", "source": {"path": "images/a.png"}},
                    ],
                    "is_copied_context": True,
                    "observation": {"results": [{"content": "ok"}]},
                },
            ],
            "subagent_trajectories": [
                {
                    "trajectory_id": "sub-1",
                    "agent": {"name": "sub", "version": "0"},
                    "steps": [
                        {
                            "step_id": 1,
                            "source": "agent",
                            "message": "SUB_STEP",
                            "observation": {"results": [{"content": "exit code 0"}]},
                        }
                    ],
                }
            ],
        },
    )
    _write(
        job / "trial-b" / "trajectory.json",
        {
            "schema_version": "ATIF-v1.6",
            "session_id": "sess-b",
            "agent": {"name": "codex", "version": "1", "model_name": "gpt-b"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "other task"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "done",
                    "tool_calls": [{"tool_call_id": "c2", "function_name": "bash", "arguments": {"command": "true"}}],
                    "observation": {"results": [{"content": "exit code 0"}]},
                },
            ],
        },
    )
    return job


def _bare(root: Path) -> Path:
    path = root / "loose" / "one.json"
    _write(
        path,
        {
            "session_id": "sess-file",
            "trajectory_id": "traj-file",
            "agent": {"name": "claude-code", "version": "1"},
            "steps": [{"step_id": 1, "source": "user", "message": "hello"}],
        },
    )
    return path


def test_load_job_trial_metadata_steps_and_ignores_job_result(tmp_path):
    loaded = tracepress.harbor_load(_job(tmp_path))
    assert list(loaded.trials["trial_key"]) == ["trial-a", "trial-b"]
    assert "JOB" not in set(loaded.trials["trial_name"])
    assert list(loaded.trials.columns) == TRIAL_COLUMNS
    assert list(loaded.steps.columns) == STEP_COLUMNS

    trial = loaded.trials.set_index("trial_key").loc["trial-a"]
    assert trial["trial_name"] == "trial-a"
    assert trial["task_name"] == "task-a"
    assert trial["session_id"] == "sess-a"
    assert trial["agent_name"] == "terminus"
    assert trial["model_name"] == "gpt-test"
    assert trial["reward"] == 0.0
    assert json.loads(trial["rewards"]) == {"partial": 0.5, "reward": 0.0}
    assert trial["exception"] == "TimeoutError: timed out"
    assert trial["failed_tests"] == "test_out: expected 3"
    assert trial["n_steps"] == 5
    assert trial["total_prompt_tokens"] == 10
    assert trial["total_cost_usd"] == 0.25
    assert trial["source_path"].endswith("trial-a")

    steps = loaded.steps[loaded.steps["trial_key"] == "trial-a"]
    assert steps["source"].tolist() == ["system", "user", "agent", "agent", "agent"]
    assert steps["is_subagent"].tolist() == [False, False, False, False, True]
    assert steps["message"].iloc[3] == "look\n[image]"
    assert bool(steps["is_copied_context"].iloc[3]) is True
    assert steps["tool_calls"].iloc[2] == 'bash({"command":"rg foo"})'
    assert steps["observation"].iloc[2].endswith("TAIL_MARKER")
    assert steps["observation"].iloc[2].startswith("…[truncated]\n")
    assert "START" not in steps["observation"].iloc[2]
    assert steps["step_model"].iloc[2] == "gpt-step"
    assert steps["prompt_tokens"].iloc[2] == 8
    assert steps["message"].iloc[4] == "SUB_STEP"
    assert steps["session_id"].iloc[4] == "sess-a"
    assert steps["failed_tests"].iloc[0] == "test_out: expected 3"

    other = loaded.trials.set_index("trial_key").loc["trial-b"]
    assert other["trial_name"] == ""
    assert other["task_name"] == ""
    assert pd.isna(other["reward"])
    assert other["failed_tests"] == ""
    assert other["n_steps"] == 2
    assert other["agent_name"] == "codex"


def test_load_bare_file_uses_session_id_and_list_prefixes_keys(tmp_path):
    job = _job(tmp_path)
    bare = _bare(tmp_path)
    alone = tracepress.harbor_load(bare)
    assert alone.trials["trial_key"].tolist() == ["sess-file"]
    assert alone.steps["message"].tolist() == ["hello"]
    assert alone.trials["task_name"].tolist() == [""]

    both = tracepress.harbor_load([job, bare])
    assert set(both.trials["trial_key"]) == {"job/trial-a", "job/trial-b", "one.json/sess-file"}


def test_load_trial_directory_and_errors(tmp_path):
    job = _job(tmp_path)
    trial = tracepress.harbor_load(job / "trial-b")
    assert trial.trials["trial_key"].tolist() == ["trial-b"]
    assert len(trial.steps) == 2

    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="harbor job download"):
        tracepress.harbor_load(missing)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="not a Harbor"):
        tracepress.harbor_load(empty)

    broken = tmp_path / "broken.json"
    broken.write_text("{")
    with pytest.raises(ValueError, match="not valid JSON"):
        tracepress.harbor_load(broken)

    not_atif = tmp_path / "note.json"
    not_atif.write_text(json.dumps({"agent": {"name": "x", "version": "1"}}))
    with pytest.raises(ValueError, match="missing steps"):
        tracepress.harbor_load(not_atif)


def _summarize(request):
    step = field_value(request, "step")
    assert "SYSTEM_SECRET" not in step
    assert "SYSTEM_SECRET" not in field_value(request, "instruction")
    return {"step_summary": step[:80] or "empty step"}


def _tat(request):
    records = field_value(request, "records")
    assert "SYSTEM_SECRET" not in records
    modes = ["premature_termination", "disobey_specification"] if "task-a" in records else []
    return {"explanation": "test_out expected 3", "modes": modes}


def test_harbor_agent_taxonomy_labels_trials_and_skips_system_steps(make_lm, tmp_path):
    lm, engine = make_lm(_route)
    loaded = tracepress.harbor_load(_job(tmp_path))
    out = tracepress.harbor_agent_taxonomy(loaded.steps, lm=lm)
    assert out["trial_key"].tolist() == ["trial-a", "trial-b"]
    by_key = out.set_index("trial_key")
    assert by_key.loc["trial-a", "modes"] == ["premature_termination", "disobey_specification"]
    assert by_key.loc["trial-a", "classes"] == ["verification", "execution"]
    assert by_key.loc["trial-b", "modes"] == []
    assert by_key.loc["trial-b", "classes"] == []
    assert any("disobey_specification" in request.system for request in engine.requests)
    mapped = " ".join(field_value(request, "step") for request in engine.requests)
    instructions = " ".join(field_value(request, "instruction") for request in engine.requests)
    reduced = " ".join(field_value(request, "records") for request in engine.requests)
    assert "USER_TASK" in mapped
    assert "USER_TASK" in instructions
    assert "SUB_STEP" in mapped
    assert "SYSTEM_SECRET" not in mapped
    assert "SYSTEM_SECRET" not in instructions
    assert "task-a" in reduced
    assert "USER_TASK" in reduced
    assert any("step_summary" in output_fields(request) for request in engine.requests)
    assert any("modes" in output_fields(request) for request in engine.requests)


def _keep_failure(request):
    return {"keep": "command not found" in field_value(request, "observation") or "TAIL_MARKER" in field_value(request, "observation")}


def _subcategory(request):
    assert "rg foo" in field_value(request, "tool_calls")
    assert "USER_TASK" in field_value(request, "instruction")
    return {"subcategory": "command_not_found", "explanation": "rg was not on PATH"}


def _route(request):
    fields = output_fields(request)
    if "keep" in fields:
        return _keep_failure(request)
    if "subcategory" in fields:
        return _subcategory(request)
    if "step_summary" in fields:
        return _summarize(request)
    return _tat(request)


def test_command_taxonomy_labels_failed_commands(make_lm, tmp_path):
    lm, engine = make_lm(_route)
    loaded = tracepress.harbor_load(_job(tmp_path))
    out = tracepress.harbor_command_taxonomy(loaded.steps, lm=lm)
    assert len(out) == 1
    assert out["subcategory"].item() == "command_not_found"
    assert out["failure_class"].item() == "invocation"
    assert out["explanation"].item() == "rg was not on PATH"
    assert out["trial_key"].item() == "trial-a"
    assert "command_not_found" in engine.requests[-1].system
    # four agent steps are judged; only the failed one is classified
    assert len(engine.requests) == 5
    assert "USER_TASK" not in " ".join(field_value(request, "observation") for request in engine.requests)


def test_command_taxonomy_empty_has_columns_and_skips_the_map(make_lm):
    lm, engine = make_lm(lambda request: {"keep": False})
    steps = pd.DataFrame(
        {
            "source": ["agent", "user"],
            "message": ["ok", "task"],
            "tool_calls": ["bash({})", ""],
            "observation": ["exit code 0", ""],
        }
    )
    out = tracepress.harbor_command_taxonomy(steps, lm=lm)
    assert len(out) == 0
    assert {"subcategory", "failure_class", "explanation"} <= set(out.columns)
    assert len(engine.requests) == 1

    none, no_engine = make_lm(lambda request: {"keep": True})
    empty = tracepress.harbor_command_taxonomy(steps.iloc[0:0], lm=none)
    assert len(empty) == 0
    assert "subcategory" in empty.columns
    assert no_engine.requests == []


def test_taxonomies_take_loaded_result_not_a_path(make_lm, tmp_path):
    lm, _ = make_lm(_route)
    job = _job(tmp_path)
    loaded = tracepress.harbor_load(job)
    out = tracepress.harbor_agent_taxonomy(loaded, lm=lm)
    assert out["trial_key"].tolist() == ["trial-a", "trial-b"]
    with pytest.raises(TypeError, match="harbor_load"):
        tracepress.harbor_agent_taxonomy(job, lm=lm)
    with pytest.raises(TypeError, match="harbor_load"):
        tracepress.harbor_command_taxonomy(job, lm=lm)


def test_workspace_pipelines_run_both_taxonomies(make_lm, tmp_path):
    lm, _ = make_lm(_route)
    ws = tracepress.Workspace(tmp_path / "ws")

    @ws.pipeline
    def harbor_agent_taxonomy(steps, *, lm=None):
        return tracepress.harbor_agent_taxonomy(steps, lm=lm)

    @ws.pipeline
    def harbor_command_taxonomy(steps, *, lm=None):
        return tracepress.harbor_command_taxonomy(steps, lm=lm)

    assert set(ws.pipelines) == {"harbor_agent_taxonomy", "harbor_command_taxonomy"}
    with pytest.raises(TypeError, match="not a workspace pipeline"):
        ws.pipeline(tracepress.harbor_load)

    steps = tracepress.harbor_load(_job(tmp_path)).steps
    labels = ws.run("harbor_agent_taxonomy", steps, lm=lm)
    assert labels.manifest["status"] == "ok"
    saved = ws["agent_taxonomy_labels"]
    assert list(saved.set_index("trial_key").loc["trial-a", "classes"]) == ["verification", "execution"]
    assert "USER_TASK" in " ".join(ws["agent_step_summaries"]["step_summary"].astype(str))

    commands = ws.run("harbor_command_taxonomy", steps, lm=lm)
    assert commands.manifest["status"] == "ok"
    assert ws["command_taxonomy"]["failure_class"].tolist() == ["invocation"]
