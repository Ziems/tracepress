import asyncio

import tracepress as tp
from tracepress.pipelines import harbor as pipes

from fakes import FuncLM


def judge(fields):
    """Digests echo the step; judges match step repetition and weak verification on alpha only."""
    if "tool_calls" in fields:
        return {"digest": f"did <{fields['message'][:20]}>"}
    rubric, transcript = fields["rubric"], fields["transcript"]
    hit = "alpha" in transcript and ("Step repetition" in rubric or "Weak verification" in rubric)
    return {"matched": str(hit), "rationale": "see step 2" if hit else "no evidence"}


def test_trajectory_taxonomy_judges_every_mode_per_failed_trial(harbor_dir):
    lm = FuncLM(judge)
    rows = pipes.trajectory_taxonomy(tp.read.harbor(harbor_dir), lm=lm).collect()

    # alpha__2 is the only failed trial with a trajectory (beta__2 crashed before writing one).
    assert {r.trial for r in rows} == {"alpha__2"}
    assert [r.failure_mode for r in rows] == list(pipes.TRAJECTORY_MODES)
    assert rows[0].reward == 0.0 and rows[0].model == "anthropic/claude-opus-5" and rows[0].n_steps == 3
    matched = {r.failure_mode for r in rows if r.matched}
    assert matched == {"step_repetition", "weak_verification"}

    # The task instruction goes in verbatim, agent steps as digests, in step order.
    transcript = rows[0].transcript
    assert transcript.startswith("[step 1] user: Please solve alpha.")
    assert transcript.index("[step 2] agent: did <Looking around.>") < transcript.index("[step 3] agent: did <Done.>")


def test_trajectory_digests_see_every_step_field_uncut(tmp_path):
    from conftest import write_trial

    output = "line\n" * 5000 + "FATAL: disk full"
    steps = [
        {"step_id": 1, "source": "user", "message": "Please solve gamma."},
        {"step_id": 2, "source": "agent", "message": "Writing.", "reasoning_content": "The disk might be full.",
         "tool_calls": [{"tool_call_id": "c1", "function_name": "Bash", "arguments": {"command": "make"}}],
         "observation": {"results": [{"source_call_id": "c1", "content": output}]}},
        {"step_id": 3, "source": "user", "message": "[Image: original 1654x2339, displayed at 1414x2000.]"},
    ]
    write_trial(tmp_path / "trials", "gamma", "gamma__1", 0.0, steps=steps)
    seen = []

    def fn(fields):
        if "tool_calls" in fields:
            seen.append(fields)
            return {"digest": f"digest of {fields['source']} step"}
        return {"matched": "False", "rationale": "none"}

    rows = pipes.trajectory_taxonomy(tp.read.harbor(tmp_path), lm=FuncLM(fn)).collect()
    agent = next(f for f in seen if f["source"] == "agent")
    assert agent["reasoning"] == "The disk might be full."
    assert "FATAL: disk full" in agent["tool_calls"] and "omitted" not in agent["tool_calls"]
    # Notes the harness inserted are digested too; only the task instruction stays verbatim.
    assert rows[0].transcript.splitlines()[0] == "[step 1] user: Please solve gamma."
    assert "[step 3] user: digest of user step" in rows[0].transcript


def test_trajectory_taxonomy_all_trials_and_custom_modes(harbor_dir):
    modes = {"only": pipes.Mode("execution", "Only mode", "defn", "1. decide")}
    rows = pipes.trajectory_taxonomy(tp.read.harbor(harbor_dir), failed_only=False, modes=modes,
                                     lm=FuncLM(judge)).collect()
    assert [r.trial for r in rows] == ["alpha__1", "alpha__2", "beta__1"]
    assert {r.failure_mode for r in rows} == {"only"}
    assert "Only mode" in rows[0].rubric


def test_trajectory_labels_wide(harbor_dir):
    rows = pipes.trajectory_taxonomy(tp.read.harbor(harbor_dir), failed_only=False, lm=FuncLM(judge)).collect()
    wide = pipes.trajectory_labels_wide(rows)
    assert [w["trial"] for w in wide] == ["alpha__1", "alpha__2", "beta__1"]
    alpha = wide[0]
    assert alpha["step_repetition"] and not alpha["context_loss"]
    assert alpha["step_repetition_rationale"] == "see step 2"
    assert alpha["execution"] and alpha["verification"] and not alpha["coherence"]
    assert not any(wide[2][c] for c in pipes.TRAJECTORY_CLASSES)


def test_trajectory_labels_wide_with_skipped_judges():
    """A trial whose judge calls for a whole class were skipped still gets every class."""
    def row(trial, mode, cls, matched):
        return tp.Row((("trial", trial), ("mode", mode)),
                      {"failure_mode": mode, "mode_class": cls, "matched": matched, "rationale": "r"})

    wide = pipes.trajectory_labels_wide([row("t1", "a", "execution", True), row("t1", "b", "coherence", False),
                                         row("t2", "b", "coherence", True)])
    assert wide[1] == {"trial": "t2", "execution": False, "coherence": True, "b": True, "b_rationale": "r"}
    assert "a" not in wide[1]  # the skipped mode's verdict is unknown, not false


def test_step_taxonomy_detects_then_classifies(harbor_dir):
    def fn(fields):
        if fields["name"] != "Bash":
            return {"is_failure_present": "False"}
        # Detection and classification share the input; answer both field sets.
        return {"is_failure_present": "True", "error_category": "Filesystem & Permissions",
                "error_subcategory": "FILE NOT FOUND"}

    rows = pipes.step_taxonomy(tp.read.harbor(harbor_dir), lm=FuncLM(fn)).collect()
    # Three trials with trajectories, one Bash call each; the Read calls aren't failures.
    assert [r.trial for r in rows] == ["alpha__1", "alpha__2", "beta__1"]
    assert all(r.name == "Bash" for r in rows)
    assert {(r.error_category, r.error_subcategory) for r in rows} == {("Filesystem & Permissions", "file not found")}


def test_step_taxonomy_snaps_unknown_subcategory_and_filters_failed(harbor_dir):
    def fn(fields):
        return {"is_failure_present": "True", "error_category": "Network & Remote Access",
                "error_subcategory": "gremlins"}

    rows = pipes.step_taxonomy(tp.read.harbor(harbor_dir), failed_only=True, lm=FuncLM(fn)).collect()
    assert {r.trial for r in rows} == {"alpha__2"}
    assert len(rows) == 2
    assert {r.error_subcategory for r in rows} == {pipes.OTHER}


def test_clip_keeps_head_and_tail():
    text = "HEAD" + "x" * 1000 + "TAIL"
    clipped = pipes.clip(text, 90)
    assert clipped.startswith("HEAD") and clipped.endswith("TAIL") and "omitted" in clipped
    assert pipes.clip("short", 90) == "short"


def test_pipelines_build_inside_an_event_loop_and_over_replay(harbor_dir):
    async def run():
        src = tp.replay(tp.read.harbor(harbor_dir))
        rows = await pipes.trajectory_taxonomy(src, lm=FuncLM(judge)).acollect()
        calls = pipes.step_taxonomy(src, failed_only=True, lm=FuncLM(lambda f: {"is_failure_present": "False"}))
        return rows, await calls.acollect()

    rows, calls = asyncio.run(run())
    assert {r.trial for r in rows} == {"alpha__2"}
    assert calls == []

