import dspy


def prepare_harbor_trajectory(record):
    steps = record.get("steps", [])
    task = steps[0].get("message", "") if steps else ""
    final = steps[-1].get("message", "") if steps else ""
    agent = record.get("agent", {})
    eval_info = record.get("eval") or {}
    verifier_summary = _verifier_summary(eval_info)
    return {
        **record,
        "trajectory_path": record.get("_path", ""),
        "benchmark": record.get("_path", "").split("/", 1)[0],
        "task_name": eval_info.get("task_name", ""),
        "trial_id": eval_info.get("trial_id", ""),
        "reported_passed": bool(eval_info.get("passed")),
        "verifier_summary": verifier_summary,
        "task": _clip(task, 1800),
        "final_response": _clip(final, 1800),
        "step_count": len(steps),
        "agent_model": agent.get("model_name", ""),
    }


def split_harbor_steps(record, ctx):
    for step in record.get("steps", []):
        ctx.emit(
            {
                "step_index": step.get("step_id", 0),
                "source": step.get("source", ""),
                "model_name": step.get("model_name", ""),
                "message": _clip(step.get("message", ""), 1200),
                "tool_calls": _tool_call_names(step),
                "observation": _clip(_observation_text(step), 1200),
            }
        )


class SummarizeHarborStep(dspy.Signature):
    """Summarize one step from an agent trajectory in exactly one sentence."""

    step_index: int = dspy.InputField()
    source: str = dspy.InputField()
    model_name: str = dspy.InputField()
    message: str = dspy.InputField()
    tool_calls: list[str] = dspy.InputField()
    observation: str = dspy.InputField()

    step_summary: str = dspy.OutputField(desc="Exactly one concise sentence describing this step.")


summarize_harbor_step = dspy.Predict(SummarizeHarborStep)


class CategorizeHarborTrajectory(dspy.Signature):
    """Categorize one terminal-bench trajectory from one-sentence step summaries and metadata."""

    trajectory_path: str = dspy.InputField()
    benchmark: str = dspy.InputField()
    task_name: str = dspy.InputField()
    trial_id: str = dspy.InputField()
    reported_passed: bool = dspy.InputField()
    verifier_summary: str = dspy.InputField()
    task: str = dspy.InputField()
    final_response: str = dspy.InputField()
    step_count: int = dspy.InputField()
    agent_model: str = dspy.InputField()
    step_summary: list[str] = dspy.InputField()

    domain: str = dspy.OutputField(
        desc="Task domain. One of: bioinformatics, systems, data_processing, other."
    )
    outcome: str = dspy.OutputField(
        desc="Outcome. One of: passed, wrong_output, env_setup, incomplete, other."
    )
    trajectory_summary: str = dspy.OutputField(
        desc="One short sentence, at most 30 words, ending with a single period. No second sentence."
    )


categorize_harbor_trajectory = dspy.Predict(CategorizeHarborTrajectory)


def compact_harbor_category(record):
    return {
        "session_id": record.get("session_id"),
        "trajectory_path": record["trajectory_path"],
        "task_name": record["task_name"],
        "domain": record["domain"],
        "outcome": record["outcome"],
        "trajectory_summary": _one_sentence(record.get("trajectory_summary", "")),
    }


def _tool_call_names(step):
    calls = step.get("tool_calls") or []
    return [call.get("function_name", "") for call in calls[:8] if call.get("function_name")]


def _observation_text(step):
    observation = step.get("observation") or {}
    results = observation.get("results") or []
    chunks = []
    for result in results[:3]:
        content = result.get("content")
        if content:
            chunks.append(str(content))
    return " ".join(chunks)


def _verifier_summary(eval_info):
    ctrf_summary = ((eval_info.get("ctrf") or {}).get("summary") or {})
    failed_tests = (eval_info.get("ctrf") or {}).get("failed_tests") or []
    failed_names = [test.get("name", "") for test in failed_tests[:8] if test.get("name")]
    parts = [
        f"reward={eval_info.get('reward')}",
        f"passed={eval_info.get('passed')}",
        f"tests={ctrf_summary.get('tests')}",
        f"tests_passed={ctrf_summary.get('passed')}",
        f"tests_failed={ctrf_summary.get('failed')}",
        f"failed_test_names={failed_names}",
    ]
    return _clip("; ".join(parts), 1400)


def _clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _one_sentence(text):
    text = " ".join(str(text).split()).strip()
    for ending in (". ", "! ", "? "):
        head, sep, rest = text.partition(ending)
        if sep:
            return head.rstrip() + ending[0]
    return text
