import dspy


def active(trace):
    return trace if trace.get("enabled", True) else None


def split_readings(trace, ctx):
    for position, reading in enumerate(trace["readings"], start=1):
        ctx.emit(
            {
                "position": position,
                "reading": reading,
                "notes": trace["notes"],
            }
        )


class LabelReading(dspy.Signature):
    """Classify one mine sensor reading using its value and nearby field notes."""

    position: int = dspy.InputField()
    reading: float = dspy.InputField()
    notes: str = dspy.InputField()

    material: str = dspy.OutputField(desc="A short label such as ore, waste, clay, void, or unknown.")
    rationale: str = dspy.OutputField(desc="A reason under 20 words.")


label_reading = dspy.Predict(LabelReading)


def summarize_trace(readings, ctx):
    labels = [reading["material"] for reading in readings]
    return {
        "trace_id": ctx.parent["id"],
        "labeled_readings": readings,
        "materials": labels,
        "sample_count": len(readings),
    }


HARBOR_EXAMPLE_TRAJECTORY = "atrx-vep-crispr__76064f07"


def selected_harbor_trajectory(record):
    return record if HARBOR_EXAMPLE_TRAJECTORY in record.get("_path", "") else None


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


class AnalyzeHarborTrajectory(dspy.Signature):
    """Analyze one terminal-bench trajectory from one-sentence step summaries and metadata."""

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

    failed: bool = dspy.OutputField(desc="Whether the trajectory failed.")
    failure_cause: str = dspy.OutputField(desc="If failed, the specific cause; otherwise say none.")
    trajectory_summary: str = dspy.OutputField(desc="Exactly one sentence summarizing the whole trajectory.")


analyze_harbor_trajectory = dspy.Predict(AnalyzeHarborTrajectory)


def compact_harbor_summary(record):
    return {
        "trajectory_path": record["trajectory_path"],
        "task_name": record["task_name"],
        "trial_id": record["trial_id"],
        "agent_model": record["agent_model"],
        "step_count": record["step_count"],
        "reported_passed": record["reported_passed"],
        "verifier_summary": record["verifier_summary"],
        "failed": record["failed"],
        "failure_cause": record["failure_cause"],
        "trajectory_summary": record["trajectory_summary"],
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
