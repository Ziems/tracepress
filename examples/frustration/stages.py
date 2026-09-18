import dspy


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


class DetectFrustrationInStep(dspy.Signature):
    """Detect whether a human user is unhappy or frustrated in one agent trajectory step.

    Only flag dissatisfaction in user messages. Ignore failing tests, stack traces,
    missing packages, agent self-corrections, and other technical errors.
    """

    step_index: int = dspy.InputField()
    source: str = dspy.InputField()
    model_name: str = dspy.InputField()
    message: str = dspy.InputField()
    tool_calls: list[str] = dspy.InputField()
    observation: str = dspy.InputField()

    has_frustration: bool = dspy.OutputField(
        desc="True if a human user expresses frustration, impatience, or dissatisfaction."
    )
    frustration_kind: str = dspy.OutputField(
        desc="One of: frustrated, impatient, annoyed, or none. Never null."
    )
    evidence: str = dspy.OutputField(
        desc="Short user quote if frustrated, otherwise an empty string. Never null. Never explain."
    )


detect_frustration_in_step = dspy.Predict(DetectFrustrationInStep)


def aggregate_frustration_findings(steps, ctx):
    findings = [
        {
            "step_index": step["step_index"],
            "frustration_kind": step["frustration_kind"],
            "evidence": step["evidence"],
        }
        for step in steps
        if step["has_frustration"]
    ]
    return {
        "session_id": ctx.parent["session_id"],
        "trajectory_path": ctx.parent.get("_path", ""),
        "has_frustration": bool(findings),
        "findings": findings,
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


def _clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
