import dspy


def split_steps(record, ctx):
    for step in record.get("steps") or []:
        ctx.emit(
            {
                "step_id": step.get("step_id"),
                "message": _clip(step.get("message") or "", 400),
            }
        )


class SummarizeStep(dspy.Signature):
    """Name the concrete action in this step."""

    step_id: int = dspy.InputField()
    message: str = dspy.InputField()

    step_summary: str = dspy.OutputField(desc="One sentence, at most 15 words.")


summarize_step = dspy.Predict(SummarizeStep)


class LabelTrajectory(dspy.Signature):
    """Name what this agent trajectory was trying to do."""

    step_summary: list[str] = dspy.InputField()

    label: str = dspy.OutputField(desc="A few words naming the task.")


label_trajectory = dspy.Predict(LabelTrajectory)


def compact(record):
    eval_info = record.get("eval") or {}
    return {
        "session_id": record.get("session_id"),
        "task_name": eval_info.get("task_name", ""),
        "label": record.get("label", ""),
    }


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
