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
