from typing import Literal

import dspy
from pydantic import BaseModel, Field


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


class PiiFinding(BaseModel):
    kind: Literal["ssn", "phone", "email", "other"]
    evidence: str = Field(description="Short quote of the PII span.")


class DetectPiiInStep(dspy.Signature):
    """Detect personally identifiable information in one agent trajectory step.

    Look for Social Security numbers, phone numbers, email addresses, and similar
    personal identifiers. Ignore genomic IDs, file paths, UUIDs, hashes, and other
    technical identifiers.
    """

    step_index: int = dspy.InputField()
    source: str = dspy.InputField()
    model_name: str = dspy.InputField()
    message: str = dspy.InputField()
    tool_calls: list[str] = dspy.InputField()
    observation: str = dspy.InputField()

    findings: list[PiiFinding] = dspy.OutputField(desc="Every distinct PII span in this step. Empty if none.")


detect_pii_in_step = dspy.Predict(DetectPiiInStep)


def aggregate_pii_findings(steps, ctx):
    findings = []
    for step in steps:
        for finding in step.get("findings") or []:
            item = _finding_dict(finding)
            if item is None:
                continue
            findings.append({"step_index": step["step_index"], **item})
    return {
        "session_id": ctx.parent["session_id"],
        "trajectory_path": ctx.parent.get("_path", ""),
        "has_pii": bool(findings),
        "findings": findings,
    }


def _finding_dict(finding):
    data = finding.model_dump() if hasattr(finding, "model_dump") else finding
    if not isinstance(data, dict) or not data.get("kind"):
        return None
    return {"kind": data["kind"], "evidence": data.get("evidence") or ""}


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
