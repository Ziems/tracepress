import re
from typing import Literal

import dspy
from pydantic import BaseModel, Field


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
    """Name the concrete action in this step. Skip empty or no-op steps."""

    step_index: int = dspy.InputField()
    source: str = dspy.InputField()
    model_name: str = dspy.InputField()
    message: str = dspy.InputField()
    tool_calls: list[str] = dspy.InputField()
    observation: str = dspy.InputField()

    step_summary: str = dspy.OutputField(
        desc="One sentence, at most 20 words. Name the tool and what it did, or say no-op. No preamble."
    )


summarize_harbor_step = dspy.Predict(SummarizeHarborStep)


class AnalyzeHarborTrajectory(dspy.Signature):
    """Diagnose one terminal-bench trajectory. Be specific and brief."""

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
    failure_cause: str = dspy.OutputField(
        desc="If failed, one sentence of at most 25 words naming the concrete cause. Otherwise the word none."
    )
    trajectory_summary: str = dspy.OutputField(
        desc="One sentence, at most 30 words: the task goal and what the agent actually did. End with a period."
    )


analyze_harbor_trajectory = dspy.Predict(AnalyzeHarborTrajectory)


_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"\(\d{3}\) \d{3}-\d{4}")


def flag_sensitive_step(record):
    """Alert when a step contains a planted SSN, phone number, or frustrated user turn."""
    text = f"{record.get('message') or ''} {record.get('observation') or ''}"
    if _SSN.search(text):
        return "ssn"
    if _PHONE.search(text):
        return "phone number"
    if "taking forever" in text.lower():
        return "frustrated user"
    return None


def compact_harbor_summary(record):
    return {
        "session_id": record.get("session_id"),
        "trajectory_path": record["trajectory_path"],
        "task_name": record["task_name"],
        "trial_id": record["trial_id"],
        "agent_model": record["agent_model"],
        "step_count": record["step_count"],
        "reported_passed": record["reported_passed"],
        "verifier_summary": record["verifier_summary"],
        "failed": record["failed"],
        "failure_cause": _one_sentence(record.get("failure_cause", "")),
        "trajectory_summary": _one_sentence(record.get("trajectory_summary", "")),
    }


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
        "task_name": (ctx.parent.get("eval") or {}).get("task_name", ""),
        "has_pii": bool(findings),
        "findings": findings,
    }


def _finding_dict(finding):
    data = finding.model_dump() if hasattr(finding, "model_dump") else finding
    if not isinstance(data, dict) or not data.get("kind"):
        return None
    return {"kind": data["kind"], "evidence": data.get("evidence") or ""}


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
        "task_name": (ctx.parent.get("eval") or {}).get("task_name", ""),
        "has_frustration": bool(findings),
        "findings": findings,
    }


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
