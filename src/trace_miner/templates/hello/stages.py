import re
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from trace_miner.jev import JevPredict


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


class CategorizeTrajectory(dspy.Signature):
    """Categorize one terminal-bench trajectory from its task and verifier outcome."""

    task: str = dspy.InputField()
    reported_passed: bool = dspy.InputField()
    verifier_summary: str = dspy.InputField()

    domain: Literal["bioinformatics", "systems", "data_processing", "other"] = dspy.OutputField(
        desc="Task domain."
    )
    outcome: Literal["passed", "wrong_output", "env_setup", "incomplete", "other"] = dspy.OutputField(
        desc="How the trajectory ended."
    )
    failed: bool = dspy.OutputField(desc="Whether the trajectory failed.")


categorize_trajectory = JevPredict(CategorizeTrajectory)


class DetectDoomLoop(dspy.Signature):
    """Decide whether these steps show the agent stuck in a genuine doom loop.

    A doom loop is the agent repeating the same or near-identical action three
    or more times and getting the same unproductive result each time, with no
    new information gained and no change in approach. The action has to be a
    real attempt: a tool, command, edit, or compile. Empty turns do not count.

    Judge the ACTION-RESULT PAIR, not the surface activity type. Do NOT flag:
    - Empty or no-op steps: summaries that say no-op, skipped, empty, or that
      name no tool, message, or observation. A run of those is idle, not a loop.
    - Exploration where each step targets a different file, query, hypothesis,
      or parameter, even if the tool or command shape looks similar (grepping
      different terms, reading different files, testing different values).
    - Verification or regression checks that confirm success or stability
      after a real change, even if the exact same command is rerun.
    - Polling or waiting on a long-running job where the checked state
      visibly changes between checks (progress counters, timestamps, partial
      output, growing byte counts).
    - A single transient error, or one flat step inside an otherwise
      progressing sequence.

    Only flag when the SAME specific action keeps producing the SAME specific
    unproductive result, and the agent does not adapt its approach.
    """

    step_summary: list[str] = dspy.InputField()

    doom_loop: bool = dspy.OutputField(
        desc="True only for a clear, sustained doom loop with no progress."
    )
    pattern: str = dspy.OutputField(
        desc="The exact repeated action and result, or an empty string if none."
    )


class _DetectDoomLoopFlag(dspy.Signature):
    __doc__ = DetectDoomLoop.__doc__

    step_summary: list[str] = dspy.InputField()

    doom_loop: bool = dspy.OutputField(
        desc="True only for a clear, sustained doom loop with no progress."
    )


class _ExplainDoomLoop(dspy.Signature):
    """A doom loop was already confirmed. Name the exact repeated action and result."""

    step_summary: list[str] = dspy.InputField()

    pattern: str = dspy.OutputField(desc="The exact repeated action and result.")


class _DetectDoomLoop(dspy.Module):
    """Jev answers the bool on every call; only a real hit pays for the explain call.

    A plain function can't be declared as an LM stage, so this is a custom
    `dspy.Module` instead — the runtime recognizes any `dspy.Module` and
    dispatches it exactly like `dspy.Predict`: bounded by `lm.max_concurrent`,
    run under `dspy.context(lm=...)` for whatever model this stage's role
    resolves to, and counted by `dspy.track_usage()`, which captures both
    sub-calls below since they happen inside that same `forward()`. `signature`
    exists only so the runtime can validate and route this stage; it is never
    called directly.
    """

    signature = DetectDoomLoop

    def __init__(self):
        super().__init__()
        self._flag = JevPredict(_DetectDoomLoopFlag)
        self._explain = dspy.Predict(_ExplainDoomLoop)

    def forward(self, step_summary):
        flag = self._flag(step_summary=step_summary)
        if not flag.doom_loop:
            return dspy.Prediction(doom_loop=False, pattern="")
        explained = self._explain(step_summary=step_summary)
        return dspy.Prediction(doom_loop=True, pattern=explained.pattern)


detect_doom_loop = _DetectDoomLoop()


def flag_doom_loop(record):
    """Alert when the reduce over the current step window judges a doom loop."""
    if not record.get("doom_loop"):
        return None
    return record.get("pattern") or "stuck repeating itself"


def flag_rollout_complete(record):
    """Alert once when the client marks its last posted step `final`."""
    if not record.get("final"):
        return None
    verdict = "doom loop" if record.get("doom_loop") else "clean"
    return f"rollout complete ({verdict})"


def compact_harbor_classification(record):
    return {
        "session_id": record.get("session_id"),
        "trajectory_path": record["trajectory_path"],
        "task_name": record["task_name"],
        "domain": record["domain"],
        "outcome": record["outcome"],
        "failed": record["failed"],
    }


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
