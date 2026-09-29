"""``agent_run``: a tool-calling loop in which a strong model answers from a workspace.

The agent uses native tool calls: it sends lm15 requests through DSPy (``dspy.LM``
accepts a ``Request`` and returns a ``Response``), and this module runs the loop. It sends
the conversation, runs every tool call in the reply, appends the results, and repeats
until the model answers in text or the step limit is hit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from dspy.lm15 import Message, Request, ToolChoice

from ..model import Model, Usage, budget
from ..settings import get_agent_lm
from ..telemetry import capture_agent_run, in_agent
from ..workspace import Workspace
from .tools import Session, artifact_summary

SYSTEM = """\
You answer questions using a workspace of data produced by LLM pipelines.

The workspace holds artifacts: DataFrames saved by pipeline runs. Each artifact records \
the operation and instruction that produced it and its parent artifacts, so any summary \
can be traced back to the rows it came from.

How to work:
- Start from the overview below. Use describe_artifact and read_rows to look at data.
- Use python for anything exact: counting, filtering, joining, grouping. \
load("name") returns an artifact as a DataFrame.
- Reading artifacts is free. run_op and run_pipeline call models and cost money. Use \
them only when the existing artifacts can't answer the question. Try a small limit \
first, and subset with where.
- Summaries can be wrong. When a claim matters, check it against the underlying rows.

Answer in markdown. Be direct. Cite evidence as artifact names with row ids, e.g. \
`run_reasons` row 12, so the reader can check it.
"""


def overview(ws: Workspace, max_artifacts: int = 60) -> str:
    pipelines = [p.info() for p in ws.pipelines.values()]
    records = list({r["name"]: r for r in ws.artifact_records()}.values())
    shown = [artifact_summary(r, instruction_limit=200) for r in records[-max_artifacts:]]
    more = f"\n({len(records) - max_artifacts} older artifact names not shown; use list_artifacts.)" if len(records) > max_artifacts else ""
    return (
        "## Registered pipelines\n"
        + (json.dumps(pipelines, indent=1) if pipelines else "(none)")
        + "\n\n## Artifacts (latest version of each name)\n"
        + (json.dumps(shown, indent=1, default=str) if shown else "(none)")
        + more
    )


@dataclass
class Answer:
    """The agent's answer. Renders as markdown in a notebook."""

    question: str
    text: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    agent_usage: Usage = field(default_factory=Usage)
    op_calls: int = 0
    cost_usd: float = 0.0
    transcript_path: str | None = None
    run_id: str | None = None

    def __str__(self) -> str:
        return self.text

    def _repr_markdown_(self) -> str:
        return self.text


def _preview(arguments: dict[str, Any]) -> str:
    text = json.dumps(arguments, ensure_ascii=False, default=str)
    return text if len(text) <= 160 else text[:160] + "…"


def agent_run(
    ws: Workspace,
    question: str,
    *,
    model: Model | str | None = None,
    max_steps: int = 40,
    max_calls: int | None = 500,
    budget_usd: float | None = None,
    verbose: bool = True,
) -> Answer:
    """Answer ``question`` from the workspace.

    Args:
        model: the agent model; defaults to ``tracepress.configure(agent_lm=...)``, then ``lm``.
            Operators the agent runs use the configured ``lm``.
        max_steps: model turns before the agent must answer with what it has.
        max_calls: cap on the operator model calls the agent may cause.
        budget_usd: cap on dollars, counting the agent's own turns too. Enforced only for
            models constructed with ``price=``.
        verbose: print each tool call as it happens.
    """
    started = time.monotonic()
    exc: BaseException | None = None
    agent: Model | None = None
    usage = Usage()
    steps: list[dict[str, Any]] = []
    op_calls = 0
    n_turns = 0
    hit_step_limit = False
    scope = in_agent.set(True)
    try:
        agent = Model(model, cache=False) if isinstance(model, str) else get_agent_lm(model)
        session = Session(ws, question)
        tools = {t.spec.name: t for t in session.tools()}
        system = SYSTEM + "\n\n" + overview(ws)
        messages: list[Message] = [Message.user(question)]
        text = ""

        with budget(max_calls, budget_usd) as b:
            try:
                for turn in range(max_steps + 1):
                    n_turns += 1
                    final_turn = turn == max_steps
                    config = agent.config()
                    if final_turn:
                        messages.append(Message.user("You have reached the step limit. Answer now with what you have."))
                        config = replace(config, tool_choice=ToolChoice(mode="none"))
                    request = Request(
                        model=agent.lm.model,
                        system=system + f"\n\nRemaining budget for operator calls: {b.remaining()}.",
                        messages=tuple(messages),
                        tools=tuple(t.spec for t in tools.values()),
                        config=config,
                    )
                    response, used = agent.send(request)
                    usage = usage + used
                    messages.append(response.message)
                    calls = response.tool_calls
                    if response.text:
                        steps.append({"type": "text", "text": response.text})
                    if not calls or final_turn:
                        text = response.text or ""
                        hit_step_limit = final_turn
                        break
                    results = {}
                    for call in calls:
                        if verbose:
                            print(f"→ {call.name} {_preview(call.input)}")
                        output, is_error = session.dispatch(tools, call.name, call.input)
                        steps.append({"type": "tool", "name": call.name, "input": call.input, "output": output, "error": is_error})
                        results[call.id] = output
                    messages.append(Message.tool(results))
            finally:
                op_calls, cost = b.calls, b.cost_usd

        session.close()
        answer = Answer(
            question=question,
            text=text,
            steps=steps,
            agent_usage=usage,
            op_calls=op_calls,
            cost_usd=cost,
            run_id=session._run.id if session._run else None,
        )
        transcripts = ws.path / "agent_runs"
        transcripts.mkdir(exist_ok=True)
        path = transcripts / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
        path.write_text(
            json.dumps(
                {
                    "question": question,
                    "model": agent.model,
                    "answer": text,
                    "steps": steps,
                    "agent_usage": usage.as_dict(),
                    "op_calls": op_calls,
                    "cost_usd": cost,
                    "run_id": answer.run_id,
                },
                indent=1,
                default=str,
            )
        )
        answer.transcript_path = str(path)
        return answer
    except BaseException as caught:
        exc = caught
        raise
    finally:
        in_agent.reset(scope)
        capture_agent_run(
            ws,
            time.monotonic() - started,
            exc,
            model=agent.model if agent is not None else None,
            usage=usage,
            op_calls=op_calls,
            n_turns=n_turns,
            steps=steps,
            hit_step_limit=hit_step_limit,
        )
