"""Classify Harbor steps under the Terminal-Bench 2.0 taxonomies.

The Terminal Agent Taxonomy labels a whole trial. The command taxonomy labels each
failed shell, REPL, or CLI command. Both are the taxonomies published with
Terminal-Bench 2.0. Classes are filled in here from the model's labels, so the model
only chooses modes or a subcategory.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..model import Model
from ..run import ATTR, artifact_of, current_run, record
from .load import HarborLoad

TAT_CLASS = {
    "disobey_specification": "execution",
    "step_repetition": "execution",
    "unaware_of_termination": "execution",
    "reasoning_action_mismatch": "coherence",
    "context_loss": "coherence",
    "task_derailment": "coherence",
    "premature_termination": "verification",
    "no_or_incorrect_verification": "verification",
    "weak_verification": "verification",
}
TAT_MODES = tuple(TAT_CLASS)

COMMAND_CLASS = {
    "command_not_found": "invocation",
    "shell_syntax": "invocation",
    "other_invocation": "invocation",
    "module_not_found": "repl",
    "script_syntax": "repl",
    "other_repl": "repl",
    "application_failure": "runtime",
    "other_runtime": "runtime",
    "file_not_found": "filesystem",
    "other_filesystem": "filesystem",
    "other": "other",
}
COMMAND_SUBCATEGORIES = tuple(COMMAND_CLASS)

STEP_SUMMARY_INSTRUCTION = (
    "This is one step of a terminal-bench trial. instruction is the original task. "
    "Say what the agent was trying to do, using its reasoning when that is present, "
    "and then what actually happened. Name the command, path, claim, or error when "
    "there is one. Say if the agent declared the task done. Two or three sentences. "
    "Do not assign a taxonomy mode."
)

TAT_INSTRUCTION = """\
These records are summaries of the steps of one terminal-bench trial. instruction is the original task; it appears on the first record. Label why the agent failed, using the failed tests as the outcome, the instruction as the objective, and the summaries as the behavior. modes is every mode the summaries actually show, and may be empty. Mark a mode only when the summaries show it. A passing trial (a reward of 1 and no failed tests) has an empty modes list. explanation starts with one sentence on what the agent was supposed to do, then names the concrete command, path, error, or test that shows what went wrong.

disobey_specification (execution): materially contradicts an explicit directive, such as a required method, source of truth, constraint, or output location.
step_repetition (execution): repeats the same phase, meaning the same sub-goal, tool, target, and method, without a meaningful strategy change.
unaware_of_termination (execution): keeps acting after clear success, after established futility, or after declaring completion.
reasoning_action_mismatch (coherence): a stated claim is contradicted by the actions, logs, or artifacts.
context_loss (coherence): forgets or contradicts recent environment state or an earlier commitment.
task_derailment (coherence): leaves the objective for irrelevant or unproductive work.
premature_termination (verification): declares completion before the objectives or required artifacts are done, without naming the gaps.
no_or_incorrect_verification (verification): marks the task done or skips a verifier without checking the deliverable.
weak_verification (verification): checks something other than the property the task depends on, or checks fabricated inputs.\
"""

COMMAND_FILTER = (
    "A shell, REPL, or CLI command in this step failed: non-zero exit, exception, "
    "command not found, syntax error, missing file, or missing module. "
    "Successful commands, file reads that worked, and planning text do not count."
)

COMMAND_INSTRUCTION = """\
This step is one failed shell, REPL, or CLI command. instruction is the original task. reasoning is what the agent was trying to do. Pick the one subcategory that best describes the failure. explanation names what the agent was trying to do, the command, and the error.

command_not_found: the command is not on PATH.
shell_syntax: shell syntax, continuation, or heredoc quoting.
other_invocation: an unknown flag, the wrong interpreter, or a GNU/BSD mismatch.
module_not_found: the interpreter cannot import a module.
script_syntax: the program itself has a syntax error.
other_repl: any other interpreter failure, including a prompt that never returns.
application_failure: a correctly invoked program exited non-zero.
other_runtime: a crash, out of memory, or a signal.
file_not_found: a path in the command does not exist.
other_filesystem: permissions, ENOTDIR, an empty glob, or a dangling symlink.
other: none of the above.\
"""

_TAT_COLUMNS = ["trial_key", "explanation", "modes", "n_rows", "classes"]
_COMMAND_COLUMNS = ["subcategory", "failure_class", "explanation"]


def harbor_agent_taxonomy(
    loaded: HarborLoad | pd.DataFrame, *, lm: Model | None = None
) -> pd.DataFrame:
    """Label each Harbor trial with the Terminal Agent Taxonomy.

    ``loaded`` is the result of ``harbor_load``, or a steps DataFrame from one.
    A path is not accepted: load the job first. User and agent steps are kept
    (system prompts are dropped), each is mapped to a short ``step_summary``, then
    those notes are aggregated by ``trial_key``. The result is one row per trial with
    ``explanation``, ``modes``, and ``classes``. A passing trial has an empty ``modes``
    list. Inside a pipeline the summaries are ``agent_step_summaries`` and the labels
    are ``agent_taxonomy_labels``.
    """
    steps = _as_steps(loaded)
    framed = steps.loc[steps["source"].isin(["user", "agent"])].copy()
    if framed.empty:
        return pd.DataFrame(columns=_TAT_COLUMNS)
    framed["instruction"] = _task_instruction(framed)
    framed["step"] = [_render_step(row) for row in framed.to_dict("records")]
    summarized = framed.lm_map(
        "instruction, step_id, source, step -> step_summary",
        STEP_SUMMARY_INSTRUCTION,
        lm=lm,
        save_as="agent_step_summaries",
    )
    labeled = summarized.copy()
    labeled.loc[labeled.duplicated("trial_key"), "instruction"] = ""
    out = labeled.lm_agg(
        "instruction, task_name, reward, failed_tests, step_id, source, step_summary -> "
        f"explanation, modes: list[{_literal(TAT_MODES)}]",
        TAT_INSTRUCTION,
        by="trial_key",
        lm=lm,
        save_as="agent_taxonomy_labels",
    )
    out["classes"] = out["modes"].map(_classes_for)
    return _persist_derived(out)


def harbor_command_taxonomy(
    loaded: HarborLoad | pd.DataFrame, *, lm: Model | None = None
) -> pd.DataFrame:
    """Label failed commands in Harbor steps with the Terminal-Bench command taxonomy.

    ``loaded`` is the result of ``harbor_load``, or a steps DataFrame from one.
    A path is not accepted: load the job first. Agent steps are filtered to shell,
    REPL, or CLI commands that failed, then each remaining row is mapped to one
    subcategory. The map also sees the trial ``instruction`` and the step's
    ``reasoning``. ``failure_class`` is derived from that subcategory. When nothing
    failed, the result has zero rows and still has ``subcategory``, ``failure_class``,
    and ``explanation``. Inside a pipeline the filter is saved as ``command_failures``
    and the labels as ``command_taxonomy``.
    """
    steps = _as_steps(loaded)
    needed = steps.loc[steps["source"].isin(["user", "agent"])].copy()
    needed["instruction"] = _task_instruction(needed)
    agent = needed.loc[needed["source"].eq("agent")].copy()
    if agent.empty:
        return _empty_commands(agent)
    if "reasoning" not in agent.columns:
        agent["reasoning"] = ""
    failed = agent.lm_filter("tool_calls, observation", COMMAND_FILTER, lm=lm, save_as="command_failures")
    if failed.empty:
        return _empty_commands(failed)
    failed = failed.copy()
    failed["reasoning"] = [_cell(value) for value in failed["reasoning"]]
    out = failed.lm_map(
        "instruction, reasoning, message, tool_calls, observation -> "
        f"subcategory: {_literal(COMMAND_SUBCATEGORIES)}, explanation",
        COMMAND_INSTRUCTION,
        lm=lm,
        save_as="command_taxonomy",
    )
    out.insert(out.columns.get_loc("subcategory") + 1, "failure_class", out["subcategory"].map(COMMAND_CLASS))
    return _persist_derived(out)


def _as_steps(loaded: HarborLoad | pd.DataFrame) -> pd.DataFrame:
    """HarborLoad or a steps frame. Paths have to go through harbor_load first."""
    if isinstance(loaded, HarborLoad):
        return loaded.steps
    if isinstance(loaded, pd.DataFrame):
        return loaded
    raise TypeError(
        "Pass the result of harbor_load (or its .steps), not a path. "
        "Load first: loaded = tracepress.harbor_load(path)"
    )


def _task_instruction(steps: pd.DataFrame) -> pd.Series:
    """The first user message of each trial, copied onto every row of that trial."""
    if "trial_key" not in steps.columns:
        user = steps.loc[steps["source"].eq("user"), "message"].map(_cell)
        text = user.iloc[0] if len(user) else ""
        return pd.Series(text, index=steps.index)
    by_key = (
        steps.loc[steps["source"].eq("user")]
        .groupby("trial_key", sort=False)["message"]
        .first()
        .map(_cell)
    )
    return steps["trial_key"].map(by_key).fillna("")


def _render_step(row: dict[str, Any]) -> str:
    parts = [f"{row.get('step_id')} {row.get('source')}:"]
    if row.get("is_subagent"):
        parts.append("[subagent]")
    if row.get("is_copied_context"):
        parts.append("[copied]")
    message = _cell(row.get("message"))
    if message:
        parts.append(message)
    reasoning = _cell(row.get("reasoning"))
    if reasoning:
        parts.append("reasoning: " + reasoning)
    tools = _cell(row.get("tool_calls"))
    if tools:
        parts.append("tools: " + tools)
    observation = _cell(row.get("observation"))
    if observation:
        parts.append("observation: " + observation)
    return "\n".join(parts)


def _classes_for(modes: Any) -> list[str]:
    if modes is None:
        return []
    try:
        if modes != modes:  # NaN
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(modes, str):
        modes = [modes]
    classes: list[str] = []
    for mode in modes:
        cls = TAT_CLASS.get(str(mode))
        if cls and cls not in classes:
            classes.append(cls)
    return classes


def _persist_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite the artifact just saved by an operator so derived columns are on disk too.

    ``lm_agg`` and ``lm_map`` write the artifact before ``classes`` or ``failure_class``
    exist. The frame stored on the run is this same object, so only the parquet and the
    manifest's column list are behind.
    """
    run = current_run()
    ref = artifact_of(df)
    if run is None or not ref or run._frames.get(ref) is not df:
        return df
    from ..workspace import _write_parquet

    name = ref.split("/", 1)[1]
    _write_parquet(df, run.dir / f"{name}.parquet")
    for artifact in run.manifest["artifacts"]:
        if artifact["name"] == name:
            artifact["columns"] = [str(column) for column in df.columns]
    run._write_manifest()
    return df


def _empty_commands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.iloc[0:0].copy()
    out.attrs.pop(ATTR, None)
    for name in _COMMAND_COLUMNS:
        out[name] = pd.Series(dtype="object")
    return record(
        out,
        op="map",
        name="command_taxonomy",
        inputs=[df],
        instruction=COMMAND_INSTRUCTION,
        params={"empty": True},
    )


def _literal(names: tuple[str, ...]) -> str:
    return "Literal[" + ", ".join(repr(name) for name in names) + "]"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()

