"""Built-in failure taxonomies for Harbor jobs, from the Terminal-Bench 2.0 paper.

Merrill, Shaw, Carlini et al., "Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks
in Command Line Interfaces" (arXiv:2601.11868), §4.4-4.5 and Appendices C and E.

- `trajectory_taxonomy`: judges each failed trial against 9 failure modes in three classes
  (Execution, Coherence, Verification), one rubric judge per mode as in the paper. Steps are
  first digested one by one (map), then joined into a per-trial transcript (reduce), so long
  trials fit in one prompt and the judges still see the whole arc.
- `step_taxonomy`: two stages per tool call, as in the paper. Detect whether the command
  failed, then classify the failure into a category and subcategory of the command taxonomy.

Both taxonomies are plain module-level data; pass your own to override them.

    src = tp.read.harbor("jobs/my-job")
    labels = tp.pipelines.harbor.trajectory_taxonomy(src).collect()
    wide = tp.pipelines.harbor.trajectory_labels_wide(labels)
    failures = tp.pipelines.harbor.step_taxonomy(src).collect()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import dspy

from ..adapters.base import Adapter
from ..adapters.harbor import HarborAdapter, trial_values
from ..adapters.replay import ReplayAdapter
from ..plan import Table
from ..rows import Row

# --- Inputs -----------------------------------------------------------------------------

# Every agent-trace adapter's steps have these columns, so they go into signatures as they
# are, e.g. f"{STEP_INPUTS} -> what_happened". DSPy labels each field; nothing is cut.
STEP_INPUTS = "source, message, reasoning, tool_calls: list[dict], observation"

# The same for one row of a `tool_calls` table.
CALL_INPUTS = "name, arguments: dict, result"


def clip(value: Any, limit: int) -> str:
    """Text of at most about `limit` chars, keeping the head and the tail. For printing, not prompts."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 3], text[-(limit // 3):]
    return f"{head}\n... [{len(text) - len(head) - len(tail)} chars omitted] ...\n{tail}"


def _failed(reward: float | None) -> bool:
    return reward is None or reward < 1


def _tagged(table: Table, pipeline: str) -> Table:
    """Mark a pipeline's output node, so usage telemetry can count built-in pipelines."""
    table._node.pipeline = pipeline
    return table


def _rewards(src: Adapter) -> dict[str, float | None]:
    """Trial name -> reward, read from the trials' result.json without running a plan.

    There are no joins yet, so steps get their trial's reward through this lookup. Reading
    the metadata directly (rather than collecting the trials table) keeps the pipelines
    buildable inside a running event loop, e.g. Jupyter.
    """
    while isinstance(src, ReplayAdapter):
        src = src.source
    if not isinstance(src, HarborAdapter):
        raise TypeError(f"tp.pipelines.harbor needs a tp.read.harbor(...) source, got {type(src).__name__}")
    return {trial.result["trial_name"]: trial_values(trial)["reward"] for trial in src.trials}


# --- Trajectory taxonomy (paper §4.4, Appendix C) ----------------------------------------


@dataclass(frozen=True)
class Mode:
    cls: str  # "execution" | "coherence" | "verification"
    name: str
    definition: str
    rubric: str  # decision procedure and exclusions, given to the judge


TRAJECTORY_MODES: dict[str, Mode] = {
    "disobey_specification": Mode(
        "execution", "Disobey task specification",
        "The agent materially contradicts an explicit directive of the task: a required method, "
        "tool or source of truth, a constraint, or an output location or name.",
        "1. List the task's hard directives (must, only, exactly, required paths, sources of truth) "
        "and soft ones (should, prefer).\n"
        "2. Find actions or deliverables that contradict a directive. Using a different source of "
        "truth than the one the task names counts.\n"
        "3. Keep only material violations, ones that change the result or what the task asked for.\n"
        "4. If the agent noticed and fully reversed the violation before finishing, it doesn't count.\n"
        "Exclude: formatting-only deviations, and transient violations that were fully undone.",
    ),
    "step_repetition": Mode(
        "execution", "Step repetition",
        "The agent repeats the same phase or action without a meaningful change of strategy.",
        "1. Split the trajectory into phases (explore, implement, test, debug, verify).\n"
        "2. Within a phase, look for repeated tool calls or command blocks; treat calls that are "
        "the same in substance (same command with trivial edits, same idea re-tried) as identical.\n"
        "3. Classify the repetition: blind retry after an error with no change; repeating work "
        "that already succeeded; re-running the same verification over and over; abort loops "
        "(starting, abandoning and restarting the same approach).\n"
        "4. Match only if the same thing happens at least twice more after the first attempt "
        "without new information motivating it.\n"
        "Exclude: polling a long-running process, retries after a real fix, and iterating on "
        "a failing test while making real changes in between.",
    ),
    "unaware_of_termination": Mode(
        "execution", "Unaware of termination conditions",
        "The agent doesn't recognize when to stop: it keeps acting after the task succeeded, "
        "after it is clearly futile, or after it already declared completion.",
        "1. Check whether there was a clear stopping point.\n"
        "2. Identify the trigger: the goal was met and verified (C1), further progress was "
        "clearly impossible (C2), or the agent declared it was done (P1).\n"
        "3. Match if the agent kept taking substantive actions after that trigger: more edits "
        "after success, endless attempts after futility, or work after declaring completion.\n"
        "Exclude: a single verification turn right after success, and runs cut off by time limits.",
    ),
    "reasoning_action_mismatch": Mode(
        "coherence", "Reasoning-action mismatch",
        "What the agent says (its plan, or claims such as 'tests pass' or 'file written') "
        "is contradicted by what it actually does or by the logs and artifacts.",
        "1. Collect the agent's commitments and claims: stated plans, and claims about results.\n"
        "2. Ignore issues that are only about malformed tool-call JSON.\n"
        "3. Collect the behavioral evidence: the commands actually run and their outputs.\n"
        "4. Compare: does the agent claim success that the output contradicts, say it will do X "
        "and then do Y, or describe a file or result that doesn't exist?\n"
        "Match only if the mismatch is repeated or blocks progress, not a single harmless slip.",
    ),
    "context_loss": Mode(
        "coherence", "Context loss",
        "The agent forgets or contradicts recent environment state (files, installed tools, "
        "errors already seen) or its own recent commitments (plans, constraints it adopted).",
        "1. Find state or context that was clearly established in recent steps: a file's contents, "
        "a missing dependency, a decided plan, a constraint from the task.\n"
        "2. Find a later action that contradicts it: re-creating what exists, using what was shown "
        "to be missing, re-asking a question already answered, abandoning a plan without reason.\n"
        "3. Classify it as a state contradiction or a context contradiction.\n"
        "Exclude: contradictions explained by new information, and gaps right after a context "
        "summarization the agent couldn't see past.",
    ),
    "task_derailment": Mode(
        "coherence", "Task derailment",
        "The agent drifts away from the task's objective into irrelevant or unproductive work.",
        "1. State the task's objective.\n"
        "2. Find stretches of steps that don't serve it: polishing unrelated parts, rabbit holes on "
        "side issues, solving a different problem than the one asked.\n"
        "3. Match if such a stretch is substantial (several steps) or the final work targets the "
        "wrong objective.\n"
        "Exclude: brief exploration that informs the task, and necessary setup or debugging.",
    ),
    "premature_termination": Mode(
        "verification", "Premature termination",
        "The agent declares completion before meeting the objectives or delivering the required "
        "artifacts, without an actionable handoff.",
        "1. List the task's objectives and required deliverables.\n"
        "2. Confirm the agent itself ended the run (declared done or called the completion tool).\n"
        "3. Check whether any objective or deliverable was clearly missing or unfinished at that point.\n"
        "4. If the agent ended with an actionable handoff that names what is left and why, it "
        "doesn't count.\n"
        "Exclude: involuntary endings (crashes, timeouts, context limits).",
    ),
    "no_or_incorrect_verification": Mode(
        "verification", "No or incorrect verification",
        "The agent marks the task done without a substantive check of the final deliverable, "
        "or ignores failing core checks.",
        "1. Completion means the agent declared done or called its completion tool.\n"
        "2. Separate core functional properties of the deliverable (it runs, produces the right "
        "output, passes the tests) from peripheral structural ones (the file exists, a name).\n"
        "3. Look for an observed result checking a core property after the final change. The "
        "agent's own assertion that it works is not a check.\n"
        "4. Match if there was no such check, or a core check failed and the agent finished anyway.",
    ),
    "weak_verification": Mode(
        "verification", "Weak verification",
        "The agent's checks miss task-critical properties, or it fabricates data to justify progress.",
        "1. Data fabrication: did the agent invent inputs, outputs or results (mock data, "
        "hardcoded expected values) and use them to satisfy a requirement? Flag only if the "
        "fabricated data is used to claim a requirement is met.\n"
        "2. Weak verification: list the checks the agent ran, list the task's essential "
        "properties, and compare. Match if the checks skip essentials (only a smoke test, only "
        "the happy path, only that a file exists) and the agent relies on them to finish.\n"
        "3. Don't match if the agent later ran a decisive check that covers the gap.",
    ),
}

TRAJECTORY_CLASSES = ("execution", "coherence", "verification")

DIGEST_INSTRUCTIONS = (
    "This is one step of a coding agent working on a terminal task. Write a compact, factual "
    "digest of it that a reviewer can use later to judge the whole run without the raw step. "
    "Keep, in a few lines: the commands or actions taken (with key arguments); what the output "
    "showed (errors, exit status, test results, key values); any claims the agent made (e.g. "
    "'tests pass', 'done'); plans or commitments it stated; checks it performed and their result; "
    "and whether it declared the task complete. If `source` isn't 'agent', the step is something "
    "the agent was given (a user message, or a note or loaded skill from the harness); say what "
    "it tells the agent. Don't add judgments or guesses."
)

JUDGE_INSTRUCTIONS = (
    "You are reviewing a failed run of a coding agent on a terminal task. The transcript holds "
    "the task instruction and a digest of each agent step, in order. Decide whether the failure "
    "mode in `rubric` occurs in this run by following its decision procedure and exclusions. "
    "Answer matched=true only on clear evidence. In `rationale`, cite the step numbers that "
    "support the decision in one to three sentences."
)

def _transcript(rows: list[Row]) -> dict[str, Any]:
    # The task instruction goes in verbatim, since judges check the agent against its exact
    # directives. Every other step, including notes the harness inserted, as its digest.
    task = next((row for row in rows if row.source == "user"), None)
    parts = [f"[step {row.step}] {row.source}: {row.message if row is task else row.digest}" for row in rows]
    return {"transcript": "\n\n".join(parts), "n_steps": len(rows)}


def trajectory_taxonomy(src: Adapter, *, failed_only: bool = True, lm=None,
                        modes: dict[str, Mode] | None = None) -> Table:
    """One row per (trial, failure mode): `matched: bool` and a `rationale` citing steps.

    Keys are `dataset > task > trial > model > reward > mode`; columns include
    `failure_mode`, `mode_class`, `transcript` and `n_steps`. Trials without a trajectory
    (the agent crashed before writing one) are skipped. Costs one LM call per step plus one
    per mode per trial; `trajectory_labels_wide` pivots to one row per trial.
    """
    modes = TRAJECTORY_MODES if modes is None else modes
    rewards = _rewards(src)
    steps = src.table("steps")
    if failed_only:
        steps = steps.filter(lambda r: _failed(rewards.get(r.trial)))
    steps = steps.map(lambda r: {"reward": rewards.get(r.trial)}, columns={"reward": float})
    digests = steps.lm_map(f"{STEP_INPUTS} -> digest", instructions=DIGEST_INSTRUCTIONS, lm=lm)
    per_trial = digests.group_by("trial", "model", "reward").reduce(
        _transcript, columns={"transcript": str, "n_steps": int})
    judged = (
        per_trial
        .map(lambda r: {"modes": list(modes)}, columns={"modes": list})
        .explode("modes", level="mode", keep=("transcript", "n_steps"))
        .map(lambda r: _mode_columns(modes[r.mode], r.mode),
             columns={"failure_mode": str, "mode_class": str, "rubric": str})
    )
    return _tagged(judged.lm_map("rubric, transcript -> matched: bool, rationale: str",
                                 instructions=JUDGE_INSTRUCTIONS, lm=lm), "harbor.trajectory_taxonomy")


def _mode_columns(mode: Mode, mode_id: str) -> dict[str, Any]:
    rubric = f"Failure mode: {mode.name}\nDefinition: {mode.definition}\nDecision procedure:\n{mode.rubric}"
    return {"failure_mode": mode_id, "mode_class": mode.cls, "rubric": rubric}


def trajectory_labels_wide(rows: list[Row]) -> list[dict[str, Any]]:
    """Pivot `trajectory_taxonomy` rows to one dict per trial.

    Each dict has the trial's key fields, `<mode>: bool` and `<mode>_rationale` per mode,
    and `<class>: bool`, true when any mode in that class matched. A mode whose judge call
    was skipped is left out, and its class is false unless another of its modes matched.
    """
    classes = {row.mode_class for row in rows}
    by_trial: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = dict(row.key)
        out = by_trial.setdefault(row.key[:-1], {**{k: v for k, v in key.items() if k != "mode"},
                                                 **dict.fromkeys(classes, False)})
        out[row.failure_mode] = bool(row.matched)
        out[f"{row.failure_mode}_rationale"] = row.rationale
        out[row.mode_class] = out[row.mode_class] or bool(row.matched)
    return list(by_trial.values())


# --- Command taxonomy (paper §4.5, Appendix E) -------------------------------------------

COMMAND_TAXONOMY: dict[str, list[str]] = {
    "Invocation & CLI": [
        "command not found on PATH",
        "executable missing at specified path",
        "not executable or invalid shebang",
        "permission denied (execute)",
        "wrong interpreter for script",
        "non-standard CLI semantics or interactive default",
        "unknown option or subcommand",
        "missing required arguments",
        "shell syntax error, continuation or heredoc",
        "cross-shell or GNU/BSD variant mismatch",
        "stream editor (sed/awk) syntax error",
    ],
    "Filesystem & Permissions": [
        "file not found",
        "directory not found (cd target)",
        "destination not a directory (ENOTDIR/EISDIR)",
        "device or special path missing",
        "glob matched nothing",
        "permission denied (read/write)",
        "file exists / would overwrite (EEXIST)",
        "read-only filesystem or immutable",
        "broken symlink or dangling target",
        "path too long or invalid characters",
    ],
    "Environment & Configuration": [
        "missing required environment variable",
        "misconfigured tool paths or directories",
        "wrong working directory or project layout",
        "installer aborted or profile misconfigured",
        "conflicting environment managers or PATH shadowing",
        "locale or encoding issues",
        "security policy block (SELinux/AppArmor/Gatekeeper)",
        "quarantine or unsigned binary blocked",
    ],
    "Build, Toolchain & Packages": [
        "missing CLI utility dependency",
        "missing headers or libraries",
        "compiler or linker unavailable or misconfigured",
        "plugin or extension not installed",
        "unsupported language or library version",
        "architecture, ABI or loader mismatch",
        "dynamic linker or rpath issues",
        "GPU driver or toolkit mismatch",
        "missing project configuration files",
        "linker symbol conflicts or undefined references",
        "cross-platform tool differences (GNU/BSD)",
        "cache or state corruption",
    ],
    "Packages & Repositories": [
        "package not found in repositories",
        "repository misconfiguration",
        "externally managed environment blocks install (PEP 668)",
        "dependency resolution unsatisfied",
        "VCS tooling or auth missing for installs",
        "integrity or signature failure",
        "platform or marker incompatibility",
        "mixed package managers conflict",
    ],
    "Network & Remote Access": [
        "DNS resolution failure",
        "connection refused or unreachable",
        "HTTP/FTP 404 or unexpected redirect or content",
        "authentication or authorization failure",
        "TLS/SSL certificate error or clock skew",
        "proxy, firewall or egress blocked",
        "timeout",
        "port or address already in use",
        "rate limiting or quotas",
    ],
    "Runtime, Interpreters & Processes": [
        "process crash or segmentation fault",
        "unsupported instruction or opcode",
        "application-level failure reported",
        "no output produced by pipeline",
        "signals ignored or unhandled",
        "broken pipe",
        "memory exhaustion (OOM)",
        "disk full",
        "file descriptor exhaustion",
        "CPU time or ulimit exhaustion",
        "non-zero exit with diagnostic code",
    ],
    "Interpreters & REPLs": [
        "interpreter waiting at prompt instead of evaluating",
        "shell command typed into an interpreter",
        "undefined names or symbols at runtime",
        "type or attribute errors at runtime",
        "language syntax errors in source",
        "module or library not found at runtime",
    ],
    "Services & Platforms": [
        "service manager unavailable (systemd)",
        "service not running or PID file missing",
        "display or X server unavailable",
        "unsupported accelerator or platform mode",
        "container or VM preconditions unmet",
    ],
    "Data & Formats": [
        "corrupt or truncated archive or database",
        "encrypted content with wrong password",
        "invalid input format or dataset split",
        "empty or missing required content",
        "encoding and line-ending mismatches",
        "content-type mismatch",
    ],
    "Testing & Quality": [
        "test collection or import failures",
        "assertion or spec violation",
        "performance or threshold not satisfied",
        "merge conflict artifacts left in code",
        "flaky or time-dependent tests",
    ],
}

OTHER = "other"

DETECT_INSTRUCTIONS = (
    "You are an expert at reading command-line inputs and outputs. This is one tool call a coding "
    "agent made: the tool's `name`, its `arguments`, and everything it returned as `result`. First identify what kind of "
    "command it is and what kind of output it produced, then decide whether the output shows "
    "that the command failed (an error, a non-zero exit, a missing tool or file, a crash, a "
    "failing test). Expected non-matches, such as grep finding nothing while searching, are not failures."
)


def render_taxonomy(taxonomy: dict[str, list[str]]) -> str:
    return "\n".join(f"- {cat}\n" + "\n".join(f"  - {sub}" for sub in subs) for cat, subs in taxonomy.items())


def classify_signature(taxonomy: dict[str, list[str]]) -> type[dspy.Signature]:
    """`name, arguments, result -> error_category, error_subcategory`, with the category constrained to the taxonomy."""
    category = Literal[tuple(taxonomy)]  # type: ignore[valid-type]
    instructions = (
        "You are an expert at classifying errors in command-line inputs and outputs. "
        "This is one tool call a coding agent made (`name`, `arguments`, `result`), and it failed. "
        "Classify the failure into the taxonomy below: pick an `error_category` and one of its "
        "subcategories as `error_subcategory`, copied exactly. When several apply, pick the most "
        "specific one. If none fit, use the closest category and the subcategory 'other'.\n\n"
        f"Taxonomy:\n{render_taxonomy(taxonomy)}"
    )
    sig = dspy.Signature(f"{CALL_INPUTS} -> error_category, error_subcategory", instructions)
    return sig.with_updated_fields("error_category", type_=category)


def _snap(taxonomy: dict[str, list[str]], row: Row) -> dict[str, Any]:
    """Map the LM's subcategory onto the taxonomy, case-insensitively; 'other' if it isn't there."""
    subs = {s.lower(): s for s in taxonomy.get(row.error_category, [])}
    raw = (row.error_subcategory or "").strip()
    return {"error_subcategory": subs.get(raw.lower(), OTHER)}


def step_taxonomy(src: Adapter, *, failed_only: bool = False, lm=None,
                  taxonomy: dict[str, list[str]] | None = None) -> Table:
    """One row per failed tool call, with `error_category` and `error_subcategory`.

    Keys are the `tool_calls` keys (`dataset > task > trial > step > call`); columns include
    `name`, `arguments` and `result`. Every tool call costs one detection
    call, and each failed one a classification call. `failed_only` keeps calls from failed
    trials only.
    """
    taxonomy = COMMAND_TAXONOMY if taxonomy is None else taxonomy
    calls = src.table("tool_calls")
    if failed_only:
        rewards = _rewards(src)
        calls = calls.filter(lambda r: _failed(rewards.get(r.trial)))
    failures = calls.lm_filter(f"{CALL_INPUTS} -> is_failure_present: bool", instructions=DETECT_INSTRUCTIONS, lm=lm)
    return _tagged(
        failures
        .lm_map(classify_signature(taxonomy), lm=lm)
        .map(lambda r: _snap(taxonomy, r), columns={"error_subcategory": str}),
        "harbor.step_taxonomy",
    )
