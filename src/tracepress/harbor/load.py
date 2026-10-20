"""Load Harbor jobs, trials, and ATIF trajectories into DataFrames.

    loaded = tracepress.harbor_load("jobs/my-job")
    loaded.trials   # one row per trial
    loaded.steps    # one row per step, including embedded subagents
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

TRIAL_COLUMNS = [
    "trial_key",
    "trial_name",
    "task_name",
    "session_id",
    "trajectory_id",
    "agent_name",
    "agent_version",
    "model_name",
    "reward",
    "rewards",
    "exception",
    "failed_tests",
    "n_steps",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_cost_usd",
    "source_path",
]

STEP_COLUMNS = TRIAL_COLUMNS + [
    "step_id",
    "source",
    "message",
    "reasoning",
    "tool_calls",
    "observation",
    "step_model",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "is_subagent",
    "is_copied_context",
]


@dataclass(frozen=True)
class HarborLoad:
    """One load of Harbor data: a row per trial, and a row per step."""

    trials: pd.DataFrame
    steps: pd.DataFrame


def harbor_load(path: str | Path | Iterable[str | Path]) -> HarborLoad:
    """Load a job directory, a trial directory, an ATIF file, or a list of those.

    This reads files. It is not a workspace pipeline: call it yourself, then pass
    the result to ``harbor_agent_taxonomy`` or ``harbor_command_taxonomy``.

    A directory is a trial when it contains ``agent/trajectory.json`` or
    ``trajectory.json``. A directory is a job when it is not a trial and has trial
    subdirectories. A file must be an ATIF object with a ``steps`` array. The
    job-level ``result.json`` is not a trial.

    ``trial_key`` is the trial's path relative to what was loaded. A list of paths is
    prefixed with each path's name so two jobs can be concatenated without collisions.
    A bare trajectory file uses ``session_id``, then ``trajectory_id``, then the file stem.
    """
    paths = _paths(path)
    prefixes = _prefixes(paths)
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for item, prefix in zip(paths, prefixes):
        found_trials, found_steps = _load_one(item, prefix)
        trials.extend(found_trials)
        steps.extend(found_steps)
    return HarborLoad(
        trials=pd.DataFrame(trials, columns=TRIAL_COLUMNS),
        steps=pd.DataFrame(steps, columns=STEP_COLUMNS),
    )


harbor_load._tracepress_pipeline = False


def _paths(path: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(path, (str, Path)):
        return [Path(path)]
    return [Path(item) for item in path]


def _prefixes(paths: list[Path]) -> list[str | None]:
    """A name prefix per path when more than one root was loaded, else none."""
    if len(paths) <= 1:
        return [None] * len(paths)
    names = [item.name for item in paths]
    if len(set(names)) == len(names):
        return names
    return [item.as_posix() for item in paths]


def _load_one(path: Path, prefix: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Download the job from Harbor Hub "
            '(harbor job download "<job-id>"), then load that directory.'
        )
    if path.is_file():
        return _load_file(path, prefix)
    trajectory = _trajectory_file(path)
    if trajectory is not None:
        key = f"{prefix}/{path.name}" if prefix else path.name
        return _load_trial(path, trajectory, key)
    children = sorted(child for child in path.iterdir() if child.is_dir() and _trajectory_file(child) is not None)
    if not children:
        raise ValueError(f"{path} is not a Harbor trajectory, trial directory, or job directory")
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for child in children:
        relative = child.relative_to(path).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        found_trials, found_steps = _load_trial(child, _trajectory_file(child), key)
        trials.extend(found_trials)
        steps.extend(found_steps)
    return trials, steps


def _trajectory_file(directory: Path) -> Path | None:
    under_agent = directory / "agent" / "trajectory.json"
    if under_agent.is_file():
        return under_agent
    direct = directory / "trajectory.json"
    if direct.is_file():
        return direct
    return None


def _load_file(path: Path, prefix: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory = _read_json(path)
    _require_steps(trajectory, path)
    key = _str(trajectory.get("session_id")) or _str(trajectory.get("trajectory_id")) or path.stem
    if prefix:
        key = f"{prefix}/{key}"
    return _rows(trajectory, trial_key=key, source_path=path, meta=_blank_meta(), failed_tests="")


def _load_trial(
    directory: Path, trajectory_path: Path | None, trial_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if trajectory_path is None:
        raise ValueError(f"{directory} has no trajectory.json")
    trajectory = _read_json(trajectory_path)
    _require_steps(trajectory, trajectory_path)
    return _rows(
        trajectory,
        trial_key=trial_key,
        source_path=directory,
        meta=_result_meta(directory / "result.json"),
        failed_tests=_failed_tests(directory / "verifier" / "ctrf.json"),
    )


def _rows(
    trajectory: Mapping[str, Any],
    *,
    trial_key: str,
    source_path: Path,
    meta: dict[str, Any],
    failed_tests: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agent = trajectory.get("agent") if isinstance(trajectory.get("agent"), dict) else {}
    metrics = trajectory.get("final_metrics") if isinstance(trajectory.get("final_metrics"), dict) else {}
    steps = _step_rows(trajectory, is_subagent=False)
    trial = {
        "trial_key": trial_key,
        "trial_name": meta["trial_name"],
        "task_name": meta["task_name"],
        "session_id": _str(trajectory.get("session_id")),
        "trajectory_id": _str(trajectory.get("trajectory_id")),
        "agent_name": _str(agent.get("name")),
        "agent_version": _str(agent.get("version")),
        "model_name": _str(agent.get("model_name")),
        "reward": meta["reward"],
        "rewards": meta["rewards"],
        "exception": meta["exception"],
        "failed_tests": failed_tests,
        "n_steps": len(steps),
        "total_prompt_tokens": _optional_int(metrics.get("total_prompt_tokens")),
        "total_completion_tokens": _optional_int(metrics.get("total_completion_tokens")),
        "total_cost_usd": _optional_float(metrics.get("total_cost_usd")),
        "source_path": str(source_path),
    }
    return [trial], [{**trial, **step} for step in steps]


def _step_rows(trajectory: Mapping[str, Any], *, is_subagent: bool) -> list[dict[str, Any]]:
    rows = [
        _one_step(step, index, is_subagent=is_subagent)
        for index, step in enumerate(_as_list(trajectory.get("steps")), 1)
    ]
    for subagent in _as_list(trajectory.get("subagent_trajectories")):
        if isinstance(subagent, dict):
            rows.extend(_step_rows(subagent, is_subagent=True))
    return rows


def _one_step(step: Any, index: int, *, is_subagent: bool) -> dict[str, Any]:
    if not isinstance(step, dict):
        step = {}
    metrics = step.get("metrics") if isinstance(step.get("metrics"), dict) else {}
    step_id = _optional_int(step.get("step_id"))
    return {
        "step_id": index if step_id is None else step_id,
        "source": _str(step.get("source")),
        "message": _content_text(step.get("message")),
        "reasoning": _str(step.get("reasoning_content")),
        "tool_calls": _tool_calls_text(step.get("tool_calls")),
        "observation": _observation_text(step.get("observation")),
        "step_model": _str(step.get("model_name")),
        "prompt_tokens": _optional_int(metrics.get("prompt_tokens")),
        "completion_tokens": _optional_int(metrics.get("completion_tokens")),
        "cost_usd": _optional_float(metrics.get("cost_usd")),
        "is_subagent": is_subagent,
        "is_copied_context": bool(step.get("is_copied_context")),
    }


def _blank_meta() -> dict[str, Any]:
    return {"trial_name": "", "task_name": "", "reward": None, "rewards": "", "exception": ""}


def _result_meta(path: Path) -> dict[str, Any]:
    meta = _blank_meta()
    if not path.is_file():
        return meta
    data = _read_json(path)
    if not isinstance(data, dict):
        return meta
    meta["trial_name"] = _str(data.get("trial_name"))
    meta["task_name"] = _str(data.get("task_name"))
    verifier = data.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if isinstance(rewards, dict):
        meta["rewards"] = json.dumps(rewards, default=str, sort_keys=True)
        if "reward" in rewards:
            meta["reward"] = _optional_float(rewards.get("reward"))
    meta["exception"] = _exception_text(data.get("exception_info"))
    return meta


def _failed_tests(path: Path) -> str:
    if not path.is_file():
        return ""
    data = _read_json(path)
    tests = _ctrf_tests(data)
    lines = []
    for test in tests:
        if not isinstance(test, dict) or _str(test.get("status")).lower() != "failed":
            continue
        name = _str(test.get("name")) or "unknown"
        message = _content_text(test.get("message")).strip()
        lines.append(f"{name}: {message}" if message else name)
    return "\n".join(lines)


def _ctrf_tests(data: Any) -> list[Any]:
    if not isinstance(data, dict):
        return []
    results = data.get("results")
    if isinstance(results, dict) and isinstance(results.get("tests"), list):
        return results["tests"]
    if isinstance(data.get("tests"), list):
        return data["tests"]
    return []


def _exception_text(info: Any) -> str:
    if info is None or info == "":
        return ""
    if isinstance(info, str):
        return info
    if isinstance(info, dict):
        kind = _str(info.get("exception_type") or info.get("type"))
        message = _str(info.get("exception_message") or info.get("message") or info.get("error"))
        if kind and message:
            return f"{kind}: {message}"
        if kind or message:
            return kind or message
        return json.dumps(info, default=str, sort_keys=True)
    return str(info)


def _observation_text(observation: Any) -> str:
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    if isinstance(observation, list):
        return _results_text(observation)
    if isinstance(observation, dict):
        if "results" in observation:
            return _results_text(observation.get("results"))
        if "content" in observation:
            return _content_text(observation.get("content"))
        return _content_text(observation)
    return _text(observation)


def _results_text(results: Any) -> str:
    if not isinstance(results, list):
        return _content_text(results)
    chunks = []
    for result in results:
        if isinstance(result, str):
            chunks.append(result)
        elif isinstance(result, dict):
            content = _content_text(result.get("content"))
            if content:
                chunks.append(content)
        else:
            chunks.append(_text(result))
    return "\n".join(chunks)


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(chunk for chunk in (_content_text(part) for part in value) if chunk)
    if isinstance(value, dict):
        kind = _str(value.get("type"))
        if kind == "text" or (not kind and "text" in value):
            return _content_text(value.get("text"))
        if kind in {"image", "image_url"} or "image_url" in value:
            return "[image]"
        if kind:
            return "[image]"
        return _text(value)
    return _text(value)


def _tool_calls_text(calls: Any) -> str:
    if not isinstance(calls, list):
        return ""
    lines = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = _str(call.get("function_name")) or "tool"
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            rendered = json.dumps(arguments, default=str, separators=(",", ":"))
        else:
            rendered = _text(arguments)
        lines.append(f"{name}({rendered})")
    return "\n".join(lines)


def _require_steps(data: Any, path: Path) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise ValueError(f"{path} is not an ATIF trajectory (missing steps)")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc.msg}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc.reason}") from exc


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
