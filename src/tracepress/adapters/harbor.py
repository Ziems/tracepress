"""Harbor jobs: trial metadata from result.json, steps from the ATIF trajectory.

Levels: dataset > task > trial > step. Job, agent and model are columns, so a task's
trials can be compared across jobs with `group_by("task")`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..rows import Closed, Event, Key, Row, RowAdded
from ..schema import Schema
from .atif import STEP_COLUMNS, TOOL_CALL_COLUMNS, load_json, step_rows
from .base import Adapter

TRIAL_COLUMNS = {
    "job_id": str,
    "agent": str,
    "model": str,
    "agent_kwargs": dict,
    "reward": float,
    "exception_type": str,
    "exception_message": str,
    "cost_usd": float,
    "input_tokens": int,
    "output_tokens": int,
    "cache_tokens": int,
    "started_at": str,
    "finished_at": str,
    "duration_s": float,
    "path": str,
}

# Small trial fields copied onto steps, tool calls and tests so they can be grouped on directly.
CARRIED = ("job_id", "agent", "model")

TEST_COLUMNS = {"name": str, "status": str, "duration_ms": int, "message": str, "trace": str}


@dataclass
class Trial:
    path: Path
    result: dict
    dataset: str
    task: str

    @property
    def key(self) -> Key:
        return (("dataset", self.dataset), ("task", self.task), ("trial", self.result["trial_name"]))


def _reward(result: dict) -> float | None:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get("reward")
    return float(value) if value is not None else None


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (parse(end) - parse(start)).total_seconds()


def trial_values(trial: Trial) -> dict[str, Any]:
    r = trial.result
    config = r.get("config") or {}
    agent = config.get("agent") or {}
    agent_result = r.get("agent_result") or {}
    exc = r.get("exception_info") or {}
    return {
        "job_id": config.get("job_id"),
        "agent": agent.get("name"),
        "model": agent.get("model_name"),
        "agent_kwargs": agent.get("kwargs") or {},
        "reward": _reward(r),
        "exception_type": exc.get("exception_type"),
        "exception_message": exc.get("exception_message"),
        "cost_usd": agent_result.get("cost_usd"),
        "input_tokens": agent_result.get("n_input_tokens"),
        "output_tokens": agent_result.get("n_output_tokens"),
        "cache_tokens": agent_result.get("n_cache_tokens"),
        "started_at": r.get("started_at"),
        "finished_at": r.get("finished_at"),
        "duration_s": _seconds(r.get("started_at"), r.get("finished_at")),
        "path": str(trial.path),
    }


class HarborAdapter(Adapter):
    levels = ("dataset", "task", "trial", "step")

    def __init__(self, trials: list[Trial]):
        self.trials = trials

    def tables(self) -> dict[str, Schema]:
        carried = {c: str for c in CARRIED}
        return {
            "trials": Schema(self.levels[:3], dict(TRIAL_COLUMNS)),
            "steps": Schema(self.levels, {**carried, **STEP_COLUMNS}),
            "tool_calls": Schema((*self.levels, "call"), {**carried, **TOOL_CALL_COLUMNS}),
            "tests": Schema((*self.levels[:3], "test"), {**carried, **TEST_COLUMNS}),
        }

    async def events(self, table: str) -> AsyncIterator[Event]:
        # Batch: every trial is finished, so close each task and dataset as soon as
        # its trials have been emitted. Streaming closure for Harbor is still open.
        by_task: dict[tuple[str, str], list[Trial]] = {}
        for trial in self.trials:
            by_task.setdefault((trial.dataset, trial.task), []).append(trial)
        datasets = sorted({d for d, _ in by_task})
        for dataset in datasets:
            for (d, task), trials in sorted(by_task.items()):
                if d != dataset:
                    continue
                for trial in trials:
                    async for event in self._trial_events(trial, table):
                        yield event
                    yield Closed(trial.key)
                yield Closed(trial.key[:2])
            yield Closed((("dataset", dataset),))

    async def _trial_events(self, trial: Trial, table: str) -> AsyncIterator[Event]:
        values = trial_values(trial)
        carried = {c: values[c] for c in CARRIED}
        if table == "trials":
            yield RowAdded(Row(trial.key, values))
        elif table == "tests":
            ctrf = trial.path / "verifier" / "ctrf.json"
            if ctrf.exists():
                data = await load_json(ctrf)
                for test in (data.get("results") or {}).get("tests", []):
                    yield RowAdded(Row((*trial.key, ("test", test.get("name"))), {
                        **carried,
                        "name": test.get("name"),
                        "status": test.get("status"),
                        "duration_ms": test.get("duration"),
                        "message": test.get("message"),
                        "trace": test.get("trace"),
                    }))
        else:
            path = trial.path / "agent" / "trajectory.json"
            if not path.exists():
                return  # the agent crashed before writing one; the trial row still exists
            for step, calls in step_rows(await load_json(path), trial.key, carried):
                if table == "steps":
                    yield RowAdded(step)
                else:
                    for call in calls:
                        yield RowAdded(call)
                    yield Closed(step.key)


def _is_trial_result(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and "trial_name" in data else None


def read_harbor(path: str | Path, tasks: list[str] | None = None, limit: int | None = None) -> HarborAdapter:
    """A Harbor job directory, its `trials/` directory, or a directory of jobs.

    `tasks` keeps only those tasks (full `org/name` or just `name`); `limit` keeps the
    first N trials in (task, trial) order, handy for cheap test runs.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    found = []
    for result_path in sorted(root.rglob("result.json")):
        if not (result_path.parent / "config.json").exists():
            continue  # not a trial directory (e.g. a result.json inside artifacts/)
        result = _is_trial_result(result_path)
        if result is None:
            continue
        task = result.get("task_name") or result["trial_name"].rsplit("__", 1)[0]
        dataset = result.get("source") or "local"
        if tasks and task not in tasks and task.rsplit("/", 1)[-1] not in tasks:
            continue
        found.append(Trial(result_path.parent, result, dataset, task))
    found.sort(key=lambda t: (t.dataset, t.task, t.result["trial_name"]))
    if limit is not None:
        found = found[:limit]
    if not found:
        raise FileNotFoundError(f"no Harbor trials found under {root}")
    return HarborAdapter(found)
