from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from mine_trace.errors import ConfigError
from mine_trace.model import Config, Defaults, GroupSpec, PipelineSpec, StageRef, Step


_KNOWN_TOP_LEVEL = {"from", "defaults", "id_field", "pipelines", "concurrency"}
_KNOWN_DEFAULTS = {"on_child_error", "on_group_incomplete", "ordered"}
_STEP_KEYS = {"map", "filter", "stage", "tee", "group"}
_GROUP_KEYS = {
    "expand",
    "by",
    "every",
    "pipeline",
    "reduce",
    "on_child_error",
    "on_group_incomplete",
    "ordered",
    "sidechannels",
}
_GROUP_FORMS = {"expand", "by", "every"}
_CHILD_ERROR_POLICIES = {"fail", "partial", "skip"}
_INCOMPLETE_POLICIES = {"fail", "partial"}


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    base_dir = config_path.parent
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config must be a mapping.")

    unknown = set(raw) - _KNOWN_TOP_LEVEL
    if unknown:
        raise ConfigError(f"Unknown top-level key(s): {sorted(unknown)}")
    if "concurrency" in raw:
        raise ConfigError("`concurrency` is post-MVP and is not supported yet.")
    if "from" not in raw:
        raise ConfigError("Missing required top-level key: `from`.")
    if "pipelines" not in raw:
        raise ConfigError("Missing required top-level key: `pipelines`.")

    module = _load_stage_module(raw["from"], base_dir)
    defaults = _parse_defaults(raw.get("defaults", {}))
    id_field = raw.get("id_field")
    if id_field is not None and not isinstance(id_field, str):
        raise ConfigError("`id_field` must be a string when provided.")

    pipelines_raw = raw["pipelines"]
    if not isinstance(pipelines_raw, list) or not pipelines_raw:
        raise ConfigError("`pipelines` must be a non-empty list.")

    pipelines = tuple(
        _parse_pipeline(block, i, base_dir, module, defaults)
        for i, block in enumerate(pipelines_raw)
    )
    pipelines = _sort_pipelines(pipelines)
    return Config(config_path, base_dir, module, defaults, id_field, pipelines)


def _parse_defaults(raw: Any) -> Defaults:
    if raw is None:
        return Defaults()
    if not isinstance(raw, dict):
        raise ConfigError("`defaults` must be a mapping.")
    unknown = set(raw) - _KNOWN_DEFAULTS
    if unknown:
        raise ConfigError(f"Unknown defaults key(s): {sorted(unknown)}")
    on_child_error = raw.get("on_child_error", "fail")
    on_group_incomplete = raw.get("on_group_incomplete", "fail")
    ordered = raw.get("ordered", True)
    if on_child_error not in _CHILD_ERROR_POLICIES:
        raise ConfigError("`defaults.on_child_error` must be fail, partial, or skip.")
    if on_group_incomplete not in _INCOMPLETE_POLICIES:
        raise ConfigError("`defaults.on_group_incomplete` must be fail or partial.")
    if not isinstance(ordered, bool):
        raise ConfigError("`defaults.ordered` must be a boolean.")
    return Defaults(on_child_error, on_group_incomplete, ordered)


def _parse_pipeline(
    raw: Any,
    source_index: int,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
) -> PipelineSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"Pipeline {source_index + 1} must be a mapping.")
    required = {"in", "out", "pipeline"}
    missing = required - set(raw)
    if missing:
        raise ConfigError(f"Pipeline {source_index + 1} missing key(s): {sorted(missing)}")
    unknown = set(raw) - required
    if unknown:
        raise ConfigError(f"Pipeline {source_index + 1} unknown key(s): {sorted(unknown)}")

    inputs_raw = raw["in"]
    input_items = inputs_raw if isinstance(inputs_raw, list) else [inputs_raw]
    if not input_items:
        raise ConfigError(f"Pipeline {source_index + 1} must have at least one input.")
    inputs = tuple(_normalize_path(item, base_dir, f"pipeline {source_index + 1} input") for item in input_items)
    output = _normalize_path(raw["out"], base_dir, f"pipeline {source_index + 1} output")
    steps = _parse_steps(raw["pipeline"], base_dir, module, defaults, f"pipeline {source_index + 1}")
    return PipelineSpec(inputs, output, steps, source_index)


def _parse_steps(
    raw: Any,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
    where: str,
) -> tuple[Step, ...]:
    if not isinstance(raw, list):
        raise ConfigError(f"`pipeline` in {where} must be a list.")
    steps = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or len(item) != 1:
            raise ConfigError(f"Step {idx} in {where} must be a single-key mapping.")
        key, value = next(iter(item.items()))
        if key not in _STEP_KEYS:
            raise ConfigError(f"Unknown step key `{key}` in {where} step {idx}.")
        if key in {"map", "filter", "stage"}:
            steps.append(Step(kind=key, stage=_resolve_stage(value, module)))
        elif key == "tee":
            steps.append(Step(kind="tee", path=_normalize_path(value, base_dir, f"{where} tee")))
        else:
            steps.append(Step(kind="group", group=_parse_group(value, base_dir, module, defaults, f"{where} group step {idx}")))
    return tuple(steps)


def _parse_group(
    raw: Any,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
    where: str,
) -> GroupSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping.")
    unknown = set(raw) - _GROUP_KEYS
    if unknown:
        raise ConfigError(f"Unknown group key(s) in {where}: {sorted(unknown)}")
    if "sidechannels" in raw:
        raise ConfigError("`sidechannels` is post-MVP and is not supported yet.")
    present_forms = [key for key in _GROUP_FORMS if key in raw]
    if len(present_forms) != 1:
        raise ConfigError(f"{where} must declare exactly one of expand, by, or every.")
    if "reduce" not in raw:
        raise ConfigError(f"{where} missing required key: `reduce`.")

    on_child_error = raw.get("on_child_error", defaults.on_child_error)
    on_group_incomplete = raw.get("on_group_incomplete", defaults.on_group_incomplete)
    ordered = raw.get("ordered", defaults.ordered)
    if on_child_error not in _CHILD_ERROR_POLICIES:
        raise ConfigError(f"{where} `on_child_error` must be fail, partial, or skip.")
    if on_group_incomplete not in _INCOMPLETE_POLICIES:
        raise ConfigError(f"{where} `on_group_incomplete` must be fail or partial.")
    if not isinstance(ordered, bool):
        raise ConfigError(f"{where} `ordered` must be a boolean.")

    kind = present_forms[0]
    expand = _resolve_stage(raw["expand"], module) if kind == "expand" else None
    by: str | StageRef | None = None
    every: int | None = None
    if kind == "by":
        if not isinstance(raw["by"], str):
            raise ConfigError(f"{where} `by` must be a field or stage name.")
        by = _resolve_stage(raw["by"], module, missing_ok=True) or raw["by"]
    elif kind == "every":
        every = raw["every"]
        if not isinstance(every, int) or every <= 0:
            raise ConfigError(f"{where} `every` must be a positive integer.")

    child_steps = _parse_steps(raw.get("pipeline", []), base_dir, module, defaults, where)
    reduce = _resolve_stage(raw["reduce"], module)
    return GroupSpec(kind, expand, by, every, child_steps, reduce, on_child_error, on_group_incomplete, ordered)


def _resolve_stage(raw: Any, module: ModuleType, missing_ok: bool = False) -> StageRef | None:
    if not isinstance(raw, str):
        raise ConfigError("Stage names must be strings.")
    if not hasattr(module, raw):
        if missing_ok:
            return None
        raise ConfigError(f"Unknown stage `{raw}`.")
    fn = getattr(module, raw)
    if not callable(fn):
        raise ConfigError(f"Stage `{raw}` is not callable.")
    if inspect.isgeneratorfunction(fn) or inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        raise ConfigError(f"Stage `{raw}` uses a post-MVP callable shape.")

    sig = inspect.signature(fn)
    one_arg = _binds(sig, object())
    two_arg = _binds(sig, object(), object())
    if not one_arg and not two_arg:
        raise ConfigError(f"Stage `{raw}` cannot be called as (record) or (record, ctx): {sig}")
    return StageRef(raw, fn, accepts_ctx=two_arg)


def _binds(sig: inspect.Signature, *args: object) -> bool:
    try:
        sig.bind(*args)
    except TypeError:
        return False
    return True


def _load_stage_module(raw: Any, base_dir: Path) -> ModuleType:
    if not isinstance(raw, str):
        raise ConfigError("`from` must be a string.")
    if raw.startswith(".") or raw.startswith("/") or "/" in raw:
        path = Path(raw)
        if not path.is_absolute():
            path = base_dir / path
        candidates = [path]
        if path.suffix == "":
            candidates = [path.with_suffix(".py"), path / "__init__.py"]
        for candidate in candidates:
            if candidate.exists():
                return _module_from_file(candidate.resolve())
        raise ConfigError(f"Cannot find stage module at `{raw}`.")
    try:
        return importlib.import_module(raw)
    except ImportError as exc:
        raise ConfigError(f"Cannot import stage module `{raw}`: {exc}") from exc


def _module_from_file(path: Path) -> ModuleType:
    module_name = f"mine_trace_user_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"Cannot load stage module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"Error importing stage module {path}: {exc}") from exc
    return module


def _normalize_path(raw: Any, base_dir: Path, where: str) -> Path:
    if not isinstance(raw, str):
        raise ConfigError(f"{where} path must be a string.")
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return Path(os.path.realpath(path))


def _sort_pipelines(pipelines: tuple[PipelineSpec, ...]) -> tuple[PipelineSpec, ...]:
    producer: dict[Path, int] = {}
    for i, pipeline in enumerate(pipelines):
        for path in (pipeline.output, *_collect_tee_paths(pipeline.steps)):
            if path in producer:
                raise ConfigError(f"Multiple pipeline outputs or tee targets write {path}.")
            producer[path] = i

    deps: dict[int, set[int]] = {i: set() for i in range(len(pipelines))}
    for i, pipeline in enumerate(pipelines):
        for input_path in pipeline.inputs:
            dep = producer.get(input_path)
            if dep is None:
                if not input_path.exists():
                    raise ConfigError(f"Input file does not exist: {input_path}")
            elif dep != i:
                deps[i].add(dep)

    ordered: list[int] = []
    visiting: list[int] = []
    visited: set[int] = set()

    def visit(i: int) -> None:
        if i in visited:
            return
        if i in visiting:
            cycle = visiting[visiting.index(i) :] + [i]
            rendered = " -> ".join(str(pipelines[j].output) for j in cycle)
            raise ConfigError(f"Pipeline cycle detected: {rendered}")
        visiting.append(i)
        for dep in sorted(deps[i]):
            visit(dep)
        visiting.pop()
        visited.add(i)
        ordered.append(i)

    for i in range(len(pipelines)):
        visit(i)
    return tuple(pipelines[i] for i in ordered)


def _collect_tee_paths(steps: tuple[Step, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for step in steps:
        if step.kind == "tee" and step.path is not None:
            paths.append(step.path)
        elif step.kind == "group" and step.group is not None:
            paths.extend(_collect_tee_paths(step.group.pipeline))
    return tuple(paths)

