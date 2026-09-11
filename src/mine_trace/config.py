from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, get_origin

import yaml

from mine_trace.errors import ConfigError
from mine_trace.model import (
    DEFAULT_LM_MODEL,
    DEFAULT_LM_MAX_CONCURRENT,
    DEFAULT_LM_PROVIDER,
    Config,
    Defaults,
    GroupSpec,
    LmField,
    LmSignature,
    PipelineSpec,
    StageRef,
    Step,
)


_KNOWN_TOP_LEVEL = {"from", "defaults", "id_field", "pipelines", "concurrency", "lm"}
_KNOWN_LM_KEYS = {"default", "max_concurrent", "provider"}
_KNOWN_STEP_OPTIONS = {"stage", "lm"}
_KNOWN_PER_STEP_LM_KEYS = {"model"}
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
    lm_provider, lm_default_model, lm_max_concurrent = _parse_lm_config(raw.get("lm"))
    id_field = raw.get("id_field")
    if id_field is not None and not isinstance(id_field, str):
        raise ConfigError("`id_field` must be a string when provided.")

    pipelines_raw = raw["pipelines"]
    if not isinstance(pipelines_raw, list) or not pipelines_raw:
        raise ConfigError("`pipelines` must be a non-empty list.")

    pipelines = tuple(
        _parse_pipeline(block, i, base_dir, module, defaults, lm_provider, lm_default_model)
        for i, block in enumerate(pipelines_raw)
    )
    pipelines = _sort_pipelines(pipelines)
    return Config(
        config_path,
        base_dir,
        module,
        defaults,
        id_field,
        lm_provider,
        lm_default_model,
        lm_max_concurrent,
        pipelines,
    )


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


def _parse_lm_config(raw: Any) -> tuple[str, str, int]:
    if raw is None:
        return DEFAULT_LM_PROVIDER, DEFAULT_LM_MODEL, DEFAULT_LM_MAX_CONCURRENT
    if not isinstance(raw, dict):
        raise ConfigError("`lm` must be a mapping.")
    unknown = set(raw) - _KNOWN_LM_KEYS
    if unknown:
        raise ConfigError(f"Unsupported post-MVP `lm` key(s): {sorted(unknown)}")
    provider = raw.get("provider", DEFAULT_LM_PROVIDER)
    if not isinstance(provider, str):
        raise ConfigError("`lm.provider` must be a string when provided.")
    if provider != "openrouter":
        raise ConfigError("Only `lm.provider: openrouter` is supported in the LM MVP.")
    default = raw.get("default")
    if default is None:
        default = DEFAULT_LM_MODEL
    if default is not None and not isinstance(default, str):
        raise ConfigError("`lm.default` must be a string when provided.")
    max_concurrent = raw.get("max_concurrent", DEFAULT_LM_MAX_CONCURRENT)
    if not isinstance(max_concurrent, int) or max_concurrent <= 0:
        raise ConfigError("`lm.max_concurrent` must be a positive integer when provided.")
    return provider, default, max_concurrent


def _parse_pipeline(
    raw: Any,
    source_index: int,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
    lm_provider: str,
    lm_default_model: str,
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
    steps = _parse_steps(raw["pipeline"], base_dir, module, defaults, lm_provider, lm_default_model, f"pipeline {source_index + 1}")
    return PipelineSpec(inputs, output, steps, source_index)


def _parse_steps(
    raw: Any,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
    lm_provider: str,
    lm_default_model: str,
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
            steps.append(Step(kind=key, stage=_resolve_stage(value, module, lm_provider, lm_default_model, role=key)))
        elif key == "tee":
            steps.append(Step(kind="tee", path=_normalize_path(value, base_dir, f"{where} tee")))
        else:
            steps.append(
                Step(
                    kind="group",
                    group=_parse_group(value, base_dir, module, defaults, lm_provider, lm_default_model, f"{where} group step {idx}"),
                )
            )
    return tuple(steps)


def _parse_group(
    raw: Any,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
    lm_provider: str,
    lm_default_model: str,
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
    expand = _resolve_stage(raw["expand"], module, lm_provider, lm_default_model, role="expand") if kind == "expand" else None
    by: str | StageRef | None = None
    every: int | None = None
    if kind == "by":
        if isinstance(raw["by"], dict):
            by = _resolve_stage(raw["by"], module, lm_provider, lm_default_model, role="by")
        elif not isinstance(raw["by"], str):
            raise ConfigError(f"{where} `by` must be a field or stage name.")
        else:
            by = _resolve_stage(raw["by"], module, lm_provider, lm_default_model, role="by", missing_ok=True) or raw["by"]
    elif kind == "every":
        every = raw["every"]
        if not isinstance(every, int) or every <= 0:
            raise ConfigError(f"{where} `every` must be a positive integer.")

    child_steps = _parse_steps(raw.get("pipeline", []), base_dir, module, defaults, lm_provider, lm_default_model, where)
    reduce = _resolve_stage(raw["reduce"], module, lm_provider, lm_default_model, role="reduce")
    return GroupSpec(kind, expand, by, every, child_steps, reduce, on_child_error, on_group_incomplete, ordered)


def _resolve_stage(
    raw: Any,
    module: ModuleType,
    lm_provider: str,
    lm_default_model: str,
    role: str,
    missing_ok: bool = False,
) -> StageRef | None:
    name, step_model = _parse_stage_selector(raw)
    if not hasattr(module, name):
        if missing_ok:
            return None
        raise ConfigError(f"Unknown stage `{name}`.")
    fn = getattr(module, name)
    lm_signature = _lm_signature(fn)
    if lm_signature is not None:
        model = step_model or lm_default_model
        _validate_lm_stage(name, role, lm_signature)
        return StageRef(name, fn, kind="lm", role=role, lm_signature=lm_signature, lm_model=model, lm_provider=lm_provider)
    if step_model is not None:
        raise ConfigError(f"Python stage `{name}` cannot declare `lm` options.")
    if not callable(fn):
        raise ConfigError(f"Stage `{name}` is not callable.")
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        raise ConfigError(f"Stage `{name}` uses a post-MVP callable shape.")

    sig = inspect.signature(fn)
    one_arg = _binds(sig, object())
    two_arg = _binds(sig, object(), object())
    if not one_arg and not two_arg:
        raise ConfigError(f"Stage `{name}` cannot be called as (record) or (record, ctx): {sig}")
    return StageRef(name, fn, kind="python", role=role, accepts_ctx=two_arg, is_async=inspect.iscoroutinefunction(fn))


def _parse_stage_selector(raw: Any) -> tuple[str, str | None]:
    if isinstance(raw, str):
        return raw, None
    if not isinstance(raw, dict):
        raise ConfigError("Stage names must be strings or mappings with `stage`.")
    unknown = set(raw) - _KNOWN_STEP_OPTIONS
    if unknown:
        raise ConfigError(f"Unknown stage option(s): {sorted(unknown)}")
    if "stage" not in raw:
        raise ConfigError("Expanded stage form must include `stage`.")
    name = raw["stage"]
    if not isinstance(name, str):
        raise ConfigError("`stage` must be a string.")
    lm_raw = raw.get("lm")
    if lm_raw is None:
        return name, None
    if not isinstance(lm_raw, dict):
        raise ConfigError("Per-step `lm` must be a mapping.")
    unknown_lm = set(lm_raw) - _KNOWN_PER_STEP_LM_KEYS
    if unknown_lm:
        raise ConfigError(f"Unsupported post-MVP per-step `lm` key(s): {sorted(unknown_lm)}")
    model = lm_raw.get("model")
    if not isinstance(model, str):
        raise ConfigError("Per-step `lm.model` must be a string.")
    return name, model


def _lm_signature(target: Any) -> LmSignature | None:
    signature = getattr(target, "signature", None)
    if signature is None or not hasattr(signature, "input_fields") or not hasattr(signature, "output_fields"):
        return None
    return LmSignature(
        input_fields=_lm_fields(signature, "input_fields"),
        output_fields=_lm_fields(signature, "output_fields"),
    )


def _lm_fields(signature: Any, attribute: str) -> tuple[LmField, ...]:
    raw_fields = getattr(signature, attribute)
    annotations = getattr(signature, "annotations", None) or getattr(signature, "__annotations__", {})
    fields: list[LmField] = []
    if isinstance(raw_fields, dict):
        iterable = raw_fields.items()
    else:
        iterable = [(None, item) for item in raw_fields]
    for fallback_name, field in iterable:
        name = fallback_name if fallback_name is not None else getattr(field, "name", field)
        if not isinstance(name, str):
            raise ConfigError(f"LM signature `{attribute}` contains a field without a string name.")
        annotation = (
            getattr(field, "annotation", None)
            or getattr(field, "type", None)
            or getattr(field, "annotation_", None)
            or annotations.get(name)
        )
        fields.append(LmField(name, annotation))
    return tuple(fields)


def _validate_lm_stage(name: str, role: str, signature: LmSignature) -> None:
    if role == "stage":
        raise ConfigError(f"LM stage `{name}` cannot be used with `stage`; use map, filter, expand, or reduce.")
    if role == "filter":
        bool_outputs = [field for field in signature.output_fields if _is_bool_annotation(field.annotation)]
        if len(bool_outputs) != 1 or len(signature.output_fields) != 1:
            raise ConfigError(f"LM filter `{name}` must have exactly one bool output field.")
    elif role == "expand":
        list_outputs = [field for field in signature.output_fields if _is_list_annotation(field.annotation)]
        if len(list_outputs) != 1 or len(signature.output_fields) != 1:
            raise ConfigError(f"LM expand `{name}` must have exactly one list output field.")
    elif role == "reduce":
        list_inputs = [field for field in signature.input_fields if _is_list_annotation(field.annotation)]
        if len(list_inputs) != 1:
            raise ConfigError(f"LM reduce `{name}` must have exactly one list input field.")
    elif role == "by":
        raise ConfigError(f"LM stage `{name}` cannot be used as `by`; use a Python key stage.")


def _is_bool_annotation(annotation: Any) -> bool:
    return annotation is bool or annotation == "bool"


def _is_list_annotation(annotation: Any) -> bool:
    if annotation is list or annotation == "list":
        return True
    if isinstance(annotation, str):
        return annotation.startswith("list[") or annotation.startswith("typing.List[")
    return get_origin(annotation) is list


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
