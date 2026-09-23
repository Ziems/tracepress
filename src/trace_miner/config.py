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

from trace_miner.errors import ConfigError
from trace_miner.model import (
    DEFAULT_LM_CACHE,
    DEFAULT_LM_MODEL,
    DEFAULT_LM_MAX_CONCURRENT,
    DEFAULT_LM_MAX_TOKENS,
    DEFAULT_LM_PROVIDER,
    DEFAULT_REDUCE_LM_MODEL,
    LM_MODEL_ROLES,
    LM_PROVIDERS,
    Config,
    Defaults,
    GroupSpec,
    LmChoice,
    LmField,
    LmModels,
    LmSignature,
    PipelineSpec,
    StageRef,
    Step,
)


_KNOWN_TOP_LEVEL = {"from", "defaults", "id_field", "pipelines", "concurrency", "lm"}
_KNOWN_LM_KEYS = {"cache", "default", "max_concurrent", "max_tokens", "provider", *LM_MODEL_ROLES}
_KNOWN_STEP_OPTIONS = {"stage", "lm"}
_KNOWN_PER_STEP_LM_KEYS = {"model", "provider"}
_KNOWN_LM_CHOICE_KEYS = {"model", "provider"}
_KNOWN_DEFAULTS = {"on_child_error", "ordered"}
_STEP_KEYS = {"map", "filter", "stage", "tee", "alert", "group"}
_GROUP_KEYS = {
    "expand",
    "pipeline",
    "reduce",
    "on_child_error",
    "ordered",
    "sidechannels",
}
_CHILD_ERROR_POLICIES = {"fail", "partial", "skip"}


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
    lm_models, lm_max_concurrent, lm_cache, lm_max_tokens = _parse_lm_config(raw.get("lm"))
    id_field = raw.get("id_field")
    if id_field is not None and not isinstance(id_field, str):
        raise ConfigError("`id_field` must be a string when provided.")

    pipelines_raw = raw["pipelines"]
    if not isinstance(pipelines_raw, list) or not pipelines_raw:
        raise ConfigError("`pipelines` must be a non-empty list.")

    pipelines = tuple(
        _parse_pipeline(block, i, base_dir, module, defaults, lm_models)
        for i, block in enumerate(pipelines_raw)
    )
    pipelines = _sort_pipelines(pipelines)
    _validate_pipeline_names(pipelines)
    return Config(
        config_path,
        base_dir,
        module,
        defaults,
        id_field,
        lm_models.default.provider,
        lm_models.default.model,
        lm_max_concurrent,
        lm_cache,
        lm_max_tokens,
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
    ordered = raw.get("ordered", True)
    if on_child_error not in _CHILD_ERROR_POLICIES:
        raise ConfigError("`defaults.on_child_error` must be fail, partial, or skip.")
    if not isinstance(ordered, bool):
        raise ConfigError("`defaults.ordered` must be a boolean.")
    return Defaults(on_child_error, ordered)


def _parse_lm_config(raw: Any) -> tuple[LmModels, int, bool, int | None]:
    if raw is None:
        return (
            _lm_models(DEFAULT_LM_PROVIDER, None, {}, default_was_set=False),
            DEFAULT_LM_MAX_CONCURRENT,
            DEFAULT_LM_CACHE,
            DEFAULT_LM_MAX_TOKENS,
        )
    if not isinstance(raw, dict):
        raise ConfigError("`lm` must be a mapping.")
    unknown = set(raw) - _KNOWN_LM_KEYS
    if unknown:
        raise ConfigError(f"Unsupported post-MVP `lm` key(s): {sorted(unknown)}")
    provider = _parse_provider(raw.get("provider", DEFAULT_LM_PROVIDER), "`lm.provider`")
    default_was_set = "default" in raw
    default = (
        _parse_lm_choice(raw["default"], where="lm.default", fallback_provider=provider, fallback_model=DEFAULT_LM_MODEL)
        if default_was_set
        else LmChoice(provider, DEFAULT_LM_MODEL)
    )
    by_role: dict[str, LmChoice] = {}
    for role in LM_MODEL_ROLES:
        if role not in raw:
            continue
        by_role[role] = _parse_lm_choice(
            raw[role],
            where=f"lm.{role}",
            fallback_provider=default.provider,
            fallback_model=default.model,
        )
    max_concurrent = raw.get("max_concurrent", DEFAULT_LM_MAX_CONCURRENT)
    if not isinstance(max_concurrent, int) or max_concurrent <= 0:
        raise ConfigError("`lm.max_concurrent` must be a positive integer when provided.")
    cache = raw.get("cache", DEFAULT_LM_CACHE)
    if not isinstance(cache, bool):
        raise ConfigError("`lm.cache` must be a boolean when provided.")
    max_tokens = raw.get("max_tokens", DEFAULT_LM_MAX_TOKENS)
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        raise ConfigError("`lm.max_tokens` must be a positive integer when provided.")
    return (
        _lm_models(provider, default, by_role, default_was_set=default_was_set),
        max_concurrent,
        cache,
        max_tokens,
    )


def _lm_models(
    provider: str,
    default: LmChoice | None,
    by_role: dict[str, LmChoice],
    *,
    default_was_set: bool,
) -> LmModels:
    choice = default or LmChoice(provider, DEFAULT_LM_MODEL)
    resolved = dict(by_role)
    if not default_was_set and "reduce" not in resolved:
        resolved["reduce"] = LmChoice(choice.provider, DEFAULT_REDUCE_LM_MODEL)
    return LmModels(choice, resolved)


def _parse_lm_choice(raw: Any, *, where: str, fallback_provider: str, fallback_model: str) -> LmChoice:
    if isinstance(raw, str):
        if not raw:
            raise ConfigError(f"`{where}` must be a non-empty string when provided.")
        return LmChoice(fallback_provider, raw)
    if not isinstance(raw, dict):
        raise ConfigError(f"`{where}` must be a model id or a mapping with `provider` and `model`.")
    unknown = set(raw) - _KNOWN_LM_CHOICE_KEYS
    if unknown:
        raise ConfigError(f"Unknown `{where}` key(s): {sorted(unknown)}")
    if "provider" not in raw and "model" not in raw:
        raise ConfigError(f"`{where}` needs `provider` or `model`.")
    provider = _parse_provider(raw.get("provider", fallback_provider), f"`{where}.provider`")
    model = raw.get("model", fallback_model)
    if not isinstance(model, str) or not model:
        raise ConfigError(f"`{where}.model` must be a non-empty string when provided.")
    return LmChoice(provider, model)


def _parse_provider(provider: Any, where: str) -> str:
    if not isinstance(provider, str):
        raise ConfigError(f"{where} must be a string when provided.")
    if provider not in LM_PROVIDERS:
        supported = ", ".join(sorted(LM_PROVIDERS))
        raise ConfigError(f"{where} must be one of: {supported}.")
    return provider


def _parse_pipeline(
    raw: Any,
    source_index: int,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
    lm_models: LmModels,
) -> PipelineSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"Pipeline {source_index + 1} must be a mapping.")
    required = {"in", "out", "pipeline"}
    missing = required - set(raw)
    if missing:
        raise ConfigError(f"Pipeline {source_index + 1} missing key(s): {sorted(missing)}")
    unknown = set(raw) - required - {"name"}
    if unknown:
        raise ConfigError(f"Pipeline {source_index + 1} unknown key(s): {sorted(unknown)}")
    name = _parse_pipeline_name(raw.get("name"), source_index)

    inputs_raw = raw["in"]
    input_items = inputs_raw if isinstance(inputs_raw, list) else [inputs_raw]
    if not input_items:
        raise ConfigError(f"Pipeline {source_index + 1} must have at least one input.")
    inputs = tuple(_normalize_path(item, base_dir, f"pipeline {source_index + 1} input") for item in input_items)
    output = _normalize_path(raw["out"], base_dir, f"pipeline {source_index + 1} output")
    steps = _parse_steps(raw["pipeline"], base_dir, module, defaults, lm_models, f"pipeline {source_index + 1}")
    return PipelineSpec(inputs, output, steps, source_index, name)


def _parse_steps(
    raw: Any,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
    lm_models: LmModels,
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
        if key in {"map", "filter", "stage", "alert"}:
            steps.append(Step(kind=key, stage=_resolve_stage(value, module, lm_models, role=key)))
        elif key == "tee":
            steps.append(Step(kind="tee", path=_normalize_path(value, base_dir, f"{where} tee")))
        else:
            steps.append(
                Step(
                    kind="group",
                    group=_parse_group(value, base_dir, module, defaults, lm_models, f"{where} group step {idx}"),
                )
            )
    return tuple(steps)


def _parse_group(
    raw: Any,
    base_dir: Path,
    module: ModuleType,
    defaults: Defaults,
    lm_models: LmModels,
    where: str,
) -> GroupSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping.")
    if "by" in raw or "every" in raw:
        raise ConfigError(f"{where} only supports `expand`.")
    if "on_group_incomplete" in raw:
        raise ConfigError(f"{where} does not support `on_group_incomplete`.")
    unknown = set(raw) - _GROUP_KEYS
    if unknown:
        raise ConfigError(f"Unknown group key(s) in {where}: {sorted(unknown)}")
    if "sidechannels" in raw:
        raise ConfigError("`sidechannels` is post-MVP and is not supported yet.")
    if "expand" not in raw:
        raise ConfigError(f"{where} must declare `expand`.")
    if "reduce" not in raw:
        raise ConfigError(f"{where} missing required key: `reduce`.")

    on_child_error = raw.get("on_child_error", defaults.on_child_error)
    ordered = raw.get("ordered", defaults.ordered)
    if on_child_error not in _CHILD_ERROR_POLICIES:
        raise ConfigError(f"{where} `on_child_error` must be fail, partial, or skip.")
    if not isinstance(ordered, bool):
        raise ConfigError(f"{where} `ordered` must be a boolean.")

    expand = _resolve_stage(raw["expand"], module, lm_models, role="expand")
    assert expand is not None
    child_steps = _parse_steps(raw.get("pipeline", []), base_dir, module, defaults, lm_models, where)
    reduce = _resolve_stage(raw["reduce"], module, lm_models, role="reduce")
    assert reduce is not None
    return GroupSpec(expand, child_steps, reduce, on_child_error, ordered)


def _resolve_stage(
    raw: Any,
    module: ModuleType,
    lm_models: LmModels,
    role: str,
    missing_ok: bool = False,
) -> StageRef | None:
    name, step_model, step_provider = _parse_stage_selector(raw)
    if not hasattr(module, name):
        if missing_ok:
            return None
        raise ConfigError(f"Unknown stage `{name}`.")
    fn = getattr(module, name)
    if _is_dspy_module(fn):
        lm_signature = _lm_signature(fn)
        if lm_signature is None:
            raise ConfigError(
                f"DSPy stage `{name}` must expose a signature with `input_fields` and `output_fields`."
            )
        if getattr(fn, "uses_jev", False):
            if step_model is not None or step_provider is not None:
                raise ConfigError(f"Jev stage `{name}` cannot declare `lm` options.")
            if role == "expand":
                raise ConfigError(f"Jev stage `{name}` cannot be used as `expand`.")
            _validate_lm_stage(name, role, lm_signature)
            return StageRef(name, fn, kind="lm", role=role, lm_signature=lm_signature, uses_jev=True)
        choice = lm_models.for_role(role)
        model = step_model or choice.model
        provider = step_provider or choice.provider
        if not model:
            raise ConfigError(f"LM stage `{name}` needs a model.")
        _validate_lm_stage(name, role, lm_signature)
        return StageRef(name, fn, kind="lm", role=role, lm_signature=lm_signature, lm_model=model, lm_provider=provider)
    if _is_dspy_signature(fn):
        raise ConfigError(f"Stage `{name}` is a DSPy signature. Use `dspy.Predict` or another `dspy.Module`.")
    if _lm_signature(fn) is not None:
        raise ConfigError(f"Stage `{name}` is not a DSPy module. LM stages must be `dspy.Predict` or another `dspy.Module`.")
    if step_model is not None or step_provider is not None:
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


def _parse_stage_selector(raw: Any) -> tuple[str, str | None, str | None]:
    if isinstance(raw, str):
        return raw, None, None
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
        return name, None, None
    if not isinstance(lm_raw, dict):
        raise ConfigError("Per-step `lm` must be a mapping.")
    unknown_lm = set(lm_raw) - _KNOWN_PER_STEP_LM_KEYS
    if unknown_lm:
        raise ConfigError(f"Unsupported post-MVP per-step `lm` key(s): {sorted(unknown_lm)}")
    if "model" not in lm_raw and "provider" not in lm_raw:
        raise ConfigError("Per-step `lm` needs `provider` or `model`.")
    model = lm_raw.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise ConfigError("Per-step `lm.model` must be a non-empty string when provided.")
    provider = lm_raw.get("provider")
    if provider is not None:
        provider = _parse_provider(provider, "Per-step `lm.provider`")
    return name, model, provider


def _is_dspy_module(target: Any) -> bool:
    import dspy

    return isinstance(target, dspy.Module)


def _is_dspy_signature(target: Any) -> bool:
    import dspy

    if isinstance(target, dspy.Signature):
        return True
    return isinstance(target, type) and issubclass(target, dspy.Signature)


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
        raise ConfigError(f"LM stage `{name}` cannot be used with `stage`; use map, filter, alert, expand, or reduce.")
    if role == "alert":
        bool_outputs = [field for field in signature.output_fields if _is_bool_annotation(field.annotation)]
        if len(bool_outputs) != 1:
            raise ConfigError(f"LM alert `{name}` must have exactly one bool output field.")
        return
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
    module_name = f"trace_miner_user_{abs(hash(path))}"
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


def _parse_pipeline_name(raw: Any, source_index: int) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ConfigError(f"Pipeline {source_index + 1} `name` must be a non-empty string.")
    return raw


def _validate_pipeline_names(pipelines: tuple[PipelineSpec, ...]) -> None:
    seen: dict[str, int] = {}
    for pipeline in pipelines:
        if pipeline.name is None:
            continue
        previous = seen.get(pipeline.name)
        if previous is not None:
            raise ConfigError(f"Duplicate pipeline name `{pipeline.name}`.")
        seen[pipeline.name] = pipeline.source_index


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
