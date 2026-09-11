from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, get_origin
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mine_trace.config import load_config
from mine_trace.errors import PipelineExecutionError
from mine_trace.model import (
    Config,
    Failure,
    GroupSpec,
    PipelineSpec,
    RunManifest,
    RunResult,
    RunStats,
    StageRef,
    Step,
)


@dataclass
class _Envelope:
    record: Any
    path: tuple[int, ...]
    group: _GroupState | None = None


@dataclass
class _CallContext:
    runtime: _Runtime
    stage_name: str
    source: _Envelope | None
    parent: Any
    group_id: str | None
    key: Any
    index: int | None
    path: tuple[int, ...]
    depth: int
    failures: list[Failure]
    emitted: list[Any] = field(default_factory=list)

    def emit(self, record: Any) -> None:
        self.emitted.append(record)
        self.runtime.stats.emitted += 1

    def side(self, channel: str, record: Any) -> None:
        raise PipelineExecutionError("`ctx.side` is post-MVP and is not supported yet.")

    @property
    def log(self) -> "_Logger":
        return _Logger(self.stage_name, self.group_id)


@dataclass
class _Logger:
    stage_name: str
    group_id: str | None

    def info(self, message: str) -> None:
        print(f"[mine-trace] {self.stage_name} {self.group_id or '-'}: {message}")


class _GroupSkipped(Exception):
    pass


@dataclass
class _GroupState:
    runtime: _Runtime
    spec: GroupSpec
    parent_env: _Envelope | None
    key: Any
    depth: int
    group_id: str
    outstanding: int = 1
    fired: bool = False
    skipped: bool = False
    children: list[_Envelope] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    reduce_outputs: list[_Envelope] = field(default_factory=list)

    @property
    def parent_record(self) -> Any:
        if self.parent_env is not None:
            return self.parent_env.record
        if self.spec.kind == "by" and isinstance(self.spec.by, str):
            return {self.spec.by: self.key}
        return None

    def increment(self, count: int = 1) -> None:
        self.outstanding += count

    def decrement(self, count: int = 1) -> None:
        self.outstanding -= count
        if self.outstanding < 0:
            raise PipelineExecutionError(f"Group {self.group_id} accounting went negative.")
        if self.outstanding == 0:
            self.reduce()

    def adjust_for_outputs(self, count: int) -> None:
        if count == 0:
            self.decrement()
        elif count > 1:
            self.increment(count - 1)

    def adjust_for_replacement(self, input_count: int, output_count: int) -> None:
        delta = output_count - input_count
        if delta > 0:
            self.increment(delta)
        elif delta < 0:
            self.decrement(-delta)

    def arrive(self, env: _Envelope) -> None:
        self.children.append(env)
        self.runtime.stats.retired += 1
        self.decrement()

    def fail_child(self, env: _Envelope, stage_name: str, error: BaseException) -> list[_Envelope]:
        self.runtime.stats.failed += 1
        identity = self.runtime.record_identity(env.record, env.path)
        failure = Failure(stage_name, identity, self.group_id, str(error))
        if self.spec.on_child_error == "fail":
            raise PipelineExecutionError(
                f"Stage `{stage_name}` failed for record {identity} in group {self.group_id}: {error}"
            ) from error
        if self.spec.on_child_error == "partial":
            self.failures.append(failure)
            self.runtime.stats.retired += 1
            self.decrement()
            return []
        self.skip(failure)
        raise _GroupSkipped()

    def skip(self, failure: Failure) -> None:
        if self.skipped:
            return
        self.skipped = True
        self.fired = True
        self.runtime.stats.groups_skipped += 1
        self.runtime.manifest.skipped_groups.append(
            {
                "group_id": self.group_id,
                "parent": self.runtime.record_identity(
                    self.parent_record,
                    self.parent_env.path if self.parent_env is not None else (),
                ),
                "failing_child": failure.record_identity,
                "stage": failure.stage,
                "error": failure.error,
            }
        )

    def close_producer(self) -> None:
        self.decrement()

    def reduce(self) -> None:
        if self.fired:
            return
        self.fired = True
        if self.skipped:
            return
        children = self.children
        if self.spec.ordered:
            children = sorted(children, key=lambda env: env.path)
        child_records = [env.record for env in children]
        ctx_path = self.parent_env.path if self.parent_env is not None else ()
        outputs = self.runtime.call_stage(
            self.spec.reduce,
            child_records,
            source=self.parent_env,
            parent=self.parent_record,
            group_id=self.group_id,
            key=self.key,
            index=None,
            path=ctx_path,
            depth=self.depth,
            failures=self.failures,
        )
        base_path = self.parent_env.path if self.parent_env is not None else (self.runtime.next_root_index(),)
        self.reduce_outputs = [
            _Envelope(record, base_path + (i,), None)
            for i, record in enumerate(outputs, start=1)
        ]
        self.runtime.stats.groups_reduced += 1


class _Runtime:
    def __init__(self, config: Config):
        self.config = config
        self.stats = RunStats()
        self.manifest = RunManifest()
        self._group_counter = 0
        self._root_counter = 0
        self._open_groups: list[_GroupState] = []

    def run(self) -> RunResult:
        for pipeline in self.config.pipelines:
            self._run_pipeline(pipeline)
        leaks = [group for group in self._open_groups if not group.fired]
        if leaks:
            details = ", ".join(f"{group.group_id} count={group.outstanding}" for group in leaks)
            raise PipelineExecutionError(f"Open groups remain at shutdown: {details}")
        return RunResult(self.stats, self.manifest)

    def _run_pipeline(self, pipeline: PipelineSpec) -> None:
        envelopes = list(self._read_inputs(pipeline.inputs))
        outputs = self.apply_steps(envelopes, pipeline.steps, active_group=None)
        self._write_jsonl(pipeline.output, [env.record for env in outputs])
        self.stats.retired += len(outputs)

    def _read_inputs(self, paths: Iterable[Path]) -> Iterable[_Envelope]:
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise PipelineExecutionError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                    self.stats.records_read += 1
                    yield _Envelope(record, (self.next_root_index(),), None)

    def _write_jsonl(self, path: Path, records: Iterable[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                handle.write("\n")

    def apply_steps(
        self,
        envelopes: list[_Envelope],
        steps: tuple[Step, ...],
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        current = envelopes
        for step in steps:
            if active_group is not None and active_group.skipped:
                raise _GroupSkipped()
            if step.kind in {"map", "filter", "stage"}:
                current = self._apply_stage_step(current, step, active_group)
            elif step.kind == "tee":
                assert step.path is not None
                self._append_jsonl(step.path, [env.record for env in current])
            elif step.kind == "group":
                assert step.group is not None
                current = self._apply_group_step(current, step.group, active_group)
            else:
                raise AssertionError(f"Unhandled step kind: {step.kind}")
        return current

    def _apply_stage_step(
        self,
        envelopes: list[_Envelope],
        step: Step,
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        assert step.stage is not None
        output: list[_Envelope] = []
        for env in envelopes:
            failed_under_policy = False
            try:
                emitted = self.call_stage_for_env(step.stage, env)
            except Exception as exc:
                if active_group is None:
                    raise
                emitted = active_group.fail_child(env, step.stage.name, exc)
                failed_under_policy = True
            if failed_under_policy:
                continue
            self._assert_emit_count(step, env, emitted)
            if active_group is not None:
                active_group.adjust_for_outputs(len(emitted))
            if not emitted:
                if active_group is None:
                    self.stats.dropped += 1
                else:
                    self.stats.dropped += 1
                continue
            for i, record in enumerate(emitted, start=1):
                output.append(_Envelope(record, env.path + (i,), active_group))
        return output

    def _apply_group_step(
        self,
        envelopes: list[_Envelope],
        spec: GroupSpec,
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        if spec.kind == "expand":
            outputs = []
            for env in envelopes:
                nested_outputs = self._process_expand_group(spec, env, depth=self._depth_for(active_group))
                if active_group is not None:
                    active_group.adjust_for_outputs(len(nested_outputs))
                outputs.extend(_Envelope(out.record, out.path, active_group) for out in nested_outputs)
            return outputs
        if spec.kind == "by":
            return self._process_by_groups(spec, envelopes, active_group)
        if spec.kind == "every":
            return self._process_every_groups(spec, envelopes, active_group)
        raise AssertionError(f"Unhandled group kind: {spec.kind}")

    def _process_expand_group(self, spec: GroupSpec, env: _Envelope, depth: int) -> list[_Envelope]:
        group = self._open_group(spec, env, key=None, depth=depth)
        try:
            assert spec.expand is not None
            emitted = self.call_stage_for_env(
                spec.expand,
                env,
                parent=env.record,
                group_id=group.group_id,
                key=None,
                depth=depth,
                failures=group.failures,
            )
            child_envs = []
            for i, record in enumerate(emitted, start=1):
                group.increment()
                child_envs.append(_Envelope(record, env.path + (i,), group))
            group.close_producer()
            if group.skipped:
                return []
            self._run_group_children(group, child_envs)
        except _GroupSkipped:
            return []
        except Exception as exc:
            if isinstance(exc, PipelineExecutionError):
                raise
            raise PipelineExecutionError(
                f"Group {group.group_id} expand stage `{spec.expand.name if spec.expand else '<unknown>'}` failed: {exc}"
            ) from exc
        return group.reduce_outputs

    def _process_by_groups(
        self,
        spec: GroupSpec,
        envelopes: list[_Envelope],
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        grouped: dict[Any, list[_Envelope]] = {}
        order: list[Any] = []
        for env in envelopes:
            key = self._group_key(spec, env)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(env)

        outputs: list[_Envelope] = []
        for key in order:
            group = self._open_group(spec, None, key=key, depth=self._depth_for(active_group))
            child_envs: list[_Envelope] = []
            for i, env in enumerate(grouped[key], start=1):
                group.increment()
                child_envs.append(_Envelope(env.record, env.path + (i,), group))
            group.close_producer()
            try:
                self._run_group_children(group, child_envs)
            except _GroupSkipped:
                nested_outputs = []
            else:
                nested_outputs = group.reduce_outputs
            outputs.extend(_Envelope(out.record, out.path, active_group) for out in nested_outputs)
        if active_group is not None:
            active_group.adjust_for_replacement(len(envelopes), len(outputs))
        return outputs

    def _process_every_groups(
        self,
        spec: GroupSpec,
        envelopes: list[_Envelope],
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        assert spec.every is not None
        outputs: list[_Envelope] = []
        for start in range(0, len(envelopes), spec.every):
            window = envelopes[start : start + spec.every]
            group = self._open_group(spec, None, key=start // spec.every + 1, depth=self._depth_for(active_group))
            child_envs = []
            for i, env in enumerate(window, start=1):
                group.increment()
                child_envs.append(_Envelope(env.record, env.path + (i,), group))
            group.close_producer()
            try:
                self._run_group_children(group, child_envs)
            except _GroupSkipped:
                nested_outputs = []
            else:
                nested_outputs = group.reduce_outputs
            outputs.extend(_Envelope(out.record, out.path, active_group) for out in nested_outputs)
        if active_group is not None:
            active_group.adjust_for_replacement(len(envelopes), len(outputs))
        return outputs

    def _run_group_children(self, group: _GroupState, child_envs: list[_Envelope]) -> None:
        if group.fired:
            return
        outputs = self.apply_steps(child_envs, group.spec.pipeline, active_group=group)
        for env in outputs:
            if group.skipped:
                raise _GroupSkipped()
            group.arrive(env)

    def _group_key(self, spec: GroupSpec, env: _Envelope) -> Any:
        if isinstance(spec.by, StageRef):
            emitted = self.call_stage_for_env(spec.by, env)
            if len(emitted) != 1:
                raise PipelineExecutionError(
                    f"`by` stage `{spec.by.name}` must emit exactly one key for record "
                    f"{self.record_identity(env.record, env.path)}; got {len(emitted)}."
                )
            return _hashable_key(emitted[0])
        if isinstance(env.record, dict) and spec.by in env.record:
            return _hashable_key(env.record[spec.by])
        raise PipelineExecutionError(
            f"Record {self.record_identity(env.record, env.path)} has no grouping field `{spec.by}`."
        )

    def call_stage_for_env(
        self,
        stage: StageRef,
        env: _Envelope,
        parent: Any = None,
        group_id: str | None = None,
        key: Any = None,
        depth: int = 0,
        failures: list[Failure] | None = None,
    ) -> list[Any]:
        source_group = env.group
        return self.call_stage(
            stage,
            env.record,
            source=env,
            parent=parent if parent is not None else (source_group.parent_record if source_group else None),
            group_id=group_id if group_id is not None else (source_group.group_id if source_group else None),
            key=key if key is not None else (source_group.key if source_group else None),
            index=env.path[-1] if env.path else None,
            path=env.path,
            depth=depth if source_group is None else source_group.depth,
            failures=failures if failures is not None else (source_group.failures if source_group else []),
        )

    def call_stage(
        self,
        stage: StageRef,
        arg: Any,
        source: _Envelope | None,
        parent: Any,
        group_id: str | None,
        key: Any,
        index: int | None,
        path: tuple[int, ...],
        depth: int,
        failures: list[Failure],
    ) -> list[Any]:
        ctx = _CallContext(self, stage.name, source, parent, group_id, key, index, path, depth, failures)
        if stage.kind == "lm":
            return self._call_lm_stage(stage, arg, parent, path)
        result = stage.target(arg, ctx) if stage.accepts_ctx else stage.target(arg)
        if result is not None and ctx.emitted:
            identity = self.record_identity(arg, path)
            raise PipelineExecutionError(f"Stage `{stage.name}` both returned a value and emitted for record {identity}.")
        if result is not None:
            self.stats.emitted += 1
            return [result]
        return ctx.emitted

    def _call_lm_stage(self, stage: StageRef, arg: Any, parent: Any, path: tuple[int, ...]) -> list[Any]:
        if stage.lm_signature is None:
            raise PipelineExecutionError(f"LM stage `{stage.name}` has no signature.")
        if stage.role == "reduce":
            kwargs = self._lm_reduce_kwargs(stage, arg, parent, path)
        else:
            kwargs = self._lm_record_kwargs(stage, arg, path)
        prediction = self._invoke_lm(stage, kwargs)
        outputs = self._lm_outputs(stage, prediction)
        if stage.role == "map":
            if not isinstance(arg, dict):
                raise PipelineExecutionError(f"LM map `{stage.name}` can only merge outputs into dict records.")
            self.stats.emitted += 1
            return [{**arg, **outputs}]
        if stage.role == "filter":
            keep_field = stage.lm_signature.output_fields[0].name
            self.stats.emitted += 1 if outputs[keep_field] else 0
            return [arg] if outputs[keep_field] else []
        if stage.role == "expand":
            output_field = stage.lm_signature.output_fields[0].name
            children = outputs[output_field]
            if not isinstance(children, list):
                raise PipelineExecutionError(f"LM expand `{stage.name}` output `{output_field}` must be a list.")
            self.stats.emitted += len(children)
            return children
        if stage.role == "reduce":
            parent_record = parent if isinstance(parent, dict) else {}
            self.stats.emitted += 1
            return [{**parent_record, **outputs}]
        raise PipelineExecutionError(f"LM stage `{stage.name}` cannot be used as `{stage.role}`.")

    def _lm_record_kwargs(self, stage: StageRef, record: Any, path: tuple[int, ...]) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise PipelineExecutionError(f"LM stage `{stage.name}` requires dict records for field mapping.")
        assert stage.lm_signature is not None
        kwargs = {}
        for field in stage.lm_signature.input_fields:
            if field.name not in record:
                keys = sorted(str(key) for key in record)
                raise PipelineExecutionError(
                    f"LM stage `{stage.name}` missing input field `{field.name}` for "
                    f"{self.record_identity(record, path)}; record keys: {keys}"
                )
            kwargs[field.name] = record[field.name]
        return kwargs

    def _lm_reduce_kwargs(self, stage: StageRef, children: Any, parent: Any, path: tuple[int, ...]) -> dict[str, Any]:
        if not isinstance(children, list):
            raise PipelineExecutionError(f"LM reduce `{stage.name}` expected a list of children.")
        assert stage.lm_signature is not None
        list_fields = [field for field in stage.lm_signature.input_fields if _is_list_annotation(field.annotation)]
        list_field = list_fields[0]
        parent_record = parent if isinstance(parent, dict) else {}
        kwargs = {list_field.name: _gather_children(list_field.name, children)}
        for field in stage.lm_signature.input_fields:
            if field.name == list_field.name:
                continue
            if field.name not in parent_record:
                keys = sorted(str(key) for key in parent_record)
                raise PipelineExecutionError(
                    f"LM reduce `{stage.name}` missing parent field `{field.name}` for path "
                    f"{'.'.join(str(item) for item in path) or '<root>'}; parent keys: {keys}"
                )
            kwargs[field.name] = parent_record[field.name]
        return kwargs

    def _invoke_lm(self, stage: StageRef, kwargs: dict[str, Any]) -> Any:
        target = stage.target
        if stage.lm_provider == "openrouter" and _is_dspy_target(target):
            return _call_openrouter(stage, kwargs)
        if hasattr(target, "with_model"):
            target = target.with_model(stage.lm_model)
        if callable(target):
            return target(**kwargs)
        if stage.lm_provider == "openrouter":
            return _call_openrouter(stage, kwargs)
        raise PipelineExecutionError(f"Unsupported LM provider `{stage.lm_provider}` for stage `{stage.name}`.")

    def _lm_outputs(self, stage: StageRef, prediction: Any) -> dict[str, Any]:
        assert stage.lm_signature is not None
        outputs = {}
        for field in stage.lm_signature.output_fields:
            if isinstance(prediction, dict):
                if field.name not in prediction:
                    raise PipelineExecutionError(f"LM stage `{stage.name}` missing output field `{field.name}`.")
                outputs[field.name] = prediction[field.name]
            elif hasattr(prediction, field.name):
                outputs[field.name] = getattr(prediction, field.name)
            else:
                raise PipelineExecutionError(f"LM stage `{stage.name}` missing output field `{field.name}`.")
        return outputs

    def _assert_emit_count(self, step: Step, env: _Envelope, emitted: list[Any]) -> None:
        if step.kind == "map" and len(emitted) != 1:
            raise PipelineExecutionError(
                f"`map` stage `{step.stage.name if step.stage else '<unknown>'}` emitted {len(emitted)} "
                f"records for {self.record_identity(env.record, env.path)}; expected exactly 1."
            )
        if step.kind == "filter" and len(emitted) > 1:
            raise PipelineExecutionError(
                f"`filter` stage `{step.stage.name if step.stage else '<unknown>'}` emitted {len(emitted)} "
                f"records for {self.record_identity(env.record, env.path)}; expected 0 or 1."
            )

    def _append_jsonl(self, path: Path, records: Iterable[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                handle.write("\n")

    def _open_group(self, spec: GroupSpec, parent_env: _Envelope | None, key: Any, depth: int) -> _GroupState:
        self._group_counter += 1
        group = _GroupState(self, spec, parent_env, key, depth, f"g{self._group_counter}")
        self.stats.groups_opened += 1
        self._open_groups.append(group)
        return group

    def _depth_for(self, active_group: _GroupState | None) -> int:
        return 0 if active_group is None else active_group.depth + 1

    def next_root_index(self) -> int:
        self._root_counter += 1
        return self._root_counter

    def record_identity(self, record: Any, path: tuple[int, ...]) -> str:
        if isinstance(record, dict):
            if self.config.id_field and _is_scalar(record.get(self.config.id_field)):
                return str(record[self.config.id_field])
            if _is_scalar(record.get("id")):
                return str(record["id"])
        return ".".join(str(item) for item in path) if path else "<root>"


def run_config(config_or_path: Config | str | Path) -> RunResult:
    config = load_config(config_or_path) if not isinstance(config_or_path, Config) else config_or_path
    return _Runtime(config).run()


def _hashable_key(value: Any) -> Any:
    try:
        hash(value)
    except TypeError as exc:
        raise PipelineExecutionError(f"Group key is not hashable: {value!r}") from exc
    return value


def _is_scalar(value: Any) -> bool:
    return isinstance(value, str | int | float | bool) and value is not None


def _is_list_annotation(annotation: Any) -> bool:
    if annotation is list or annotation == "list":
        return True
    if isinstance(annotation, str):
        return annotation.startswith("list[") or annotation.startswith("typing.List[")
    return get_origin(annotation) is list


def _gather_children(field_name: str, children: list[Any]) -> list[Any]:
    if children and all(isinstance(child, dict) and field_name in child for child in children):
        return [child[field_name] for child in children]
    return children


def _is_dspy_target(target: Any) -> bool:
    module = getattr(type(target), "__module__", "")
    return module == "dspy" or module.startswith("dspy.")


def _call_openrouter(stage: StageRef, kwargs: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise PipelineExecutionError("OPENROUTER_API_KEY is required for OpenRouter LM stages.")
    assert stage.lm_signature is not None
    output_fields = [field.name for field in stage.lm_signature.output_fields]
    instructions = _lm_instructions(stage.target)
    body = {
        "model": stage.lm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are executing a Mine Trace declarative stage. "
                    "Return only a JSON object. Include exactly the requested output fields."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": stage.name,
                        "instructions": instructions,
                        "inputs": kwargs,
                        "output_fields": output_fields,
                    },
                    sort_keys=True,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "mine-trace",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PipelineExecutionError(f"OpenRouter request failed for `{stage.name}`: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise PipelineExecutionError(f"OpenRouter request failed for `{stage.name}`: {exc.reason}") from exc
    except TimeoutError as exc:
        raise PipelineExecutionError(f"OpenRouter request timed out for `{stage.name}`.") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PipelineExecutionError(f"OpenRouter response for `{stage.name}` did not contain message content.") from exc
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PipelineExecutionError(f"OpenRouter response for `{stage.name}` was not valid JSON: {content}") from exc
    if not isinstance(parsed, dict):
        raise PipelineExecutionError(f"OpenRouter response for `{stage.name}` must be a JSON object.")
    return parsed


def _lm_instructions(target: Any) -> str:
    instructions = getattr(target, "instructions", None)
    if instructions:
        return str(instructions)
    signature = getattr(target, "signature", None)
    doc = getattr(signature, "__doc__", None) or getattr(type(signature), "__doc__", None)
    if doc:
        return str(doc).strip()
    doc = getattr(target, "__doc__", None) or getattr(type(target), "__doc__", None)
    return str(doc).strip() if doc else ""
