from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO, get_origin
from pydantic import BaseModel
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

from trace_miner.config import load_config
from trace_miner.errors import PipelineExecutionError
from trace_miner.home import configure_lm_cache
from trace_miner.model import (
    Config,
    Failure,
    GroupSpec,
    LM_PROVIDERS,
    LmUsage,
    PipelineSpec,
    RunManifest,
    RunResult,
    RunStats,
    StageRef,
    Step,
)
from trace_miner.telemetry import (
    SERVE_SNAPSHOT_SECONDS,
    capture_pipeline_finished_async,
    capture_pipeline_serving_async,
    record_command_run_id,
    run_command_name,
)


_PROGRESS_NCOLS = 120


@dataclass
class _Envelope:
    record: Any
    path: tuple[int, ...]
    group: _GroupState | None = None


@dataclass(frozen=True)
class _WriteEvent:
    path: Path
    envelopes: tuple[_Envelope, ...]


@dataclass(frozen=True)
class _JsonlLine:
    path: Path
    offset: int
    next_offset: int
    record: Any
    blank: bool = False


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
        print(f"[trace-miner] {self.stage_name} {self.group_id or '-'}: {message}")


class _GroupSkipped(Exception):
    pass


@dataclass
class _GroupState:
    runtime: _Runtime
    spec: GroupSpec
    parent_env: _Envelope
    depth: int
    group_id: str
    fired: bool = False
    skipped: bool = False
    failures: list[Failure] = field(default_factory=list)
    reduce_progress: _ProgressTask | None = None

    @property
    def parent_record(self) -> Any:
        return self.parent_env.record

    async def fail_child(self, env: _Envelope, stage_name: str, error: BaseException) -> None:
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
            return
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
                "parent": self.runtime.record_identity(self.parent_record, self.parent_env.path),
                "failing_child": failure.record_identity,
                "stage": failure.stage,
                "error": failure.error,
            }
        )

    async def reduce(self, children: list[_Envelope]) -> list[_Envelope]:
        if self.fired:
            return []
        self.fired = True
        if self.skipped:
            return []
        if self.spec.ordered:
            children = sorted(children, key=lambda env: env.path)
        self.runtime.stats.retired += len(children)
        if self.reduce_progress is not None:
            self.reduce_progress.open()
        outputs = await self.runtime.call_stage(
            self.spec.reduce,
            [env.record for env in children],
            source=self.parent_env,
            parent=self.parent_record,
            group_id=self.group_id,
            key=None,
            index=None,
            path=self.parent_env.path,
            depth=self.depth,
            failures=self.failures,
        )
        self.runtime.stats.groups_reduced += 1
        if self.reduce_progress is not None:
            self.reduce_progress.advance()
        return [
            _Envelope(record, self.parent_env.path + (i,), None)
            for i, record in enumerate(outputs, start=1)
        ]


class _Runtime:
    def __init__(
        self,
        config: Config,
        progress: bool | None = False,
        progress_stream: TextIO | None = None,
        append_outputs: bool = False,
        capture_writes: bool = False,
    ):
        self.config = config
        self.stats = RunStats()
        self.manifest = RunManifest()
        self._group_counter = 0
        self._root_counter = 0
        self._open_groups: list[_GroupState] = []
        self._lm_semaphore: asyncio.Semaphore | None = None
        self._dspy_lms: dict[str, Any] = {}
        self.progress = _ProgressReporter(progress, progress_stream)
        self.append_outputs = append_outputs
        self.capture_writes = capture_writes
        self.write_events: list[_WriteEvent] = []
        self._alert_log: Path | None = None
        self._stage_kinds: list[str] = []
        self._registered_stages: set[int] = set()
        self._register_stages()
        self._prepare_dspy_lms()

    async def run(self) -> RunResult:
        started = time.perf_counter()
        run_id = uuid.uuid4().hex
        record_command_run_id(run_id)
        caught: BaseException | None = None
        try:
            for pipeline in self.config.pipelines:
                await self._run_pipeline(pipeline)
            leaks = [group for group in self._open_groups if not group.fired]
            if leaks:
                details = ", ".join(f"{group.group_id} count={group.outstanding}" for group in leaks)
                raise PipelineExecutionError(f"Open groups remain at shutdown: {details}")
            return RunResult(self.stats, self.manifest)
        except BaseException as exc:
            caught = exc
            raise
        finally:
            self.stats.elapsed_s = time.perf_counter() - started
            depth, width = _pipeline_span(self.config)
            capture_pipeline_finished_async(
                command="run",
                run_id=run_id,
                pipeline_count=len(self.config.pipelines),
                lm_provider=self.config.lm_provider,
                lm_model=self.config.lm_default_model,
                stats=self.stats,
                exception=caught,
                kinds=self.stage_kinds(),
                depth=depth,
                width=width,
            )

    async def _run_pipeline(self, pipeline: PipelineSpec) -> None:
        envelopes = list(self._read_inputs(pipeline.inputs))
        await self.process_pipeline_envelopes(pipeline, envelopes)

    async def process_pipeline_envelopes(self, pipeline: PipelineSpec, envelopes: list[_Envelope]) -> list[_Envelope]:
        previous = self._alert_log
        self._alert_log = alert_log_path(self.config, pipeline)
        self._alert_log.parent.mkdir(parents=True, exist_ok=True)
        self._alert_log.write_text("", encoding="utf-8")
        try:
            outputs = await self.apply_steps(envelopes, pipeline.steps, active_group=None)
            self._write_envelopes(pipeline.output, outputs, append=self.append_outputs)
            self.stats.retired += len(outputs)
            return outputs
        finally:
            self._alert_log = previous

    async def process_stream_pipeline_envelopes(
        self,
        pipeline: PipelineSpec,
        envelopes: list[_Envelope],
    ) -> list[_Envelope]:
        previous = self._alert_log
        self._alert_log = alert_log_path(self.config, pipeline)
        try:
            outputs = await self.apply_steps(envelopes, pipeline.steps, active_group=None)
            self._write_envelopes(pipeline.output, outputs, append=True)
            self.stats.retired += len(outputs)
            return outputs
        finally:
            self._alert_log = previous

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

    async def apply_steps(
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
                current = await self._apply_stage_step(current, step, active_group)
            elif step.kind == "alert":
                current = await self._apply_alert_step(current, step, active_group)
            elif step.kind == "tee":
                assert step.path is not None
                self._write_envelopes(step.path, current, append=True)
            elif step.kind == "group":
                assert step.group is not None
                current = await self._apply_group_step(current, step.group, active_group)
            else:
                raise AssertionError(f"Unhandled step kind: {step.kind}")
        return current

    async def _apply_alert_step(
        self,
        envelopes: list[_Envelope],
        step: Step,
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        assert step.stage is not None
        task = self.progress.start("alert", step.stage.name, len(envelopes))
        try:
            if step.stage.kind == "lm" or step.stage.is_async:
                results = await asyncio.gather(
                    *(self._call_stage_for_env_result(step.stage, env, task) for env in envelopes)
                )
            else:
                results = []
                for env in envelopes:
                    results.append(await self._call_stage_for_env_result(step.stage, env, task))
            kept: list[_Envelope] = []
            fired: list[_Envelope] = []
            for env, result in results:
                if isinstance(result, Exception):
                    if active_group is None:
                        raise result
                    await active_group.fail_child(env, step.stage.name, result)
                    continue
                body = self._alert_body(step.stage, result)
                if body is not None:
                    fired.append(_Envelope(self._alert_record(step.stage.name, body, env), env.path, active_group))
                kept.append(env)
            if fired and self._alert_log is not None:
                self._write_envelopes(self._alert_log, fired, append=True)
            return kept
        finally:
            task.close()

    def _alert_body(self, stage: StageRef, emitted: list[Any]) -> str | dict[str, Any] | None:
        if not emitted:
            return None
        if len(emitted) > 1:
            raise PipelineExecutionError(
                f"`alert` stage `{stage.name}` returned {len(emitted)} values; expected a reason string or None."
            )
        body = emitted[0]
        if stage.kind == "lm":
            if not isinstance(body, dict):
                raise PipelineExecutionError(f"LM alert `{stage.name}` must produce an output mapping.")
            return body
        if body is None:
            return None
        if not isinstance(body, str):
            raise PipelineExecutionError(f"`alert` stage `{stage.name}` must return a reason string or None.")
        text = body.strip()
        if not text or text.lower() == "none":
            return None
        return text

    def _alert_record(self, name: str, body: str | dict[str, Any], env: _Envelope) -> dict[str, Any]:
        record: dict[str, Any] = {"reason": body} if isinstance(body, str) else dict(body)
        record["name"] = name
        field_name = self.config.id_field
        if not field_name:
            return record
        value = _id_value(env.record, field_name)
        if value is None and env.group is not None:
            value = _id_value(env.group.parent_record, field_name)
        if value is not None:
            record[field_name] = value
        return record

    async def _apply_stage_step(
        self,
        envelopes: list[_Envelope],
        step: Step,
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        assert step.stage is not None
        output: list[_Envelope] = []
        task = self.progress.start(step.kind, step.stage.name, len(envelopes))
        try:
            if step.stage.kind == "lm" or step.stage.is_async:
                results = await asyncio.gather(
                    *(self._call_stage_for_env_result(step.stage, env, task) for env in envelopes)
                )
            else:
                results = []
                for env in envelopes:
                    results.append(await self._call_stage_for_env_result(step.stage, env, task))
            for env, result in results:
                if isinstance(result, Exception):
                    if active_group is None:
                        raise result
                    await active_group.fail_child(env, step.stage.name, result)
                    continue
                emitted = result
                self._assert_emit_count(step, env, emitted)
                if not emitted:
                    self.stats.dropped += 1
                    continue
                for i, record in enumerate(emitted, start=1):
                    output.append(_Envelope(record, env.path + (i,), active_group))
            return output
        finally:
            task.close()

    async def _call_stage_for_env_result(
        self,
        stage: StageRef,
        env: _Envelope,
        task: _ProgressTask,
    ) -> tuple[_Envelope, list[Any] | Exception]:
        try:
            return env, await self.call_stage_for_env(stage, env)
        except Exception as exc:
            return env, exc
        finally:
            task.advance()

    async def _apply_group_step(
        self,
        envelopes: list[_Envelope],
        spec: GroupSpec,
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        task = self.progress.start(
            "reduce",
            spec.reduce.name,
            len(envelopes),
            display_immediately=False,
            close_after_advance=True,
        )
        outputs = []
        try:
            for env in envelopes:
                nested_outputs = await self._process_expand_group(
                    spec,
                    env,
                    depth=self._depth_for(active_group),
                    reduce_progress=task,
                )
                outputs.extend(_Envelope(out.record, out.path, active_group) for out in nested_outputs)
            return outputs
        finally:
            task.close()

    async def _process_expand_group(
        self,
        spec: GroupSpec,
        env: _Envelope,
        depth: int,
        reduce_progress: _ProgressTask | None,
    ) -> list[_Envelope]:
        group = self._open_group(spec, env, depth=depth, reduce_progress=reduce_progress)
        try:
            try:
                emitted = await self.call_stage_for_env(
                    spec.expand,
                    env,
                    parent=env.record,
                    group_id=group.group_id,
                    key=None,
                    depth=depth,
                    failures=group.failures,
                )
            except Exception as exc:
                if isinstance(exc, (_GroupSkipped, PipelineExecutionError)):
                    raise
                raise PipelineExecutionError(
                    f"Group {group.group_id} expand stage `{spec.expand.name}` failed: {exc}"
                ) from exc
            child_envs = [_Envelope(record, env.path + (i,), group) for i, record in enumerate(emitted, start=1)]
            survivors = await self._run_group_children(group, child_envs)
        except _GroupSkipped:
            return []
        return await group.reduce(survivors)

    async def _run_group_children(self, group: _GroupState, child_envs: list[_Envelope]) -> list[_Envelope]:
        outputs = await self.apply_steps(child_envs, group.spec.pipeline, active_group=group)
        if group.skipped:
            raise _GroupSkipped()
        return outputs

    async def call_stage_for_env(
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
        return await self.call_stage(
            stage,
            env.record,
            source=env,
            parent=parent if parent is not None else (source_group.parent_record if source_group else None),
            group_id=group_id if group_id is not None else (source_group.group_id if source_group else None),
            key=key,
            index=env.path[-1] if env.path else None,
            path=env.path,
            depth=depth if source_group is None else source_group.depth,
            failures=failures if failures is not None else (source_group.failures if source_group else []),
        )

    async def call_stage(
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
            return await self._call_lm_stage(stage, arg, parent, path)
        try:
            result = stage.target(arg, ctx) if stage.accepts_ctx else stage.target(arg)
            if stage.is_async or inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            if _is_unconfigured_dspy(exc):
                raise PipelineExecutionError(
                    f"Stage `{stage.name}` invoked DSPy without a configured language model."
                ) from exc
            raise
        if result is not None and ctx.emitted:
            identity = self.record_identity(arg, path)
            raise PipelineExecutionError(f"Stage `{stage.name}` both returned a value and emitted for record {identity}.")
        if result is not None:
            self.stats.emitted += 1
            return [result]
        return ctx.emitted

    async def _call_lm_stage(self, stage: StageRef, arg: Any, parent: Any, path: tuple[int, ...]) -> list[Any]:
        if stage.lm_signature is None:
            raise PipelineExecutionError(f"LM stage `{stage.name}` has no signature.")
        if stage.role == "reduce":
            kwargs = self._lm_reduce_kwargs(stage, arg, parent, path)
        else:
            kwargs = self._lm_record_kwargs(stage, arg, path)
        prediction = await self._invoke_lm(stage, kwargs)
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
        if stage.role == "alert":
            flag_name = _lm_alert_flag(stage)
            if not outputs[flag_name]:
                return []
            self.stats.emitted += 1
            return [outputs]
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

    def stage_kinds(self) -> list[str]:
        return list(self._stage_kinds)

    def _register_stages(self) -> None:
        for pipeline in self.config.pipelines:
            self._register_steps(pipeline.steps)

    def _register_steps(self, steps: tuple[Step, ...]) -> None:
        for step in steps:
            if step.stage is not None:
                self._note_stage(step.stage)
            elif step.group is not None:
                self._note_stage(step.group.expand)
                self._register_steps(step.group.pipeline)
                self._note_stage(step.group.reduce)

    def _note_stage(self, stage: StageRef) -> None:
        key = id(stage)
        if key in self._registered_stages:
            return
        self._registered_stages.add(key)
        self._stage_kinds.append(stage.role)

    def _prepare_dspy_lms(self) -> None:
        models = {
            stage.lm_model
            for stage in _iter_stages(self.config)
            if stage.kind == "lm" and stage.lm_model
        }
        if not models:
            return
        provider = LM_PROVIDERS[self.config.lm_provider]
        api_key = os.environ.get(provider.api_key_env)
        if not api_key:
            raise PipelineExecutionError(
                f"{provider.api_key_env} is required for `{self.config.lm_provider}` LM stages."
            )
        configure_lm_cache()
        self._dspy_lms = {
            model: _make_lm(self.config.lm_provider, model, api_key, self.config.lm_cache, provider.extra_body)
            for model in models
        }

    def _lm_slot(self) -> asyncio.Semaphore:
        if self._lm_semaphore is None:
            self._lm_semaphore = asyncio.Semaphore(self.config.lm_max_concurrent)
        return self._lm_semaphore

    async def _invoke_lm(self, stage: StageRef, kwargs: dict[str, Any]) -> Any:
        return await self._invoke_dspy(stage, kwargs)

    async def _invoke_dspy(self, stage: StageRef, kwargs: dict[str, Any]) -> Any:
        assert stage.lm_model is not None
        lm = self._dspy_lms[stage.lm_model]
        async with self._lm_slot():
            prediction, usage = await asyncio.to_thread(_invoke_dspy_sync, stage.target, kwargs, lm)
        self.stats.record_lm(usage)
        return prediction

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

    def _write_jsonl(self, path: Path, records: Iterable[Any], append: bool | None = None) -> None:
        append = self.append_outputs if append is None else append
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=_json_default))
                handle.write("\n")
            handle.flush()

    def _write_envelopes(self, path: Path, envelopes: Iterable[_Envelope], append: bool | None = None) -> None:
        envelope_tuple = tuple(envelopes)
        should_append = self.append_outputs if append is None else append
        if should_append and not envelope_tuple:
            return
        self._write_jsonl(path, [env.record for env in envelope_tuple], append=should_append)
        if self.capture_writes and envelope_tuple:
            self.write_events.append(_WriteEvent(path, envelope_tuple))

    def _open_group(
        self,
        spec: GroupSpec,
        parent_env: _Envelope,
        depth: int,
        reduce_progress: _ProgressTask | None = None,
    ) -> _GroupState:
        self._group_counter += 1
        group = _GroupState(self, spec, parent_env, depth, f"g{self._group_counter}", reduce_progress=reduce_progress)
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


class _ProgressReporter:
    def __init__(self, enabled: bool | None, stream: TextIO | None = None):
        self.stream = stream
        self.enabled = enabled

    def start(
        self,
        kind: str,
        name: str,
        total: int,
        display_immediately: bool = True,
        close_after_advance: bool = False,
    ) -> _ProgressTask:
        disable = None if self.enabled is None else not self.enabled
        return _ProgressTask(
            kind,
            name,
            total,
            self.stream,
            disable if total > 0 else True,
            display_immediately,
            close_after_advance,
        )


class _ProgressTask:
    def __init__(
        self,
        kind: str,
        name: str,
        total: int,
        stream: TextIO | None,
        disable: bool | None,
        display_immediately: bool,
        close_after_advance: bool,
    ):
        self.kind = kind
        self.name = name
        self.total = total
        self.stream = stream
        self.disable = disable
        self.close_after_advance = close_after_advance
        self.current = 0
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._closed = False
        if display_immediately:
            self.open()

    def open(self) -> None:
        if self._progress is not None or self._closed or self.disable is True:
            return
        force_terminal = None if self.disable is None else True
        is_tty = self.stream is None or bool(getattr(self.stream, "isatty", lambda: False)())
        console = Console(
            file=self.stream,
            width=_PROGRESS_NCOLS,
            force_terminal=force_terminal,
            no_color=force_terminal is True and not is_tty,
            highlight=False,
            soft_wrap=False,
        )
        self._progress = Progress(
            TextColumn(r"[dim]\[trace-miner][/] [bold cyan]{task.fields[kind]}[/]: [bold]{task.fields[name]}[/]"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
            expand=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            "",
            total=self.total,
            completed=self.current,
            kind=escape(self.kind),
            name=escape(self.name),
        )

    def advance(self) -> None:
        if self._closed:
            return
        self.open()
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id)
        self.current += 1
        if self.close_after_advance:
            self._close_bar()

    def close(self) -> None:
        if self._closed:
            return
        self._close_bar()
        self._closed = True

    def _close_bar(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None


class _CheckpointStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, config: Config, offsets: dict[Path, int]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config": str(config.config_path),
            "inputs": {str(path): {"offset": offset} for path, offset in sorted(offsets.items())},
        }
        temp_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.path)


class _StreamRun:
    def __init__(
        self,
        config: Config,
        *,
        checkpoint_path: Path | None,
        from_start: bool,
        from_end: bool,
        reset_checkpoint: bool,
        poll_interval: float,
        progress: bool | None,
        progress_stream: TextIO | None,
        max_records: int | None,
        stop_when_idle: bool,
        wake_event: asyncio.Event | None = None,
        stop_event: asyncio.Event | None = None,
        on_write: Callable[[str, Path, Any, int], None] | None = None,
        on_ready: Callable[[], None] | None = None,
    ):
        if from_start and from_end:
            raise PipelineExecutionError("Use only one of `from_start` or `from_end`.")
        if poll_interval <= 0:
            raise PipelineExecutionError("`poll_interval` must be positive.")
        self.config = config
        self.runtime = _Runtime(config, progress=progress, progress_stream=progress_stream, append_outputs=True, capture_writes=True)
        self.checkpoint = _CheckpointStore(checkpoint_path or _default_checkpoint_path(config))
        self.from_start = from_start
        self.from_end = from_end
        self.reset_checkpoint = reset_checkpoint
        self.poll_interval = poll_interval
        self.progress = _StreamProgress(progress, progress_stream)
        self.max_records = max_records
        self.stop_when_idle = stop_when_idle
        self.run_id = uuid.uuid4().hex
        self.offsets: dict[Path, int] = {}
        self.read_offsets: dict[Path, int] = {}
        self.root_inputs: tuple[Path, ...] = ()
        self.consumers: dict[Path, list[PipelineSpec]] = {}
        self.root_consumers: dict[Path, list[PipelineSpec]] = {}
        self._write_cursor = 0
        self.wake_event = wake_event
        self.stop_event = stop_event
        self.on_write = on_write
        self.on_ready = on_ready
        self._published_by_path: dict[Path, str] = {}
        self._publish_offsets: dict[Path, int] = {}
        self._serving_snapshot = 0
        self._last_serving_snapshot = 0.0

    async def run(self) -> RunResult:
        started = time.perf_counter()
        record_command_run_id(self.run_id)
        caught: BaseException | None = None
        try:
            self._build_graph()
            self._load_offsets()
            self.checkpoint.write(self.config, self.offsets)
            self._init_publish_offsets()
            stop_event = self.stop_event or asyncio.Event()
            _bind_stop_signals(stop_event)
            if self.on_ready is not None:
                self.on_ready()
            self._note_serving_snapshot(started)
            processed = 0
            while not stop_event.is_set():
                made_progress = False
                for path in self.root_inputs:
                    for line in _read_available_jsonl(path, self.read_offsets[path]):
                        made_progress = True
                        processed += 1
                        await self._process_line(line)
                        self.progress.render(self.runtime.stats)
                        self._note_serving_snapshot(started)
                        if self.max_records is not None and processed >= self.max_records:
                            return RunResult(self.runtime.stats, self.runtime.manifest)
                if self.stop_when_idle and not made_progress:
                    return RunResult(self.runtime.stats, self.runtime.manifest)
                if not made_progress:
                    await self._wait_for_input(stop_event)
                    self._note_serving_snapshot(started)
            return RunResult(self.runtime.stats, self.runtime.manifest)
        except BaseException as exc:
            caught = exc
            raise
        finally:
            self.runtime.stats.elapsed_s = time.perf_counter() - started
            self.progress.close()
            depth, width = _pipeline_span(self.config)
            capture_pipeline_finished_async(
                command=run_command_name.get(),
                run_id=self.run_id,
                pipeline_count=len(self.config.pipelines),
                lm_provider=self.config.lm_provider,
                lm_model=self.config.lm_default_model,
                stats=self.runtime.stats,
                exception=caught,
                kinds=self.runtime.stage_kinds(),
                depth=depth,
                width=width,
            )

    def _note_serving_snapshot(self, started: float) -> None:
        now = time.perf_counter()
        if self._serving_snapshot and now - self._last_serving_snapshot < SERVE_SNAPSHOT_SECONDS:
            return
        self._serving_snapshot += 1
        self._last_serving_snapshot = now
        self.runtime.stats.elapsed_s = now - started
        depth, width = _pipeline_span(self.config)
        capture_pipeline_serving_async(
            run_id=self.run_id,
            pipeline_count=len(self.config.pipelines),
            lm_provider=self.config.lm_provider,
            lm_model=self.config.lm_default_model,
            stats=self.runtime.stats,
            kinds=self.runtime.stage_kinds(),
            depth=depth,
            width=width,
            snapshot=self._serving_snapshot,
        )

    def _build_graph(self) -> None:
        produced: set[Path] = set()
        for pipeline in self.config.pipelines:
            produced.add(pipeline.output)
            produced.update(_collect_runtime_tee_paths(pipeline.steps))

        consumers: dict[Path, list[PipelineSpec]] = {}
        for pipeline in self.config.pipelines:
            for path in pipeline.inputs:
                consumers.setdefault(path, []).append(pipeline)

        root_inputs = sorted({path for pipeline in self.config.pipelines for path in pipeline.inputs if path not in produced})
        self.root_inputs = tuple(root_inputs)
        self._published_by_path = {
            path: name for name, paths in published_outputs(self.config).items() for path in paths
        }
        self.consumers = consumers
        self.root_consumers = {path: consumers.get(path, []) for path in self.root_inputs}

    def _load_offsets(self) -> None:
        checkpoint = None if self.reset_checkpoint else self.checkpoint.load()
        checkpoint_offsets: dict[Path, int] = {}
        if checkpoint and not self.from_start and not self.from_end:
            for raw_path, item in checkpoint.get("inputs", {}).items():
                checkpoint_offsets[Path(raw_path)] = int(item.get("offset", 0))

        for path in self.root_inputs:
            if self.from_end:
                self.offsets[path] = path.stat().st_size
            elif self.from_start or path not in checkpoint_offsets:
                self.offsets[path] = 0
            else:
                self.offsets[path] = checkpoint_offsets[path]
            self.read_offsets[path] = self.offsets[path]

    async def _process_line(self, line: _JsonlLine) -> None:
        self.read_offsets[line.path] = line.next_offset
        if not line.blank:
            self.runtime.stats.records_read += 1
            envelope = _Envelope(line.record, (self.runtime.next_root_index(),), None)
            for pipeline in self.root_consumers.get(line.path, []):
                await self.runtime.process_stream_pipeline_envelopes(pipeline, [envelope])
            await self._drain_writes()
        self.offsets[line.path] = line.next_offset
        self.checkpoint.write(self.config, self.offsets)

    async def _drain_writes(self) -> None:
        while self._write_cursor < len(self.runtime.write_events):
            event = self.runtime.write_events[self._write_cursor]
            self._write_cursor += 1
            self._publish_new_outputs(event.path)
            for pipeline in self.consumers.get(event.path, []):
                await self.runtime.process_stream_pipeline_envelopes(pipeline, list(event.envelopes))

    async def _wait_for_input(self, stop_event: asyncio.Event) -> None:
        if self.wake_event is not None and self.wake_event.is_set():
            self.wake_event.clear()
            return
        waiters = [asyncio.create_task(stop_event.wait())]
        if self.wake_event is not None:
            waiters.append(asyncio.create_task(self.wake_event.wait()))
        try:
            await asyncio.wait(waiters, timeout=self.poll_interval, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in waiters:
                task.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
            if self.wake_event is not None:
                self.wake_event.clear()

    def _init_publish_offsets(self) -> None:
        if self.on_write is None:
            return
        for path in self._published_by_path:
            self._publish_offsets[path] = path.stat().st_size if path.exists() else 0

    def _publish_new_outputs(self, path: Path) -> None:
        if self.on_write is None:
            return
        name = self._published_by_path.get(path)
        if name is None:
            return
        start = self._publish_offsets.get(path, 0)
        if not path.exists():
            return
        with path.open("rb") as handle:
            handle.seek(start)
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    break
                next_offset = handle.tell()
                self._publish_offsets[path] = next_offset
                if not line.strip():
                    continue
                try:
                    record = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                self.on_write(name, path, record, next_offset)


class _StreamProgress:
    def __init__(self, enabled: bool | None, stream: TextIO | None):
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty() if enabled is None else enabled

    def render(self, stats: RunStats) -> None:
        if not self.enabled:
            return
        groups_open = stats.groups_opened - stats.groups_reduced - stats.groups_skipped
        self.stream.write(
            "\r"
            f"[trace-miner] serve: read={stats.records_read} produced={stats.emitted} "
            f"done={stats.retired} failed={stats.failed} waiting={groups_open}"
        )
        self.stream.flush()

    def close(self) -> None:
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def _pipeline_span(config: Config) -> tuple[int, int]:
    depth = 0
    width = 0
    for pipeline in config.pipelines:
        step_depth, step_width = _step_span(pipeline.steps, 0)
        depth = max(depth, step_depth)
        width = max(width, step_width)
    return depth, width


def _step_span(steps: tuple[Step, ...], group_depth: int) -> tuple[int, int]:
    depth = group_depth
    width = len(steps)
    for step in steps:
        if step.group is None:
            continue
        child_depth, child_width = _step_span(step.group.pipeline, group_depth + 1)
        depth = max(depth, child_depth)
        width = max(width, child_width)
    return depth, width


def _collect_runtime_tee_paths(steps: tuple[Step, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for step in steps:
        if step.kind == "tee" and step.path is not None:
            paths.append(step.path)
        elif step.kind == "group" and step.group is not None:
            paths.extend(_collect_runtime_tee_paths(step.group.pipeline))
    return tuple(paths)


def _read_available_jsonl(path: Path, offset: int) -> Iterable[_JsonlLine]:
    size = path.stat().st_size
    if size < offset:
        raise PipelineExecutionError(f"Streaming input was truncated: {path}")
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            line_offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                return
            next_offset = handle.tell()
            if not line.strip():
                yield _JsonlLine(path, line_offset, next_offset, None, blank=True)
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise PipelineExecutionError(f"Invalid JSON in {path} at byte {line_offset}: {exc}") from exc
            yield _JsonlLine(path, line_offset, next_offset, record)


def _default_checkpoint_path(config: Config) -> Path:
    digest = hashlib.sha256(str(config.config_path).encode("utf-8")).hexdigest()[:16]
    return config.base_dir / ".trace-miner" / "checkpoints" / f"{digest}.json"


def _bind_stop_signals(event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, event.set)
        except (NotImplementedError, RuntimeError):
            pass


def root_inputs(config: Config) -> tuple[Path, ...]:
    produced: set[Path] = set()
    for pipeline in config.pipelines:
        produced.add(pipeline.output)
        produced.update(_collect_runtime_tee_paths(pipeline.steps))
    roots = {path for pipeline in config.pipelines for path in pipeline.inputs if path not in produced}
    return tuple(sorted(roots))


def alert_log_path(config: Config, pipeline: PipelineSpec) -> Path:
    label = pipeline.name if pipeline.name is not None else str(pipeline.source_index)
    return config.base_dir / ".trace-miner" / "alerts" / f"{label}.jsonl"


def published_outputs(config: Config) -> dict[str, tuple[Path, ...]]:
    found: dict[str, tuple[Path, ...]] = {}
    for pipeline in config.pipelines:
        if pipeline.name is None:
            continue
        found[pipeline.name] = (alert_log_path(config, pipeline),)
    return found


async def run_config_async(
    config_or_path: Config | str | Path,
    *,
    progress: bool | None = False,
    progress_stream: TextIO | None = None,
) -> RunResult:
    config = load_config(config_or_path) if not isinstance(config_or_path, Config) else config_or_path
    return await _Runtime(config, progress=progress, progress_stream=progress_stream).run()


def run_config(
    config_or_path: Config | str | Path,
    *,
    progress: bool | None = False,
    progress_stream: TextIO | None = None,
) -> RunResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_config_async(config_or_path, progress=progress, progress_stream=progress_stream))
    raise PipelineExecutionError("`run_config` cannot be called from a running event loop; use `run_config_async`.")


async def run_stream_async(
    config_or_path: Config | str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    from_start: bool = False,
    from_end: bool = False,
    reset_checkpoint: bool = False,
    poll_interval: float = 1.0,
    progress: bool | None = None,
    progress_stream: TextIO | None = None,
    max_records: int | None = None,
    stop_when_idle: bool = False,
    wake_event: asyncio.Event | None = None,
    stop_event: asyncio.Event | None = None,
    on_write: Callable[[str, Path, Any, int], None] | None = None,
    on_ready: Callable[[], None] | None = None,
) -> RunResult:
    config = load_config(config_or_path) if not isinstance(config_or_path, Config) else config_or_path
    checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path is not None else None
    return await _StreamRun(
        config,
        checkpoint_path=checkpoint,
        from_start=from_start,
        from_end=from_end,
        reset_checkpoint=reset_checkpoint,
        poll_interval=poll_interval,
        progress=progress,
        progress_stream=progress_stream,
        max_records=max_records,
        stop_when_idle=stop_when_idle,
        wake_event=wake_event,
        stop_event=stop_event,
        on_write=on_write,
        on_ready=on_ready,
    ).run()


def run_stream(
    config_or_path: Config | str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    from_start: bool = False,
    from_end: bool = False,
    reset_checkpoint: bool = False,
    poll_interval: float = 1.0,
    progress: bool | None = None,
    progress_stream: TextIO | None = None,
    max_records: int | None = None,
    stop_when_idle: bool = False,
) -> RunResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_stream_async(
                config_or_path,
                checkpoint_path=checkpoint_path,
                from_start=from_start,
                from_end=from_end,
                reset_checkpoint=reset_checkpoint,
                poll_interval=poll_interval,
                progress=progress,
                progress_stream=progress_stream,
                max_records=max_records,
                stop_when_idle=stop_when_idle,
            )
        )
    raise PipelineExecutionError("`run_stream` cannot be called from a running event loop; use `run_stream_async`.")


def _lm_alert_flag(stage: StageRef) -> str:
    assert stage.lm_signature is not None
    names = [
        field.name
        for field in stage.lm_signature.output_fields
        if field.annotation is bool or field.annotation == "bool"
    ]
    if len(names) != 1:
        raise PipelineExecutionError(f"LM alert `{stage.name}` must have exactly one bool output field.")
    return names[0]


def _id_value(record: Any, field_name: str) -> Any:
    if isinstance(record, dict) and field_name in record and _is_scalar(record[field_name]):
        return record[field_name]
    return None


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


def _iter_stages(config: Config) -> Iterable[StageRef]:
    def walk(steps: tuple[Step, ...]) -> Iterable[StageRef]:
        for step in steps:
            if step.stage is not None:
                yield step.stage
            if step.group is None:
                continue
            yield step.group.expand
            yield step.group.reduce
            yield from walk(step.group.pipeline)

    for pipeline in config.pipelines:
        yield from walk(pipeline.steps)


def _make_lm(
    provider: str,
    model: str,
    api_key: str,
    cache: bool,
    extra_body: dict[str, Any] | None,
) -> Any:
    import dspy

    kwargs: dict[str, Any] = {"api_key": api_key, "cache": cache}
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return dspy.LM(f"{provider}/{model}", **kwargs)


def _invoke_dspy_sync(target: Any, kwargs: dict[str, Any], lm: Any) -> tuple[Any, LmUsage]:
    import dspy

    with dspy.context(lm=lm):
        with dspy.track_usage() as tracker:
            prediction = target(**kwargs)
    return prediction, _usage_from_tracker(tracker)


def _usage_from_tracker(tracker: Any) -> LmUsage:
    prompt = 0
    completion = 0
    cost = 0.0
    totals = tracker.get_total_tokens() if tracker is not None else {}
    for entry in totals.values():
        if not isinstance(entry, dict):
            continue
        prompt += _as_int(entry.get("prompt_tokens") or entry.get("input_tokens"))
        completion += _as_int(entry.get("completion_tokens") or entry.get("output_tokens"))
        cost += _as_float(entry.get("cost") or entry.get("response_cost"))
    return LmUsage(prompt_tokens=prompt, completion_tokens=completion, cost_usd=cost)


def _is_unconfigured_dspy(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and str(exc).startswith("No LM is loaded")


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)

