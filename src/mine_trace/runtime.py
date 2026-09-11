from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import signal
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TextIO, get_origin
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm.auto import tqdm

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


_PROGRESS_NCOLS = 120


@dataclass
class _Envelope:
    record: Any
    path: tuple[int, ...]
    group: _GroupState | None = None
    stream_tokens: tuple[_StreamToken, ...] = ()


@dataclass(frozen=True)
class _StreamToken:
    path: Path
    offset: int
    next_offset: int
    record_id: str


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
class _StreamEveryState:
    buffer: list[_Envelope] = field(default_factory=list)
    window_index: int = 0


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
    reduce_progress: _ProgressTask | None = None

    @property
    def parent_record(self) -> Any:
        if self.parent_env is not None:
            return self.parent_env.record
        if self.spec.kind == "by" and isinstance(self.spec.by, str):
            return {self.spec.by: self.key}
        return None

    def increment(self, count: int = 1) -> None:
        self.outstanding += count

    async def decrement(self, count: int = 1) -> None:
        self.outstanding -= count
        if self.outstanding < 0:
            raise PipelineExecutionError(f"Group {self.group_id} accounting went negative.")
        if self.outstanding == 0:
            await self.reduce()

    async def adjust_for_outputs(self, count: int) -> None:
        if count == 0:
            await self.decrement()
        elif count > 1:
            self.increment(count - 1)

    async def adjust_for_replacement(self, input_count: int, output_count: int) -> None:
        delta = output_count - input_count
        if delta > 0:
            self.increment(delta)
        elif delta < 0:
            await self.decrement(-delta)

    async def arrive(self, env: _Envelope) -> None:
        self.children.append(env)
        self.runtime.stats.retired += 1
        await self.decrement()

    async def fail_child(self, env: _Envelope, stage_name: str, error: BaseException) -> list[_Envelope]:
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
            await self.decrement()
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

    async def close_producer(self) -> None:
        await self.decrement()

    async def reduce(self) -> None:
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
        if self.reduce_progress is not None:
            self.reduce_progress.open()
        outputs = await self.runtime.call_stage(
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
        stream_tokens = _group_stream_tokens(children, self.parent_env)
        self.reduce_outputs = [
            _Envelope(record, base_path + (i,), None, stream_tokens)
            for i, record in enumerate(outputs, start=1)
        ]
        self.runtime.stats.groups_reduced += 1
        if self.reduce_progress is not None:
            self.reduce_progress.advance()


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
        self.progress = _ProgressReporter(progress, progress_stream)
        self.append_outputs = append_outputs
        self.capture_writes = capture_writes
        self.write_events: list[_WriteEvent] = []
        self.completed_stream_tokens: list[_StreamToken] = []
        self.stream_every: dict[tuple[int, tuple[int, ...]], _StreamEveryState] = {}
        self.held_stream_tokens: set[_StreamToken] = set()

    async def run(self) -> RunResult:
        for pipeline in self.config.pipelines:
            await self._run_pipeline(pipeline)
        leaks = [group for group in self._open_groups if not group.fired]
        if leaks:
            details = ", ".join(f"{group.group_id} count={group.outstanding}" for group in leaks)
            raise PipelineExecutionError(f"Open groups remain at shutdown: {details}")
        return RunResult(self.stats, self.manifest)

    async def _run_pipeline(self, pipeline: PipelineSpec) -> None:
        envelopes = list(self._read_inputs(pipeline.inputs))
        await self.process_pipeline_envelopes(pipeline, envelopes)

    async def process_pipeline_envelopes(self, pipeline: PipelineSpec, envelopes: list[_Envelope]) -> list[_Envelope]:
        outputs = await self.apply_steps(envelopes, pipeline.steps, active_group=None)
        self._write_envelopes(pipeline.output, outputs, append=self.append_outputs)
        self.stats.retired += len(outputs)
        return outputs

    async def process_stream_pipeline_envelopes(
        self,
        pipeline: PipelineSpec,
        envelopes: list[_Envelope],
    ) -> list[_Envelope]:
        outputs = await self.apply_stream_steps(envelopes, pipeline.steps, pipeline.source_index, ())
        self._write_envelopes(pipeline.output, outputs, append=True)
        self.stats.retired += len(outputs)
        return outputs

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
            elif step.kind == "tee":
                assert step.path is not None
                self._write_envelopes(step.path, current, append=True)
            elif step.kind == "group":
                assert step.group is not None
                current = await self._apply_group_step(current, step.group, active_group)
            else:
                raise AssertionError(f"Unhandled step kind: {step.kind}")
        return current

    async def apply_stream_steps(
        self,
        envelopes: list[_Envelope],
        steps: tuple[Step, ...],
        pipeline_index: int,
        step_path: tuple[int, ...],
    ) -> list[_Envelope]:
        current = envelopes
        for idx, step in enumerate(steps):
            current_path = step_path + (idx,)
            if step.kind in {"map", "filter", "stage"}:
                current = await self._apply_stage_step(current, step, active_group=None)
            elif step.kind == "tee":
                assert step.path is not None
                self._write_envelopes(step.path, current, append=True)
            elif step.kind == "group":
                assert step.group is not None
                if step.group.kind == "every":
                    current = await self._apply_stream_every_step(current, step.group, pipeline_index, current_path)
                else:
                    current = await self._apply_group_step(current, step.group, active_group=None)
            else:
                raise AssertionError(f"Unhandled step kind: {step.kind}")
        return current

    async def flush_stream_pipeline(self, pipeline: PipelineSpec) -> list[_Envelope]:
        outputs = await self.flush_stream_steps(pipeline.steps, pipeline.source_index, ())
        self._write_envelopes(pipeline.output, outputs, append=True)
        self.stats.retired += len(outputs)
        return outputs

    async def flush_stream_steps(
        self,
        steps: tuple[Step, ...],
        pipeline_index: int,
        step_path: tuple[int, ...],
    ) -> list[_Envelope]:
        flushed: list[_Envelope] = []
        for idx, step in enumerate(steps):
            current_path = step_path + (idx,)
            if step.kind != "group" or step.group is None or step.group.kind != "every":
                continue
            outputs = await self._flush_stream_every_step(step.group, pipeline_index, current_path)
            if outputs:
                outputs = await self.apply_stream_steps(outputs, steps[idx + 1 :], pipeline_index, current_path)
                flushed.extend(outputs)
        return flushed

    async def _apply_stream_every_step(
        self,
        envelopes: list[_Envelope],
        spec: GroupSpec,
        pipeline_index: int,
        step_path: tuple[int, ...],
    ) -> list[_Envelope]:
        assert spec.every is not None
        state = self.stream_every.setdefault((pipeline_index, step_path), _StreamEveryState())
        state.buffer.extend(envelopes)
        for env in envelopes:
            self.held_stream_tokens.update(env.stream_tokens)

        outputs: list[_Envelope] = []
        while len(state.buffer) >= spec.every:
            window = state.buffer[: spec.every]
            del state.buffer[: spec.every]
            for env in window:
                self.held_stream_tokens.difference_update(env.stream_tokens)
            state.window_index += 1
            outputs.extend(await self._run_every_window(spec, window, state.window_index, active_group=None))
        return outputs

    async def _flush_stream_every_step(
        self,
        spec: GroupSpec,
        pipeline_index: int,
        step_path: tuple[int, ...],
    ) -> list[_Envelope]:
        state = self.stream_every.get((pipeline_index, step_path))
        if state is None or not state.buffer:
            return []
        window = state.buffer
        state.buffer = []
        for env in window:
            self.held_stream_tokens.difference_update(env.stream_tokens)
        state.window_index += 1
        return await self._run_every_window(spec, window, state.window_index, active_group=None)

    async def _run_every_window(
        self,
        spec: GroupSpec,
        window: list[_Envelope],
        window_index: int,
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        group = self._open_group(spec, None, key=window_index, depth=self._depth_for(active_group))
        child_envs = []
        for i, env in enumerate(window, start=1):
            group.increment()
            child_envs.append(_Envelope(env.record, env.path + (i,), group, env.stream_tokens))
        await group.close_producer()
        try:
            await self._run_group_children(group, child_envs)
        except _GroupSkipped:
            nested_outputs = []
        else:
            nested_outputs = group.reduce_outputs
        if active_group is not None:
            await active_group.adjust_for_replacement(len(window), len(nested_outputs))
        return [_Envelope(out.record, out.path, active_group, out.stream_tokens) for out in nested_outputs]

    async def _apply_stage_step(
        self,
        envelopes: list[_Envelope],
        step: Step,
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        assert step.stage is not None
        output: list[_Envelope] = []
        task = self.progress.start(f"{step.kind}: {step.stage.name}", len(envelopes))
        if step.stage.kind == "lm" or step.stage.is_async:
            results = await asyncio.gather(
                *(self._call_stage_for_env_result(step.stage, env, task) for env in envelopes)
            )
        else:
            results = []
            for env in envelopes:
                results.append(await self._call_stage_for_env_result(step.stage, env, task))
        try:
            for env, result in results:
                failed_under_policy = False
                if isinstance(result, Exception):
                    if active_group is None:
                        raise result
                    emitted = await active_group.fail_child(env, step.stage.name, result)
                    failed_under_policy = True
                else:
                    emitted = result
                if failed_under_policy:
                    continue
                self._assert_emit_count(step, env, emitted)
                if active_group is not None:
                    await active_group.adjust_for_outputs(len(emitted))
                if not emitted:
                    self.stats.dropped += 1
                    if active_group is None:
                        self.completed_stream_tokens.extend(env.stream_tokens)
                    continue
                for i, record in enumerate(emitted, start=1):
                    output.append(_Envelope(record, env.path + (i,), active_group, env.stream_tokens))
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
        if spec.kind == "expand":
            task = self.progress.start(
                f"reduce: {spec.reduce.name}",
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
                    if active_group is not None:
                        await active_group.adjust_for_outputs(len(nested_outputs))
                    outputs.extend(_Envelope(out.record, out.path, active_group, out.stream_tokens) for out in nested_outputs)
                return outputs
            finally:
                task.close()
        if spec.kind == "by":
            return await self._process_by_groups(spec, envelopes, active_group)
        if spec.kind == "every":
            return await self._process_every_groups(spec, envelopes, active_group)
        raise AssertionError(f"Unhandled group kind: {spec.kind}")

    async def _process_expand_group(
        self,
        spec: GroupSpec,
        env: _Envelope,
        depth: int,
        reduce_progress: _ProgressTask | None,
    ) -> list[_Envelope]:
        group = self._open_group(spec, env, key=None, depth=depth, reduce_progress=reduce_progress)
        try:
            assert spec.expand is not None
            emitted = await self.call_stage_for_env(
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
                child_envs.append(_Envelope(record, env.path + (i,), group, env.stream_tokens))
            await group.close_producer()
            if group.skipped:
                return []
            await self._run_group_children(group, child_envs)
        except _GroupSkipped:
            return []
        except Exception as exc:
            if isinstance(exc, PipelineExecutionError):
                raise
            raise PipelineExecutionError(
                f"Group {group.group_id} expand stage `{spec.expand.name if spec.expand else '<unknown>'}` failed: {exc}"
            ) from exc
        return group.reduce_outputs

    async def _process_by_groups(
        self,
        spec: GroupSpec,
        envelopes: list[_Envelope],
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        grouped: dict[Any, list[_Envelope]] = {}
        order: list[Any] = []
        for env in envelopes:
            key = await self._group_key(spec, env)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(env)

        outputs: list[_Envelope] = []
        task = self.progress.start(
            f"reduce: {spec.reduce.name}",
            len(order),
            display_immediately=False,
            close_after_advance=True,
        )
        try:
            for key in order:
                group = self._open_group(spec, None, key=key, depth=self._depth_for(active_group), reduce_progress=task)
                child_envs: list[_Envelope] = []
                for i, env in enumerate(grouped[key], start=1):
                    group.increment()
                    child_envs.append(_Envelope(env.record, env.path + (i,), group, env.stream_tokens))
                await group.close_producer()
                try:
                    await self._run_group_children(group, child_envs)
                except _GroupSkipped:
                    nested_outputs = []
                else:
                    nested_outputs = group.reduce_outputs
                outputs.extend(_Envelope(out.record, out.path, active_group, out.stream_tokens) for out in nested_outputs)
            if active_group is not None:
                await active_group.adjust_for_replacement(len(envelopes), len(outputs))
            return outputs
        finally:
            task.close()

    async def _process_every_groups(
        self,
        spec: GroupSpec,
        envelopes: list[_Envelope],
        active_group: _GroupState | None,
    ) -> list[_Envelope]:
        assert spec.every is not None
        outputs: list[_Envelope] = []
        total = (len(envelopes) + spec.every - 1) // spec.every
        task = self.progress.start(
            f"reduce: {spec.reduce.name}",
            total,
            display_immediately=False,
            close_after_advance=True,
        )
        try:
            for start in range(0, len(envelopes), spec.every):
                window = envelopes[start : start + spec.every]
                group = self._open_group(
                    spec,
                    None,
                    key=start // spec.every + 1,
                    depth=self._depth_for(active_group),
                    reduce_progress=task,
                )
                child_envs = []
                for i, env in enumerate(window, start=1):
                    group.increment()
                    child_envs.append(_Envelope(env.record, env.path + (i,), group, env.stream_tokens))
                await group.close_producer()
                try:
                    await self._run_group_children(group, child_envs)
                except _GroupSkipped:
                    nested_outputs = []
                else:
                    nested_outputs = group.reduce_outputs
                outputs.extend(_Envelope(out.record, out.path, active_group, out.stream_tokens) for out in nested_outputs)
            if active_group is not None:
                await active_group.adjust_for_replacement(len(envelopes), len(outputs))
            return outputs
        finally:
            task.close()

    async def _run_group_children(self, group: _GroupState, child_envs: list[_Envelope]) -> None:
        if group.fired:
            return
        outputs = await self.apply_steps(child_envs, group.spec.pipeline, active_group=group)
        for env in outputs:
            if group.skipped:
                raise _GroupSkipped()
            await group.arrive(env)

    async def _group_key(self, spec: GroupSpec, env: _Envelope) -> Any:
        if isinstance(spec.by, StageRef):
            emitted = await self.call_stage_for_env(spec.by, env)
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
            key=key if key is not None else (source_group.key if source_group else None),
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
        result = stage.target(arg, ctx) if stage.accepts_ctx else stage.target(arg)
        if stage.is_async or inspect.isawaitable(result):
            result = await result
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

    async def _invoke_lm(self, stage: StageRef, kwargs: dict[str, Any]) -> Any:
        target = stage.target
        if stage.lm_provider == "openrouter" and _is_dspy_target(target):
            return await self._call_openrouter_async(stage, kwargs)
        if hasattr(target, "with_model"):
            target = target.with_model(stage.lm_model)
        if callable(target):
            prediction = target(**kwargs)
            if inspect.isawaitable(prediction):
                return await prediction
            return prediction
        if stage.lm_provider == "openrouter":
            return await self._call_openrouter_async(stage, kwargs)
        raise PipelineExecutionError(f"Unsupported LM provider `{stage.lm_provider}` for stage `{stage.name}`.")

    async def _call_openrouter_async(self, stage: StageRef, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self._lm_semaphore is None:
            self._lm_semaphore = asyncio.Semaphore(self.config.lm_max_concurrent)
        async with self._lm_semaphore:
            return await asyncio.to_thread(_call_openrouter, stage, kwargs)

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
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
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
        parent_env: _Envelope | None,
        key: Any,
        depth: int,
        reduce_progress: _ProgressTask | None = None,
    ) -> _GroupState:
        self._group_counter += 1
        group = _GroupState(self, spec, parent_env, key, depth, f"g{self._group_counter}", reduce_progress=reduce_progress)
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
        label: str,
        total: int,
        display_immediately: bool = True,
        close_after_advance: bool = False,
    ) -> _ProgressTask:
        disable = None if self.enabled is None else not self.enabled
        return _ProgressTask(
            label,
            total,
            self.stream,
            disable if total > 0 else True,
            display_immediately,
            close_after_advance,
        )


class _ProgressTask:
    def __init__(
        self,
        label: str,
        total: int,
        stream: TextIO | None,
        disable: bool | None,
        display_immediately: bool,
        close_after_advance: bool,
    ):
        self.label = label
        self.total = total
        self.stream = stream
        self.disable = disable
        self.close_after_advance = close_after_advance
        self.current = 0
        self._bar: tqdm | None = None
        self._closed = False
        if display_immediately:
            self.open()

    def open(self) -> None:
        if self._bar is not None or self._closed:
            return
        self._bar = tqdm(
            total=self.total,
            initial=self.current,
            desc=f"[mine-trace] {self.label}",
            file=self.stream,
            disable=self.disable,
            leave=True,
            ncols=_PROGRESS_NCOLS,
            ascii=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            unit="record",
        )

    def advance(self) -> None:
        if self._closed:
            return
        self.open()
        if self._bar is not None:
            self._bar.update(1)
        self.current += 1
        if self.close_after_advance:
            self._close_bar()

    def close(self) -> None:
        if self._closed:
            return
        self._close_bar()
        self._closed = True

    def _close_bar(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class _ExecutionJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()

    def committed_offsets(self) -> dict[Path, int]:
        offsets: dict[Path, int] = {}
        if not self.path.exists():
            return offsets
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") != "input_committed":
                    continue
                path = Path(event["path"])
                offsets[path] = max(offsets.get(path, 0), int(event["next_offset"]))
        return offsets


class _CheckpointStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, config: Config, journal_path: Path, offsets: dict[Path, int]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config": str(config.config_path),
            "journal": str(journal_path),
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
        self.journal = _ExecutionJournal(_default_journal_path(config, self.run_id))
        self.offsets: dict[Path, int] = {}
        self.read_offsets: dict[Path, int] = {}
        self.completed: dict[Path, dict[int, _StreamToken]] = {}
        self.root_inputs: tuple[Path, ...] = ()
        self.consumers: dict[Path, list[PipelineSpec]] = {}
        self.root_consumers: dict[Path, list[PipelineSpec]] = {}
        self._write_cursor = 0

    async def run(self) -> RunResult:
        _validate_streamable(self.config)
        self._build_graph()
        self._load_offsets()
        self.checkpoint.write(self.config, self.journal.path, self.offsets)
        self.journal.append({"type": "run_started", "run_id": self.run_id, "config": str(self.config.config_path)})
        stop_event = _install_stop_event()
        processed = 0
        try:
            while not stop_event.is_set():
                made_progress = False
                for path in self.root_inputs:
                    for line in _read_available_jsonl(path, self.read_offsets[path]):
                        made_progress = True
                        processed += 1
                        await self._process_line(line)
                        self.progress.render(self.runtime.stats)
                        if self.max_records is not None and processed >= self.max_records:
                            await self._flush()
                            self.journal.append({"type": "run_stopped", "reason": "max_records"})
                            return RunResult(self.runtime.stats, self.runtime.manifest)
                if self.stop_when_idle and not made_progress:
                    await self._flush()
                    self.journal.append({"type": "run_stopped", "reason": "idle"})
                    return RunResult(self.runtime.stats, self.runtime.manifest)
                if not made_progress:
                    await asyncio.sleep(self.poll_interval)
            await self._flush()
            self.journal.append({"type": "run_stopped", "reason": "signal"})
            return RunResult(self.runtime.stats, self.runtime.manifest)
        finally:
            self.progress.close()

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
        self.consumers = consumers
        self.root_consumers = {path: consumers.get(path, []) for path in self.root_inputs}

    def _load_offsets(self) -> None:
        checkpoint = None if self.reset_checkpoint else self.checkpoint.load()
        checkpoint_offsets: dict[Path, int] = {}
        checkpoint_journal: Path | None = None
        if checkpoint and not self.from_start and not self.from_end:
            checkpoint_journal = Path(checkpoint.get("journal", ""))
            for raw_path, item in checkpoint.get("inputs", {}).items():
                checkpoint_offsets[Path(raw_path)] = int(item.get("offset", 0))
        if checkpoint_journal and checkpoint_journal.exists():
            for path, offset in _ExecutionJournal(checkpoint_journal).committed_offsets().items():
                checkpoint_offsets[path] = max(checkpoint_offsets.get(path, 0), offset)

        for path in self.root_inputs:
            if self.from_end:
                self.offsets[path] = path.stat().st_size
            elif self.from_start or path not in checkpoint_offsets:
                self.offsets[path] = 0
            else:
                self.offsets[path] = checkpoint_offsets[path]
            self.read_offsets[path] = self.offsets[path]

    async def _process_line(self, line: _JsonlLine) -> None:
        if line.blank:
            token = _StreamToken(line.path, line.offset, line.next_offset, "<blank>")
            self.journal.append(
                {
                    "type": "input_admitted",
                    "path": str(line.path),
                    "offset": line.offset,
                    "next_offset": line.next_offset,
                    "record_id": token.record_id,
                }
            )
            self.read_offsets[line.path] = line.next_offset
            self._mark_tokens_complete([token])
            return
        path_tuple = (self.runtime.next_root_index(),)
        record_id = self.runtime.record_identity(line.record, path_tuple)
        token = _StreamToken(line.path, line.offset, line.next_offset, record_id)
        self.journal.append(
            {
                "type": "input_admitted",
                "path": str(line.path),
                "offset": line.offset,
                "next_offset": line.next_offset,
                "record_id": record_id,
            }
        )
        self.runtime.stats.records_read += 1
        self.read_offsets[line.path] = line.next_offset
        envelope = _Envelope(line.record, path_tuple, None, (token,))
        for pipeline in self.root_consumers.get(line.path, []):
            await self.runtime.process_stream_pipeline_envelopes(pipeline, [envelope])
        terminal_tokens = await self._drain_writes()
        terminal_tokens.extend(self.runtime.completed_stream_tokens)
        self.runtime.completed_stream_tokens.clear()
        self._mark_tokens_complete([*terminal_tokens, token])

    async def _drain_writes(self) -> list[_StreamToken]:
        terminal_tokens: list[_StreamToken] = []
        while self._write_cursor < len(self.runtime.write_events):
            event = self.runtime.write_events[self._write_cursor]
            self._write_cursor += 1
            downstream = self.consumers.get(event.path, [])
            if downstream:
                for pipeline in downstream:
                    await self.runtime.process_stream_pipeline_envelopes(pipeline, list(event.envelopes))
            else:
                terminal_tokens.extend(_envelope_stream_tokens(event.envelopes))
        return terminal_tokens

    async def _flush(self) -> None:
        for pipeline in self.config.pipelines:
            await self.runtime.flush_stream_pipeline(pipeline)
            terminal_tokens = await self._drain_writes()
            terminal_tokens.extend(self.runtime.completed_stream_tokens)
            self.runtime.completed_stream_tokens.clear()
            self._mark_tokens_complete(terminal_tokens)

    def _mark_tokens_complete(self, tokens: Iterable[_StreamToken]) -> None:
        changed = False
        for token in tokens:
            if token in self.runtime.held_stream_tokens:
                continue
            if self.offsets.get(token.path, 0) > token.offset:
                continue
            self.completed.setdefault(token.path, {})[token.offset] = token

        for path, by_offset in self.completed.items():
            current = self.offsets.get(path, 0)
            while current in by_offset:
                token = by_offset.pop(current)
                if token in self.runtime.held_stream_tokens:
                    by_offset[token.offset] = token
                    break
                self.journal.append(
                    {
                        "type": "input_committed",
                        "path": str(token.path),
                        "offset": token.offset,
                        "next_offset": token.next_offset,
                        "record_id": token.record_id,
                    }
                )
                current = token.next_offset
                self.offsets[path] = current
                changed = True
        if changed:
            self.checkpoint.write(self.config, self.journal.path, self.offsets)


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
            f"[mine-trace] stream: records={stats.records_read} emitted={stats.emitted} "
            f"retired={stats.retired} failed={stats.failed} groups_open={groups_open}"
        )
        self.stream.flush()

    def close(self) -> None:
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def _validate_streamable(config: Config) -> None:
    for pipeline in config.pipelines:
        _validate_streamable_steps(pipeline.steps)


def _validate_streamable_steps(steps: tuple[Step, ...]) -> None:
    for step in steps:
        if step.kind != "group" or step.group is None:
            continue
        if step.group.kind == "by":
            raise PipelineExecutionError("`by` groups need a close condition in streaming mode; use batch mode or wait for complete_when.")
        _validate_streamable_steps(step.group.pipeline)


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
    return config.base_dir / ".mine-trace" / "checkpoints" / f"{digest}.json"


def _default_journal_path(config: Config, run_id: str) -> Path:
    return config.base_dir / ".mine-trace" / "runs" / run_id / "journal.jsonl"


def _install_stop_event() -> asyncio.Event:
    event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, event.set)
        except (NotImplementedError, RuntimeError):
            pass
    return event


def _group_stream_tokens(children: list[_Envelope], parent_env: _Envelope | None) -> tuple[_StreamToken, ...]:
    tokens = _envelope_stream_tokens(children)
    if tokens:
        return tuple(tokens)
    return parent_env.stream_tokens if parent_env is not None else ()


def _envelope_stream_tokens(envelopes: Iterable[_Envelope]) -> list[_StreamToken]:
    seen: set[_StreamToken] = set()
    tokens: list[_StreamToken] = []
    for env in envelopes:
        for token in env.stream_tokens:
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


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
