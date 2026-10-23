"""Physical operators. Each one consumes events from one input and emits events downstream.

Ordering invariant: an operator forwards `Closed(prefix)` only after every row it will
ever emit under that prefix has been emitted. Async operators enforce this by making the
close wait on all of their in-flight work under the prefix.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import dspy

from . import signatures as sigs
from .lm import Executor, resolve_lm
from .rows import Closed, Event, Key, Row, RowAdded, RowUpdated, is_prefix, sort_key

log = logging.getLogger("tracepress")

END = object()  # end-of-stream marker


class Operator:
    def __init__(self) -> None:
        self.outputs: list[asyncio.Queue] = []
        self.executor: Executor | None = None

    def emit(self, event: Event | object) -> None:
        for queue in self.outputs:
            queue.put_nowait(event)

    def emit_row(self, row: Row, updated: bool = False) -> None:
        self.emit(RowUpdated(row) if updated else RowAdded(row))

    async def on_row(self, row: Row, updated: bool) -> None:
        raise NotImplementedError

    async def on_close(self, prefix: Key) -> None:
        self.emit(Closed(prefix))

    async def on_end(self) -> None:
        pass

    async def consume(self, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            if event is END:
                await self.on_end()
                self.emit(END)
                return
            if isinstance(event, Closed):
                await self.on_close(event.prefix)
            else:
                await self.on_row(event.row, isinstance(event, RowUpdated))


class AsyncOperator(Operator):
    """Runs work concurrently while keeping closes behind the rows they cover."""

    def __init__(self, on_error: str = "skip") -> None:
        super().__init__()
        if on_error not in ("skip", "raise"):
            raise ValueError("on_error must be 'skip' or 'raise'")
        self.on_error = on_error
        self.pending: dict[asyncio.Task, Key] = {}
        self.failure: BaseException | None = None

    def spawn(self, coro, key: Key) -> None:
        task = asyncio.create_task(coro)
        self.pending[task] = key
        task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task) -> None:
        self.pending.pop(task, None)
        if not task.cancelled() and task.exception() and self.failure is None:
            self.failure = task.exception()

    def waiting_on(self, prefix: Key) -> list[asyncio.Task]:
        return [t for t, k in self.pending.items() if is_prefix(prefix, k)]

    async def on_close(self, prefix: Key) -> None:
        self.spawn(self._close_after(self.waiting_on(prefix), prefix), prefix)

    async def _close_after(self, tasks: list[asyncio.Task], prefix: Key) -> None:
        await asyncio.gather(*tasks)
        self.emit(Closed(prefix))

    async def on_end(self) -> None:
        while self.pending:
            await asyncio.wait(list(self.pending))
        if self.failure:
            raise self.failure

    async def guarded(self, what: str, coro) -> Any:
        """Await `coro`, applying the operator's error policy. Returns None on a skipped error."""
        try:
            return await coro
        except Exception as exc:
            if self.on_error == "raise":
                raise
            self.executor.stats.errors += 1
            log.warning("tracepress: skipping %s after error: %s: %s", what, type(exc).__name__, exc)
            return None


# --- Plain Python operators -------------------------------------------------------------


class MapOp(Operator):
    def __init__(self, fn: Callable[[Row], dict[str, Any]]):
        super().__init__()
        self.fn = fn

    async def on_row(self, row: Row, updated: bool) -> None:
        self.emit_row(Row(row.key, {**row.values, **self.fn(row)}, row.final, row.version), updated)


class FilterOp(Operator):
    def __init__(self, pred: Callable[[Row], bool]):
        super().__init__()
        self.pred = pred

    async def on_row(self, row: Row, updated: bool) -> None:
        if self.pred(row):
            self.emit_row(row, updated)


class SelectOp(Operator):
    def __init__(self, columns: list[str]):
        super().__init__()
        self.columns = columns

    async def on_row(self, row: Row, updated: bool) -> None:
        values = {c: row.values[c] for c in self.columns if c in row.values}
        self.emit_row(Row(row.key, values, row.final, row.version), updated)


def explode_rows(parent: Row, items: list, level: str, column: str, keep: list[str]) -> list[Row]:
    carried = {c: parent.get(c) for c in keep}
    rows = []
    for i, item in enumerate(items or []):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        values = {**carried, **item} if isinstance(item, dict) else {**carried, column: item}
        rows.append(Row((*parent.key, (level, i)), values, parent.final, parent.version))
    return rows


class ExplodeOp(Operator):
    def __init__(self, column: str, level: str, keep: list[str]):
        super().__init__()
        self.column, self.level, self.keep = column, level, keep

    async def on_row(self, row: Row, updated: bool) -> None:
        for child in explode_rows(row, row.get(self.column), self.level, self.level, self.keep):
            self.emit_row(child, updated)
        # Everything under this row has now been emitted.
        self.emit(Closed(row.key))


# --- LM operators -----------------------------------------------------------------------


class LMRowOp(AsyncOperator):
    """Shared plumbing for lm_map / lm_filter / lm_extract: one LM call per row."""

    def __init__(self, sig, lm=None, module: type[dspy.Module] = dspy.Predict, on_error: str = "skip"):
        super().__init__(on_error)
        self.sig = sig
        self.inputs = sigs.inputs(sig)
        self.lm = resolve_lm(lm)
        self.module = module(sig)

    async def on_row(self, row: Row, updated: bool) -> None:
        self.spawn(self._run(row, updated), row.key)

    async def _run(self, row: Row, updated: bool) -> None:
        inputs = {name: row.get(name) for name in self.inputs}
        pred = await self.guarded(f"row {row.key}", self.executor.call(self.module, self.lm, **inputs))
        if pred is not None:
            self.handle(row, pred, updated)

    def handle(self, row: Row, pred: dspy.Prediction, updated: bool) -> None:
        raise NotImplementedError


class LMMapOp(LMRowOp):
    def handle(self, row, pred, updated):
        new = {name: pred[name] for name in self.sig.output_fields}
        self.emit_row(Row(row.key, {**row.values, **new}, row.final, row.version), updated)


class LMFilterOp(LMRowOp):
    def handle(self, row, pred, updated):
        if pred[sigs.bool_output(self.sig)]:
            self.emit_row(row, updated)


class LMExtractOp(LMRowOp):
    def __init__(self, sig, level: str, keep: list[str], **kwargs):
        super().__init__(sig, **kwargs)
        self.level, self.keep = level, keep
        self.output = sigs.list_output(sig)

    def handle(self, row, pred, updated):
        for child in explode_rows(row, pred[self.output], self.level, self.level, self.keep):
            self.emit_row(child, updated)
        self.emit(Closed(row.key))


# --- Grouping ---------------------------------------------------------------------------


class GroupOp(AsyncOperator):
    """Buffers rows per group and reduces each group when it closes.

    A group key is a hierarchy prefix (`hier_levels`) plus optional extra column values.
    A group closes when a close arrives for any prefix of its hierarchy part, or at the
    end of the stream.
    """

    def __init__(self, hier_levels: tuple[str, ...], extra: tuple[str, ...], on_error: str = "skip"):
        super().__init__(on_error)
        self.depth = len(hier_levels)
        self.extra = extra
        self.groups: dict[Key, dict[Key, Row]] = {}

    def group_key(self, row: Row) -> Key:
        return row.key[: self.depth] + tuple((name, row.get(name)) for name in self.extra)

    async def on_row(self, row: Row, updated: bool) -> None:
        self.groups.setdefault(self.group_key(row), {})[row.key] = row

    async def on_close(self, prefix: Key) -> None:
        if len(prefix) > self.depth:
            return  # finer than our grain: says nothing about whole groups
        for gkey in [g for g in self.groups if is_prefix(prefix, g[: self.depth])]:
            self._start(gkey)
        await super().on_close(prefix)

    async def on_end(self) -> None:
        for gkey in list(self.groups):
            self._start(gkey)
        await super().on_end()

    def _start(self, gkey: Key) -> None:
        rows = sorted(self.groups.pop(gkey).values(), key=lambda r: sort_key(r.key))
        self.spawn(self._reduce(gkey, rows), gkey)

    async def _reduce(self, gkey: Key, rows: list[Row]) -> None:
        values = await self.guarded(f"group {gkey}", self.reduce(gkey, rows))
        if values is not None:
            self.emit_row(Row(gkey, values))

    async def reduce(self, gkey: Key, rows: list[Row]) -> dict[str, Any]:
        raise NotImplementedError


class PyReduceOp(GroupOp):
    def __init__(self, fn: Callable[[list[Row]], dict[str, Any]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn

    async def reduce(self, gkey, rows):
        return self.fn(rows)


class LMReduceOp(GroupOp):
    def __init__(
        self,
        sig,
        key_inputs: dict[str, Any],  # group-key input -> type
        child_columns: list[str] | None,  # None = whole records (the `rows` input)
        children_name: str,
        hier_levels,
        extra,
        lm=None,
        module: type[dspy.Module] = dspy.Predict,
        chunk_chars: int = 120_000,
        on_error: str = "skip",
    ):
        super().__init__(hier_levels, extra, on_error)
        self.key_inputs = key_inputs
        self.child_columns = child_columns
        self.children_name = children_name
        self.chunk_chars = chunk_chars
        self.lm = resolve_lm(lm)
        reduce_sig = sigs.reduce_signature(sig, key_inputs, child_columns or [sigs.ROWS], children_name)
        self.outputs_names = list(sig.output_fields)
        self.reducer = module(reduce_sig)
        self.merger = module(sigs.merge_signature(reduce_sig, key_inputs, children_name))

    def child_record(self, row: Row) -> dict[str, Any]:
        if self.child_columns is None:
            return dict(row.values)
        return {c: row.get(c) for c in self.child_columns}

    async def reduce(self, gkey, rows):
        keys = {name: dict(gkey)[name] for name in self.key_inputs}
        children = [self.child_record(r) for r in rows]
        chunks = _chunk(children, self.chunk_chars)
        partials = await asyncio.gather(
            *(self._call(self.reducer, keys, {self.children_name: c}) for c in chunks)
        )
        # Tree-merge until one result remains.
        while len(partials) > 1:
            groups = _chunk([_plain(p) for p in partials], self.chunk_chars, min_size=2)
            partials = await asyncio.gather(*(self._call(self.merger, keys, {"partials": g}) for g in groups))
        return partials[0]

    async def _call(self, module, keys, extra) -> dict[str, Any]:
        pred = await self.executor.call(module, self.lm, **keys, **extra)
        return {name: pred[name] for name in self.outputs_names}


def _plain(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in values.items()}


def _size(item: Any) -> int:
    return len(json.dumps(item, default=str))


def _chunk(items: list, budget: int, min_size: int = 1) -> list[list]:
    """Greedy split into chunks under `budget` rendered chars (a single item may exceed it).

    `min_size=2` guarantees progress when merging: each chunk combines at least two partials.
    """
    chunks, current, used = [], [], 0
    for item in items:
        size = _size(item)
        if current and used + size > budget and len(current) >= min_size:
            chunks.append(current)
            current, used = [], 0
        current.append(item)
        used += size
    if current:
        if chunks and len(current) < min_size:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks or [[]]
