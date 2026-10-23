"""Physical operators. Each one consumes events from one input and emits events downstream.

Ordering invariant: an operator forwards `Closed(prefix)` only after every row it will
ever emit under that prefix has been emitted. Async operators enforce this by making the
close wait on all of their in-flight work under the prefix.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dspy

from . import signatures as sigs
from .lm import Executor, config, resolve_lm
from .rows import Closed, Event, Key, Row, RowAdded, RowUpdated, is_prefix, sort_key

log = logging.getLogger("tracepress")

END = object()  # end-of-stream marker


def supersedes(new: Row, old: Row | None) -> bool:
    """Whether `new` should replace `old` for the same key.

    Later versions win; a repeat of the same version with the same values is a duplicate.
    """
    if old is None:
        return True
    if new.version != old.version:
        return new.version > old.version
    return new.values != old.values


class Operator:
    def __init__(self) -> None:
        self.outputs: list[asyncio.Queue] = []
        self.executor: Executor | None = None
        self.live = False  # set by the runtime: whether any upstream source is live

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
        if row.final:
            self.emit(Closed(row.key))  # everything under this row has now been emitted


# --- LM operators -----------------------------------------------------------------------


class LMRowOp(AsyncOperator):
    """Shared plumbing for lm_map / lm_filter / lm_extract: one LM call per row.

    A row whose key was already seen with the same version and values (a duplicate
    delivery) or a higher version (a stale update) is skipped, so duplicates cost nothing.
    """

    def __init__(self, sig, lm=None, module: type[dspy.Module] = dspy.Predict, on_error: str = "skip"):
        super().__init__(on_error)
        self.sig = sig
        self.inputs = sigs.inputs(sig)
        self.lm = resolve_lm(lm)
        self.module = module(sig)
        self.seen: dict[Key, Row] = {}

    async def on_row(self, row: Row, updated: bool) -> None:
        if not supersedes(row, self.seen.get(row.key)):
            return
        self.seen[row.key] = row
        self.spawn(self._run(row, updated), row.key)

    async def on_close(self, prefix: Key) -> None:
        # Nothing more arrives under a closed prefix, so its keys can be forgotten.
        for key in [k for k in self.seen if is_prefix(prefix, k)]:
            del self.seen[key]
        await super().on_close(prefix)

    async def _run(self, row: Row, updated: bool) -> None:
        inputs = {name: row.get(name) for name in self.inputs}
        pred = await self.guarded(f"row {row.key}", self.executor.call(self.module, self.lm, inputs))
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
        if row.final:
            self.emit(Closed(row.key))


# --- Grouping ---------------------------------------------------------------------------

INF = float("inf")


@dataclass(eq=False)
class Group:
    rows: dict[Key, Row] = field(default_factory=dict)
    order: list[Key] = field(default_factory=list)  # child keys, sorted
    states: list[dict] = field(default_factory=list)  # state after each completed batch
    dirty: int = 0  # first batch whose state must be (re)computed
    invalidated_at: float = INF  # earliest batch touched while a step was in flight
    worker: asyncio.Task | None = None
    version: int = 0  # versions of this group's output emitted so far


class GroupOp(AsyncOperator):
    """Groups rows and produces one output row per group.

    A group key is a hierarchy prefix (`hier_levels`) plus optional extra column values.
    A group closes when a close arrives for any prefix of its hierarchy part, or at the
    end of the stream; `finalize` then produces the final row.

    Incremental operators also run `step` as children arrive, over batches of
    `batch` children in key order, keeping the state after each batch. A child that
    arrives out of order rewinds the computation to its batch, so the final state never
    depends on arrival order. When live, a provisional row (final=False) is emitted
    every `emit_every` children.
    """

    batch = 1

    def __init__(self, hier_levels: tuple[str, ...], extra: tuple[str, ...], on_error: str = "skip",
                 emit_every: int | None = None):
        super().__init__(on_error)
        self.depth = len(hier_levels)
        self.extra = extra
        self.emit_every = emit_every
        self.groups: dict[Key, Group] = {}

    @property
    def incremental(self) -> bool:
        return False

    def group_key(self, row: Row) -> Key:
        return row.key[: self.depth] + tuple((name, row.get(name)) for name in self.extra)

    async def on_row(self, row: Row, updated: bool) -> None:
        gkey = self.group_key(row)
        g = self.groups.setdefault(gkey, Group())
        old = g.rows.get(row.key)
        if not supersedes(row, old):
            return
        g.rows[row.key] = row
        if old is None:
            bisect.insort(g.order, row.key, key=sort_key)
        b = g.order.index(row.key) // self.batch
        g.dirty = min(g.dirty, b)
        g.invalidated_at = min(g.invalidated_at, b)
        del g.states[b:]
        if self.incremental and (g.worker is None or g.worker.done()):
            g.worker = asyncio.create_task(self._work(gkey, g))

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
        self.spawn(self._finish(gkey, self.groups.pop(gkey)), gkey)

    async def _work(self, gkey: Key, g: Group) -> None:
        while g.dirty < len(g.order) // self.batch:
            j = g.dirty
            prev = g.states[j - 1] if j else None
            batch = [g.rows[k] for k in g.order[j * self.batch : (j + 1) * self.batch]]
            g.invalidated_at = INF
            out = await self.guarded(f"group {gkey}", self.step(gkey, prev, batch))
            if g.invalidated_at <= j:
                continue  # a child landed in or before this batch while we waited: redo
            if out is None:  # skipped error: carry the previous state forward
                out = prev if prev is not None else self.empty_state()
            del g.states[j:]
            g.states.append(out)
            g.dirty = j + 1
            if self.live and self.emit_every and ((j + 1) * self.batch) % self.emit_every == 0:
                self._emit(gkey, g, self.provisional(out), final=False)

    async def _finish(self, gkey: Key, g: Group) -> None:
        if self.incremental:
            if g.worker is not None:
                await g.worker
            await self._work(gkey, g)  # nothing new can arrive now; make sure we're caught up
        rows = [g.rows[k] for k in g.order]
        values = await self.guarded(f"group {gkey}", self.finalize(gkey, rows, g.states))
        if values is not None:
            self._emit(gkey, g, values, final=True)

    def _emit(self, gkey: Key, g: Group, values: dict[str, Any], final: bool) -> None:
        g.version += 1
        self.emit_row(Row(gkey, values, final, g.version), updated=g.version > 1)

    async def step(self, gkey: Key, prev: dict | None, batch: list[Row]) -> dict[str, Any]:
        raise NotImplementedError

    def empty_state(self) -> dict[str, Any]:
        return {}

    def provisional(self, state: dict[str, Any]) -> dict[str, Any]:
        return state

    async def finalize(self, gkey: Key, rows: list[Row], states: list[dict]) -> dict[str, Any]:
        raise NotImplementedError


class PyReduceOp(GroupOp):
    def __init__(self, fn: Callable[[list[Row]], dict[str, Any]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn

    async def finalize(self, gkey, rows, states):
        return self.fn(rows)


class LMReduceOp(GroupOp):
    """One reduce per group at close (a tree of calls for large groups).

    With `emit_every`, a live stream also keeps a provisional result, updated with each
    batch of new children by a derived "running" signature. The final row always comes
    from a full reduce over every child, so it matches what batch mode produces.
    """

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
        emit_every: int | None = None,
    ):
        super().__init__(hier_levels, extra, on_error, emit_every)
        self.batch = emit_every or 1
        self.key_inputs = key_inputs
        self.child_columns = child_columns
        self.children_name = children_name
        self.chunk_chars = chunk_chars
        self.lm = resolve_lm(lm)
        reduce_sig = sigs.reduce_signature(sig, key_inputs, child_columns or [sigs.ROWS], children_name)
        self.outputs_names = list(sig.output_fields)
        self.reducer = module(reduce_sig)
        self.merger = module(sigs.merge_signature(reduce_sig, key_inputs, children_name))
        self.runner = module(sigs.running_reduce_signature(reduce_sig, key_inputs, children_name))

    @property
    def incremental(self) -> bool:
        # Provisional results only matter on a live stream; in batch this is a plain reduce.
        return self.live and self.emit_every is not None

    def child_record(self, row: Row) -> dict[str, Any]:
        if self.child_columns is None:
            return dict(row.values)
        return {c: row.get(c) for c in self.child_columns}

    def keys(self, gkey: Key) -> dict[str, Any]:
        return {name: dict(gkey)[name] for name in self.key_inputs}

    async def step(self, gkey, prev, batch):
        new = {f"new_{self.children_name}": [self.child_record(r) for r in batch]}
        return await self._call(self.runner, self.keys(gkey), {"current": _plain(prev) if prev else None, **new})

    async def finalize(self, gkey, rows, states):
        keys = self.keys(gkey)
        children = [self.child_record(r) for r in rows]
        chunks = _chunk(children, self.chunk_chars)
        partials = await asyncio.gather(
            *(self._call(self.reducer, keys, {self.children_name: c}, config().reduce_timeout) for c in chunks)
        )
        # Tree-merge until one result remains.
        while len(partials) > 1:
            groups = _chunk([_plain(p) for p in partials], self.chunk_chars, min_size=2)
            partials = await asyncio.gather(
                *(self._call(self.merger, keys, {"partials": g}, config().reduce_timeout) for g in groups)
            )
        return partials[0]

    async def _call(self, module, keys, extra, timeout: float | None = None) -> dict[str, Any]:
        pred = await self.executor.call(module, self.lm, {**keys, **extra}, timeout)
        return {name: pred[name] for name in self.outputs_names}


class FoldOp(GroupOp):
    """Carries state through a group's children one at a time, in key order.

    The final row is the state after the last child. Provisional rows (live streams
    only) show the state so far.
    """

    def __init__(
        self,
        spec: sigs.FoldSpec,
        init: dict[str, Any],
        key_inputs: dict[str, Any],
        child_inputs: dict[str, Any],
        child_name: str,
        hier_levels,
        extra,
        lm=None,
        module: type[dspy.Module] = dspy.Predict,
        on_error: str = "skip",
        emit_every: int | None = None,
    ):
        super().__init__(hier_levels, extra, on_error, emit_every)
        self.spec = spec
        self.init = init
        self.key_inputs = key_inputs
        self.lm = resolve_lm(lm)
        self.module = module(sigs.fold_signature(spec, key_inputs, child_inputs, child_name))

    @property
    def incremental(self) -> bool:
        return True

    def empty_state(self) -> dict[str, Any]:
        return {**{name: None for name in self.spec.outputs}, **self.init}

    async def step(self, gkey, prev, batch):
        state = prev if prev is not None else self.init
        (row,) = batch
        inputs = {}
        for name in self.spec.inputs:
            if name in self.spec.state:
                inputs[self.spec.previous(name)] = state[name]
            elif name in self.key_inputs:
                inputs[name] = dict(gkey)[name]
            else:
                inputs[name] = row.get(name)
        pred = await self.executor.call(self.module, self.lm, inputs)
        return {name: pred[name] for name in self.spec.outputs}

    async def finalize(self, gkey, rows, states):
        return states[-1] if states else self.empty_state()


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
