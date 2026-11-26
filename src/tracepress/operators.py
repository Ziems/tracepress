"""Physical operators. Each one consumes events from one input and emits events downstream.

Ordering invariant: an operator forwards `Closed(prefix)` only after every row it will
ever emit under that prefix has been emitted. Async operators enforce this by making the
close wait on all of their in-flight work under the prefix.
"""

from __future__ import annotations

import asyncio
import bisect
import itertools
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

# Errors every later call would hit too. Skipping the row would only skip all the rest, so
# these stop the run whatever `on_error` says.
STOP_RUN = (dspy.LMBillingError, dspy.LMAuthError)


def supersedes(new: Row, old: Row | None) -> bool:
    """Whether `new` should replace `old` for the same key.

    Later versions win; a repeat of the same version with the same values is a duplicate.
    """
    if old is None:
        return True
    if new.version != old.version:
        return new.version > old.version
    return new.values != old.values


def whole_record(row: Row, columns: tuple[str, ...] | None) -> dict[str, Any]:
    """A row as one input value: its `columns` (all of them if None), empty ones dropped."""
    names = columns if columns is not None else tuple(row.values)
    return {c: v for c in names if (v := row.values.get(c)) not in (None, "", [], {})}


class Operator:
    lineage = False  # set by the runtime for `collect(lineage=True)`

    def __init__(self) -> None:
        self.outputs: list[asyncio.Queue] = []
        self.executor: Executor | None = None
        self.live = False  # set by the runtime: whether any upstream source is live
        self.label = type(self).__name__  # set by the runtime to the plan node's label, e.g. "lm_map"

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

    def cancel(self) -> list[asyncio.Task]:
        """Stop work still running when the run ends early; returns the tasks stopped."""
        return []

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

    def spawn(self, coro, key: Key) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.pending[task] = key
        task.add_done_callback(self._done)
        return task

    def cancel(self) -> list[asyncio.Task]:
        tasks = list(self.pending)
        for task in tasks:
            task.cancel()
        return tasks

    def _done(self, task: asyncio.Task) -> None:
        self.pending.pop(task, None)
        if not task.cancelled() and (failure := task.exception()) and self.failure is None:
            self.failure = failure
            # Stop the run now, not once this operator's input ends.
            stop = self.executor.stop if self.executor else None
            if stop is not None and not stop.done():
                stop.set_exception(failure)

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
            if self.on_error == "raise" or isinstance(exc, STOP_RUN):
                raise
            assert self.executor is not None
            self.executor.stats.errors += 1
            log.warning("tracepress: skipping %s after error: %s: %s", what, type(exc).__name__, exc)
            return None


# --- Plain Python operators -------------------------------------------------------------


class MapOp(Operator):
    def __init__(self, fn: Callable[[Row], dict[str, Any]]):
        super().__init__()
        self.fn = fn

    async def on_row(self, row: Row, updated: bool) -> None:
        self.emit_row(Row(row.key, {**row.values, **self.fn(row)}, row.final, row.version, row._sources), updated)


class EditOp(Operator):
    """Changes the values of chosen rows, for `Table.rerun(edit=...)`.

    `edits` maps a key suffix (e.g. `(("trial", "t1"),)`) to the values to set on the row
    whose key ends with it. An edit that matches no row fails the run at the end.
    """

    def __init__(self, edits: dict[Key, dict[str, Any]]):
        super().__init__()
        self.edits = edits
        self.seen: set[Key] = set()

    async def on_row(self, row: Row, updated: bool) -> None:
        for suffix, values in self.edits.items():
            if row.key[-len(suffix) :] == suffix:
                self.seen.add(suffix)
                row = Row(row.key, {**row.values, **values}, row.final, row.version, row._sources)
                break
        self.emit_row(row, updated)

    async def on_end(self) -> None:
        missing = [dict(k) for k in self.edits if k not in self.seen]
        if missing:
            raise ValueError(f"rerun: no row with key {missing[0]} to edit")


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
        self.emit_row(Row(row.key, values, row.final, row.version, row._sources), updated)


def explode_rows(parent: Row, items: list, level: str, column: str, keep: list[str]) -> list[Row]:
    carried = {c: parent.get(c) for c in keep}
    rows = []
    for i, item in enumerate(items or []):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        values = {**carried, **item} if isinstance(item, dict) else {**carried, column: item}
        rows.append(Row((*parent.key, (level, i)), values, parent.final, parent.version, parent._sources))
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

    def __init__(
        self,
        sig,
        lm=None,
        module: type[dspy.Module] = dspy.Predict,
        on_error: str = "skip",
        records: dict[str, tuple[str, ...] | None] | None = None,
    ):
        super().__init__(on_error)
        self.sig = sig
        self.inputs = sigs.inputs(sig)
        self.records = records or {}  # inputs that receive the whole row -> its columns
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
        inputs = {
            name: whole_record(row, self.records[name]) if name in self.records else row.get(name)
            for name in self.inputs
        }
        assert self.executor is not None
        pred = await self.guarded(f"row {row.key}", self.executor.call(self.module, self.lm, inputs, op=self.label))
        if pred is not None:
            self.handle(row, pred, updated)

    def handle(self, row: Row, pred: dspy.Prediction, updated: bool) -> None:
        raise NotImplementedError


class LMMapOp(LMRowOp):
    def handle(self, row, pred, updated):
        new = {name: pred[name] for name in self.sig.output_fields}
        self.emit_row(Row(row.key, {**row.values, **new}, row.final, row.version, row._sources), updated)


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
    carry: tuple[str, ...] = ()  # group constants copied onto each output row

    def __init__(
        self,
        hier_levels: tuple[str, ...],
        extra: tuple[str, ...],
        on_error: str = "skip",
        emit_every: int | None = None,
    ):
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
            g.worker = self.spawn(self._work(gkey, g), gkey)  # so a failure stops the run, as any other does

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
        if self.carry and g.order:
            first = g.rows[g.order[0]]
            values = {**values, **{c: first.get(c, None) for c in self.carry if c not in values}}
        g.version += 1
        sources = tuple(g.rows[k] for k in g.order) if self.lineage else None
        self.emit_row(Row(gkey, values, final, g.version, sources), updated=g.version > 1)

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
        key_inputs: dict[str, Any],  # scalar input (group key or group constant) -> type
        child_columns: list[str] | None,  # None = whole records (the `rows` input)
        children_name: str,
        hier_levels,
        extra,
        lm=None,
        module: type[dspy.Module] = dspy.Predict,
        chunk_chars: int = 120_000,
        on_error: str = "skip",
        emit_every: int | None = None,
        constants: tuple[str, ...] = (),  # key_inputs read from the children (one value per group)
        record: tuple[str, tuple[str, ...] | None] | None = None,  # (child level name, its columns)
    ):
        super().__init__(hier_levels, extra, on_error, emit_every)
        self.batch = emit_every or 1
        self.key_inputs = key_inputs
        self.constants = constants
        self.child_columns = child_columns
        self.record = record
        self.children_name = children_name
        self.chunk_chars = chunk_chars
        self.lm = resolve_lm(lm)
        described = child_columns or [sigs.ROWS]
        if record is not None and record[1] is not None:
            described = [c for name in described for c in (record[1] if name == record[0] else (name,))]
        reduce_sig = sigs.reduce_signature(sig, key_inputs, described, children_name)
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
        out: dict[str, Any] = {}
        for c in self.child_columns:
            if self.record is not None and c == self.record[0]:
                out.update(whole_record(row, self.record[1]))
            else:
                out[c] = row.get(c)
        return out

    def keys(self, gkey: Key, row: Row) -> dict[str, Any]:
        key = dict(gkey)
        return {name: row.get(name, None) if name in self.constants else key[name] for name in self.key_inputs}

    async def step(self, gkey, prev, batch):
        new = {f"new_{self.children_name}": [self.child_record(r) for r in batch]}
        keys = self.keys(gkey, batch[0])
        return await self._call(self.runner, keys, {"current": _plain(prev) if prev else None, **new})

    async def finalize(self, gkey, rows, states):
        keys = self.keys(gkey, rows[0]) if rows else {name: None for name in self.key_inputs}
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
        assert self.executor is not None
        pred = await self.executor.call(module, self.lm, {**keys, **extra}, timeout, op=self.label)
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
        constants: tuple[str, ...] = (),
        record: tuple[str, tuple[str, ...] | None] | None = None,
    ):
        super().__init__(hier_levels, extra, on_error, emit_every)
        self.spec = spec
        self.init = init
        self.key_inputs = key_inputs
        self.constants = constants
        self.record = record
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
            elif name in self.constants:
                inputs[name] = row.get(name, None)
            elif name in self.key_inputs:
                inputs[name] = dict(gkey)[name]
            elif self.record is not None and name == self.record[0]:
                inputs[name] = whole_record(row, self.record[1])
            else:
                inputs[name] = row.get(name)
        assert self.executor is not None
        pred = await self.executor.call(self.module, self.lm, inputs, op=self.label)
        return {name: pred[name] for name in self.spec.outputs}

    async def finalize(self, gkey, rows, states):
        return states[-1] if states else self.empty_state()


# --- Joins ------------------------------------------------------------------------------


class JoinOp(Operator):
    """Adds the columns of the matching row from a second table, the right side.

    Each row has at most one match. By default (`on=None`, a join on key levels) the match is
    the right row keyed by this row's key cut to the right's depth (a step's trial); rows wait
    for it until it arrives, or until the right side closes that key or ends. With `on` set to
    column names, the match is the right row with the same values in those columns; rows wait
    until the right side has ended, since any right row could still match.

    Rows with no match pass through unchanged (`inner=False`) or are dropped. `columns` limits
    what's copied; a copied column the row already has is renamed with `suffix`, or skipped
    when there's no suffix. A newer version of a right row re-emits the rows it was joined onto.
    """

    def __init__(
        self,
        depth: int,
        on: tuple[str, ...] | None = None,
        inner: bool = False,
        columns: tuple[str, ...] | None = None,
        suffix: str | None = None,
    ):
        super().__init__()
        self.depth = depth
        self.on = on
        self.inner = inner
        self.columns = columns
        self.suffix = suffix
        self.right: dict[Any, Row] = {}  # match key -> right row
        self.right_closed: set[Key] = set()  # prefixes the right side has closed
        self.right_done = asyncio.Event()
        self.waiting: dict[Any, list[tuple[Row, bool]]] = {}  # match key -> rows waiting for it
        self.waiting_under: dict[Key, int] = {}  # key prefix -> how many waiting rows are under it
        self.joined: dict[Any, dict[Key, Row]] = {}  # match key -> rows emitted, for re-emits
        self.joined_under: dict[Key, dict[Key, set]] = {}  # key prefix -> joined row keys under it -> their matches
        self.unsettled: set = set()  # joined matches whose right row can still change
        self.unsettled_under: dict[Key, int] = {}  # key prefix -> how many rows under it are joined to one
        self.held: dict[int, Key] = {}  # closes held back behind their rows, numbered in arrival order
        self.held_at: dict[Key, list[int]] = {}  # prefix -> the numbers of its held closes
        self.holds = itertools.count()

    def left_match(self, row: Row) -> Any:
        return row.key[: self.depth] if self.on is None else tuple(row.get(n, None) for n in self.on)

    def right_match(self, row: Row) -> Any:
        return row.key if self.on is None else tuple(row.get(n, None) for n in self.on)

    def ready(self, match: Any) -> bool:
        if self.right_done.is_set():
            return True
        if self.on is not None:
            return False  # a column join can't know a value is missing until the right side ends
        return match in self.right or self.closed_on_right(match)

    def closed_on_right(self, match: Any) -> bool:
        """Whether the right side closed `match` or a prefix of it: one set lookup per level."""
        return any(match[:i] in self.right_closed for i in range(len(match) + 1))

    def settled(self, match: Any) -> bool:
        """Whether the right row for `match` can no longer change, so rows joined to it are done."""
        if self.right_done.is_set():
            return True
        right = self.right.get(match)
        return (right is not None and right.final) or self.closed_on_right(match)

    def merge(self, row: Row) -> Row | None:
        right = self.right.get(self.left_match(row))
        if right is None:
            return None if self.inner else row
        values = dict(row.values)
        for name, value in right.values.items():
            if (self.columns is not None and name not in self.columns) or (self.on and name in self.on):
                continue
            if name in values:
                if self.suffix is None:
                    continue
                name += self.suffix
            values[name] = value
        return Row(row.key, values, row.final and right.final, row.version, row._sources)

    def send(self, row: Row, updated: bool) -> None:
        self.add_joined(self.left_match(row), row)
        merged = self.merge(row)
        if merged is not None:
            self.emit_row(merged, updated)

    async def on_row(self, row: Row, updated: bool) -> None:
        match = self.left_match(row)
        if self.ready(match):
            self.send(row, updated)
        else:
            self.add_waiting(match, row, updated)

    # The indexes change only through the methods below, so they always match the rows:
    # waiting_under[p] counts the waiting rows with prefix p, joined_under[p] holds the joined
    # rows with prefix p and their matches, and unsettled_under[p] counts the joined rows with
    # prefix p whose match is in `unsettled`.

    def add_waiting(self, match: Any, row: Row, updated: bool) -> None:
        self.waiting.setdefault(match, []).append((row, updated))
        _count(self.waiting_under, row.key, 1)

    def take_waiting(self, match: Any) -> list[tuple[Row, bool]]:
        rows = self.waiting.pop(match)
        for row, _ in rows:
            _count(self.waiting_under, row.key, -1)
        return rows

    def add_joined(self, match: Any, row: Row) -> None:
        if match not in self.joined and not self.settled(match):
            self.unsettled.add(match)
        rows = self.joined.setdefault(match, {})
        if row.key not in rows and match in self.unsettled:
            _count(self.unsettled_under, row.key, 1)
        rows[row.key] = row
        for i in range(len(row.key) + 1):
            self.joined_under.setdefault(row.key[:i], {}).setdefault(row.key, set()).add(match)

    def forget_joined(self, prefix: Key) -> None:
        """Forget the joined rows under `prefix`, under every match they were joined to."""
        for key, matches in self.joined_under.pop(prefix, {}).items():
            for match in matches:
                rows = self.joined[match]
                del rows[key]
                if match in self.unsettled:
                    _count(self.unsettled_under, key, -1)
                if not rows:
                    del self.joined[match]
                    self.unsettled.discard(match)
            for i in range(len(key) + 1):
                if key[:i] != prefix:
                    under = self.joined_under[key[:i]]
                    del under[key]
                    if not under:
                        del self.joined_under[key[:i]]

    def resettle(self, event: Any) -> list[Key]:
        """Recheck whether the matches a right-side event touches are settled.

        Returns the keys of the rows joined to them, whose closes may now go out.
        """
        if event is END:
            self.unsettled.clear()  # the right side has ended: every match is settled
            self.unsettled_under.clear()
            return []
        if isinstance(event, Closed):
            if self.on is None and len(event.prefix) >= self.depth:
                matches = [event.prefix] if len(event.prefix) == self.depth else []
            else:
                matches = [m for m in self.joined if is_prefix(event.prefix, m)]
        else:
            matches = [self.right_match(event.row)]
        keys = []
        for match in matches:
            rows = self.joined.get(match, {})
            unsettled = not self.settled(match)
            if rows and unsettled != (match in self.unsettled):
                (self.unsettled.add if unsettled else self.unsettled.discard)(match)
                for key in rows:
                    _count(self.unsettled_under, key, 1 if unsettled else -1)
            keys.extend(rows)
        return keys

    # A held close goes out once no row under its prefix is waiting or joined to a match that
    # can still change. Every release sends all the closes that can go, so the ones still held
    # are blocked, and only a change to a row under one can unblock it: a release need only
    # recheck the closes over the rows that changed.

    async def on_close(self, prefix: Key) -> None:
        number = next(self.holds)
        self.held[number] = prefix
        self.held_at.setdefault(prefix, []).append(number)
        self.release([prefix])

    def blocked(self, prefix: Key) -> bool:
        """Whether rows under `prefix` are still waiting for a match, or joined to one that can still change."""
        return bool(self.waiting_under.get(prefix) or self.unsettled_under.get(prefix))

    def release(self, changed: list[Key] | None = None) -> None:
        """Send each held close that is no longer blocked, in the order they arrived.

        `changed` holds the keys of every row that stopped waiting or whose match settled since
        the last release; None rechecks every held close. A key left out holds its close back.
        """
        if changed is None:
            numbers = list(self.held)
        else:
            numbers = sorted({n for key in changed for i in range(len(key) + 1) for n in self.held_at.get(key[:i], ())})
        for number in numbers:
            prefix = self.held[number]
            if self.blocked(prefix):
                continue
            del self.held[number]
            self.held_at[prefix].remove(number)
            if not self.held_at[prefix]:
                del self.held_at[prefix]
            self.emit(Closed(prefix))
            self.forget_joined(prefix)  # nothing more is emitted under a closed prefix

    def flush(self, candidates: list | None = None) -> list[Key]:
        """Send the waiting rows whose match is now ready, checking only `candidates` if given.

        Returns the keys of the rows sent.
        """
        matches = self.waiting if candidates is None else [m for m in candidates if m in self.waiting]
        sent = []
        for match in [k for k in matches if self.ready(k)]:
            for row, updated in self.take_waiting(match):
                self.send(row, updated)
                sent.append(row.key)
        return sent

    def unblocked_by(self, event: Any) -> list | None:
        """The waiting matches a right-side event can make ready, or None for all of them."""
        if event is END:
            return None
        if self.on is not None:
            return []  # a column join's rows all wait for the right side to end
        if isinstance(event, Closed):
            if len(event.prefix) >= self.depth:
                return [event.prefix] if len(event.prefix) == self.depth else []
            return [m for m in self.waiting if is_prefix(event.prefix, m)]
        return [self.right_match(event.row)]

    async def on_end(self) -> None:
        await self.right_done.wait()
        self.flush()
        self.release()

    async def consume_right(self, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            if event is END:
                self.right_done.set()
            elif isinstance(event, Closed):
                self.right_closed.add(event.prefix)
            else:
                row = event.row
                match = self.right_match(row)
                current = self.right.get(match)
                if self.on is not None and current is not None and current.key != row.key:
                    raise ValueError(
                        f"join: the other table has more than one row with {dict(zip(self.on, match))}, "
                        "so a row could match several. Join on columns that identify one row."
                    )
                if not supersedes(row, current):
                    continue
                self.right[match] = row
                if current is not None:
                    for left in self.joined.get(match, {}).values():
                        merged = self.merge(left)
                        if merged is not None:
                            self.emit_row(merged, updated=True)
            # Sending rows, resettling matches and releasing closes run with no await between
            # them, so no close is judged while the indexes are halfway through an event.
            sent = self.flush(self.unblocked_by(event))
            resettled = self.resettle(event)
            self.release(None if event is END else [*sent, *resettled])
            if event is END:
                return


def _count(counts: dict[Key, int], key: Key, delta: int) -> None:
    """Add `delta` to the count of every prefix of `key`, dropping counts that reach zero."""
    for i in range(len(key) + 1):
        count = counts.get(key[:i], 0) + delta
        if count < 0:
            raise AssertionError(f"join: the count under {key[:i]} went below zero")  # an index out of step
        if count:
            counts[key[:i]] = count
        else:
            del counts[key[:i]]


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
