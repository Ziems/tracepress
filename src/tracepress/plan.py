"""The lazy Table API. Each method adds a node to the logical plan and checks it.

Name checks happen here, at build time, so a typo in a signature fails before any LM call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import dspy

from . import lm as lm_config
from . import operators as ops
from . import runtime
from . import signatures as sigs
from .rows import Row
from .schema import Schema
from .signatures import SigLike


@dataclass(eq=False)
class Node:
    make: Callable[[], ops.Operator]
    parent: Node | None
    schema: Schema
    label: str
    pipeline: str | None = None  # set on the output nodes of built-in pipelines, for telemetry
    others: tuple[Node, ...] = ()  # further inputs, e.g. the right side of a join


class Table:
    def __init__(self, node: Node):
        self._node = node

    @property
    def schema(self) -> Schema:
        return self._node.schema

    def __repr__(self) -> str:
        cols = "?" if self.schema.open else ", ".join(self.schema.columns)
        return f"Table[{' > '.join(self.schema.levels)}]({cols})"

    def _then(self, make, schema: Schema, label: str) -> Table:
        return Table(Node(make, self._node, schema, label))

    # --- Plain Python operators ---------------------------------------------------------

    def map(self, fn: Callable[[Row], dict[str, Any]], columns: dict[str, Any] | None = None) -> Table:
        """Add or replace columns with `fn(row) -> dict`. Pass `columns` to keep schema checks."""
        clash = [c for c in columns or () if c in self.schema.levels]
        if clash:
            raise ValueError(f"map: column(s) {clash} would shadow key level(s) of the same name")
        # `fn` may overwrite any column, so only columns it declares keep their grain.
        grain = {c: d for c, d in self.schema.grain.items() if columns is not None and c not in columns}
        schema = replace(self.schema, columns=None, grain=grain)
        if columns is not None and not self.schema.open:
            schema = replace(schema, columns={**self.schema.columns, **columns})
        return self._then(lambda: ops.MapOp(fn), schema, "map")

    def filter(self, pred: Callable[[Row], bool]) -> Table:
        return self._then(lambda: ops.FilterOp(pred), self.schema, "filter")

    def select(self, *columns: str) -> Table:
        self.schema.require(columns, "select")
        known = self.schema.columns or {}
        record = tuple(c for c in self.schema.record if c in columns) if self.schema.record is not None else None
        schema = Schema(self.schema.levels, {c: known.get(c, Any) for c in columns if c not in self.schema.levels},
                        {c: d for c, d in self.schema.grain.items() if c in columns}, record or None)
        return self._then(lambda: ops.SelectOp(list(columns)), schema, "select")

    def explode(self, column: str, level: str | None = None, keep: tuple[str, ...] = ()) -> Table:
        """One row per element of a list column, one level deeper. Elements land in a column named `level`."""
        self.schema.require([column, *keep], "explode")
        level = level or column.rstrip("s")
        known = self.schema.columns or {}
        schema = Schema((*self.schema.levels, level), {**{k: known.get(k, Any) for k in keep}, level: Any},
                        self._kept_grain(keep))
        return self._then(lambda: ops.ExplodeOp(column, level, list(keep)), schema, "explode")

    def _kept_grain(self, keep) -> dict[str, int]:
        depth = len(self.schema.levels)
        return {k: min(self.schema.grain.get(k, depth), depth) for k in keep}

    def join(self, other: Table) -> Table:
        """Add the columns of `other`'s matching row to each row.

        `other` must be keyed by a prefix of this table's levels, e.g. trials onto steps:
        `src.steps.join(src.trials)` gives every step its trial's reward and instruction.
        Columns this table already has are kept as they are.
        """
        left, right = self.schema, other.schema
        depth = len(right.levels)
        if left.levels[:depth] != right.levels:
            raise ValueError(f"join: the other table's levels {list(right.levels)} must be a prefix of this "
                             f"table's levels {list(left.levels)}")
        if left.open or right.open:
            columns = None
        else:
            columns = {**left.columns, **{c: t for c, t in right.columns.items() if c not in left.columns}}
        added = right.columns or {}
        grain = {**{c: min(right.grain.get(c, depth), depth) for c in added if c not in (left.columns or {})},
                 **left.grain}
        schema = replace(left, columns=columns, grain=grain)
        return Table(Node(lambda: ops.JoinOp(depth), self._node, schema, "join", others=(other._node,)))

    def group_by(self, *names: str) -> GroupedTable:
        return GroupedTable(self, names)

    # --- LM operators -------------------------------------------------------------------

    def lm_map(self, sig: SigLike, *, instructions: str | None = None, lm=None,
               module: type[dspy.Module] = dspy.Predict, on_error: str = "skip") -> Table:
        """One LM call per row; the signature's outputs become new columns."""
        parsed, records = self._row_inputs(sigs.parse(sig, instructions))
        self.schema.require(sigs.inputs(parsed), "lm_map")
        schema = self.schema.with_columns(sigs.outputs(parsed))
        return self._then(lambda: ops.LMMapOp(parsed, lm=lm, module=module, on_error=on_error, records=records),
                          schema, "lm_map")

    def lm_filter(self, sig: SigLike, *, instructions: str | None = None, lm=None,
                  module: type[dspy.Module] = dspy.Predict, on_error: str = "skip") -> Table:
        """Keep rows where the signature's single bool output is true."""
        parsed, records = self._row_inputs(sigs.parse(sig, instructions))
        sigs.bool_output(parsed)
        self.schema.require(sigs.inputs(parsed), "lm_filter")
        return self._then(lambda: ops.LMFilterOp(parsed, lm=lm, module=module, on_error=on_error, records=records),
                          self.schema, "lm_filter")

    def lm_extract(self, sig: SigLike, *, level: str | None = None, keep: tuple[str, ...] = (),
                   instructions: str | None = None, lm=None, module: type[dspy.Module] = dspy.Predict,
                   on_error: str = "skip") -> Table:
        """One row per element of the signature's single list output, one level deeper.

        `"trace -> failures: list[str]"` makes a `failure` level with a `failure` column.
        Elements that are objects become columns. `keep` carries parent columns along.
        """
        parsed, records = self._row_inputs(sigs.parse(sig, instructions))
        output = sigs.list_output(parsed)
        self.schema.require([*sigs.inputs(parsed), *keep], "lm_extract")
        level = level or output.rstrip("s")
        known = self.schema.columns or {}
        schema = Schema((*self.schema.levels, level), {**{k: known.get(k, Any) for k in keep}, level: Any},
                        self._kept_grain(keep))
        return self._then(
            lambda: ops.LMExtractOp(parsed, level, list(keep), lm=lm, module=module, on_error=on_error,
                                    records=records),
            schema, "lm_extract",
        )

    def _row_inputs(self, parsed):
        """An input named after the row's own level (`step` on a steps table) gets the whole row."""
        name = self.schema.row_name
        if name is None or name not in sigs.inputs(parsed):
            return parsed, {}
        return sigs.as_record(parsed, name), {name: self.schema.record}

    # --- Running ------------------------------------------------------------------------

    def events(self, max_concurrency: int | None = None):
        """Async iterator over every output event, provisional rows and closes included.

        `async for event in table.events(): ...`
        """
        return runtime.stream_events(self._node, max_concurrency)

    def collect(self, max_concurrency: int | None = None) -> list[Row]:
        """Run the plan to completion and return the final rows, sorted by key."""
        return runtime.run([self._node], max_concurrency)[0]

    async def acollect(self, max_concurrency: int | None = None) -> list[Row]:
        return (await runtime.run_async([self._node], max_concurrency))[0]

    def to_pandas(self, max_concurrency: int | None = None):
        return rows_to_pandas(self.collect(max_concurrency))


class GroupedTable:
    def __init__(self, table: Table, names: tuple[str, ...]):
        schema = table.schema
        schema.require(names, "group_by")
        levels = [n for n in names if n in schema.levels]
        depth = max((schema.levels.index(n) + 1 for n in levels), default=0)
        self._table = table
        self._hier = schema.levels[:depth]  # grouping by a level also keys by its ancestors
        self._extra = tuple(n for n in names if n not in schema.levels)
        self._key_names = [*self._hier, *self._extra]
        # Columns with one value per group (a trial's reward when grouping steps by trial)
        # become scalar inputs and are carried onto each output row.
        depth = len(self._hier)
        self._constants = [c for c in schema.grain if schema.constant(c, depth) and c not in self._key_names]

    def _then(self, make, columns: dict[str, Any] | None, label: str) -> Table:
        child = self._table.schema
        carry = tuple(c for c in self._constants if columns is None or c not in columns)
        if columns is not None:
            columns = {**columns, **{c: (child.columns or {}).get(c, Any) for c in carry}}
        schema = Schema(tuple(self._key_names), columns, {c: child.grain[c] for c in carry})

        def build():
            op = make()
            op.carry = carry
            return op

        return self._table._then(build, schema, label)

    def reduce(self, fn: Callable[[list[Row]], dict[str, Any]], columns: dict[str, Any] | None = None,
               on_error: str = "raise") -> Table:
        """Plain Python reduce: `fn(rows) -> dict` per group."""
        hier, extra = self._hier, self._extra
        return self._then(lambda: ops.PyReduceOp(fn, hier, extra, on_error=on_error), columns, "reduce")

    def _key_inputs(self, names) -> dict[str, Any]:
        """Scalar inputs: group keys and group constants, name -> type."""
        child = self._table.schema
        return {
            n: str if n in child.levels else (child.columns or {}).get(n, str)
            for n in names if n in self._key_names or n in self._constants
        }

    def _record(self, names) -> tuple[str, tuple[str, ...] | None] | None:
        """(child level, its columns) when an input names the child level, e.g. `step`."""
        child = self._table.schema
        name = child.row_name
        if name is None or name in self._key_names or name not in names:
            return None
        return name, child.record

    def lm_reduce(self, sig: SigLike, *, emit: str | every = "close", instructions: str | None = None, lm=None,
                  module: type[dspy.Module] = dspy.Predict, chunk_chars: int | None = None,
                  on_error: str = "skip") -> Table:
        """One LM call per group (a tree of calls for groups too big for one prompt).

        Inputs naming a group key, or a column with one value per group (a trial's reward
        on its steps), get that single value; such columns are also carried onto the output
        rows. Inputs naming child columns get the group's values as one aligned record per
        child. The child level's name (`step`) gets each child's whole row; `rows` gets
        whole records with every column. `group_by()` with no names reduces the whole table.

        `emit="update"` (or `tp.every(n)`) also emits provisional rows on live streams,
        updated as each child (or batch of n) arrives. The final row is the same full
        reduce either way. In batch mode `emit` changes nothing.
        """
        emit_every = _emit_every(emit)
        parsed = sigs.parse(sig, instructions)
        child = self._table.schema
        key_inputs = self._key_inputs(sigs.inputs(parsed))
        constants = tuple(n for n in key_inputs if n in self._constants)
        child_inputs = [n for n in sigs.inputs(parsed) if n not in key_inputs]
        record = self._record(child_inputs)
        if sigs.ROWS in child_inputs:
            if child_inputs != [sigs.ROWS]:
                raise ValueError("lm_reduce: `rows` already includes every column; don't list columns too")
            child_columns = None
        else:
            child.require(child_inputs, "lm_reduce")
            child_columns = child_inputs
        child_level = child.levels[-1] if child.levels else "row"
        children_name = f"{child_level}s"
        if children_name in sigs.inputs(parsed) or children_name in sigs.outputs(parsed):
            children_name = "items"
        hier, extra = self._hier, self._extra
        chunk = chunk_chars or lm_config.config().reduce_chunk_chars
        return self._then(
            lambda: ops.LMReduceOp(parsed, key_inputs, child_columns, children_name, hier, extra,
                                   lm=lm, module=module, chunk_chars=chunk, on_error=on_error,
                                   emit_every=emit_every, constants=constants, record=record),
            sigs.outputs(parsed), "lm_reduce",
        )

    def lm_fold(self, sig: str, *, init: dict[str, Any] | None = None, emit: str | every = "close",
                instructions: str | None = None, lm=None, module: type[dspy.Module] = dspy.Predict,
                on_error: str = "skip") -> Table:
        """Carry state through each group's children, one LM call per child, in key order.

        Fields on both sides of the arrow are state: `"progress, step -> progress, stuck: bool"`
        reads the previous `progress` and the newest child's `step`, and writes the new
        `progress` (plus `stuck`). The output row holds the state after the last child.

        `init` sets initial state (default: "", [], 0 or False by type). `emit="update"`
        or `tp.every(n)` also emits the state so far on live streams. Children arriving
        out of order are handled by recomputing from the earliest one affected.
        """
        emit_every = _emit_every(emit)
        spec = sigs.parse_fold(sig, instructions)
        child = self._table.schema
        key_inputs = self._key_inputs(n for n in spec.inputs if n not in spec.state)
        constants = tuple(n for n in key_inputs if n in self._constants)
        child_names = [n for n in spec.inputs if n not in spec.state and n not in key_inputs]
        child.require(child_names, "lm_fold")
        known = child.columns or {}
        record = self._record(child_names)
        child_inputs = {n: dict if record and n == record[0] else known.get(n, str) if n not in child.levels else str
                        for n in child_names}
        init = dict(init or {})
        unknown = set(init) - set(spec.state)
        if unknown:
            raise ValueError(f"lm_fold: init has non-state field(s) {sorted(unknown)}; state is {spec.state}")
        init = {name: init.get(name, sigs.default_value(spec.outputs[name])) for name in spec.state}
        child_name = child.levels[-1] if child.levels else "row"
        hier, extra = self._hier, self._extra
        return self._then(
            lambda: ops.FoldOp(spec, init, key_inputs, child_inputs, child_name, hier, extra,
                               lm=lm, module=module, on_error=on_error, emit_every=emit_every,
                               constants=constants, record=record),
            dict(spec.outputs), "lm_fold",
        )


class every:
    """Emit provisional results every `n` children: `emit=tp.every(5)`."""

    def __init__(self, n: int | str):
        if isinstance(n, str):
            raise NotImplementedError("time-based every('30s') isn't supported yet; pass a number of children")
        if n < 1:
            raise ValueError("every(n) needs n >= 1")
        self.n = n

    def __repr__(self) -> str:
        return f"every({self.n})"


def _emit_every(emit: str | every) -> int | None:
    if emit == "close":
        return None
    if emit == "update":
        return 1
    if isinstance(emit, every):
        return emit.n
    raise ValueError(f"emit must be 'close', 'update' or tp.every(n), not {emit!r}")


def collect(*tables: Table, max_concurrency: int | None = None) -> list[list[Row]]:
    """Run several tables in one pass; shared upstream work runs once."""
    return runtime.run([t._node for t in tables], max_concurrency)


async def acollect(*tables: Table, max_concurrency: int | None = None) -> list[list[Row]]:
    """`collect` for code already inside an event loop, e.g. Jupyter: `await tp.acollect(a, b)`."""
    return await runtime.run_async([t._node for t in tables], max_concurrency)


def rows_to_pandas(rows: list[Row]):
    import pandas as pd

    return pd.DataFrame([row.to_dict() for row in rows])
