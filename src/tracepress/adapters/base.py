"""The adapter interface every source implements, plus an in-memory source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable

from ..plan import Node, Table
from ..rows import Closed, Event, Key, Row, RowAdded, sort_key
from ..schema import Schema


class Adapter(ABC):
    """A source of rows. Adapters decide their own hierarchy, tables and closures."""

    levels: tuple[str, ...]
    bounded: bool = True  # False for live sources

    @abstractmethod
    def tables(self) -> dict[str, Schema]: ...

    @abstractmethod
    def events(self, table: str) -> AsyncIterator[Event]:
        """RowAdded / Closed events for one table. A bounded source's iterator ends."""

    def table(self, name: str) -> Table:
        tables = self.tables()
        if name not in tables:
            raise KeyError(f"{type(self).__name__} has no table {name!r}; tables: {sorted(tables)}")
        from ..runtime import SourceOp

        return Table(Node(lambda: SourceOp(self, name), None, tables[name], f"{type(self).__name__}.{name}"))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(tables={sorted(self.tables())})"


class RowsAdapter(Adapter):
    """Rows already in memory, e.g. the results of an earlier `collect()`."""

    def __init__(self, rows: Iterable[Row], schema: Schema):
        self.rows = sorted(rows, key=lambda r: sort_key(r.key))
        self.schema = schema
        self.levels = schema.levels

    def tables(self) -> dict[str, Schema]:
        return {"rows": self.schema}

    async def events(self, table: str) -> AsyncIterator[Event]:
        prefixes: set[Key] = set()
        for row in self.rows:
            yield RowAdded(row)
            prefixes.update(row.key[:i] for i in range(1, len(row.key)))
        # Every row is already here, so every group is complete. Close deepest first.
        for prefix in sorted(prefixes, key=lambda p: (-len(p), sort_key(p))):
            yield Closed(prefix)


def from_rows(rows: list[Row]) -> Table:
    """Turn collected rows back into a static source."""
    rows = list(rows)
    levels = tuple(level for level, _ in rows[0].key) if rows else ()
    columns: dict = {}
    for row in rows:
        columns.update({c: object for c in row.values})
    return RowsAdapter(rows, Schema(levels, columns)).table("rows")


def from_pandas(df, levels: list[str]) -> Table:
    """Turn a DataFrame into a static source; `levels` are the key columns, outermost first."""
    rows = []
    for record in df.to_dict("records"):
        key = tuple((level, record.pop(level)) for level in levels)
        rows.append(Row(key, record))
    columns = {c: object for c in df.columns if c not in levels}
    return RowsAdapter(rows, Schema(tuple(levels), columns)).table("rows")
