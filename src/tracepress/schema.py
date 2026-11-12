"""Table schemas: the key levels (grain) and the known columns."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Schema:
    levels: tuple[str, ...]
    # Column name -> type. None means the columns are unknown (e.g. after a plain `map`),
    # so name checks are skipped downstream.
    columns: dict[str, Any] | None = field(default_factory=dict)
    # Column name -> the depth (number of key levels) at which it's constant. A trial's
    # reward on a steps table has depth 3 (dataset > task > trial). Columns not listed
    # can differ from row to row. Grouping at that depth or deeper lifts them to scalars.
    grain: dict[str, int] = field(default_factory=dict)
    # The columns that make up a row when a signature names the row's own level, e.g.
    # `"step -> summary"`. None means every column.
    record: tuple[str, ...] | None = None

    @property
    def open(self) -> bool:
        return self.columns is None

    def has(self, name: str) -> bool:
        return self.open or name in self.columns or name in self.levels

    def require(self, names, what: str) -> None:
        missing = [n for n in names if not self.has(n)]
        if missing:
            known = sorted(set(self.columns or ()) | set(self.levels))
            raise ValueError(f"{what}: unknown field(s) {missing}. Available: {known}")

    def with_columns(self, new: dict[str, Any]) -> Schema:
        if self.open:
            return self
        clash = [n for n in new if n in self.columns or n in self.levels]
        if clash:
            raise ValueError(f"output field(s) {clash} already exist in the table")
        return replace(self, columns={**self.columns, **new})

    def constant(self, name: str, depth: int) -> bool:
        """Whether `name` is a column with one value per key prefix of length `depth`."""
        return name not in self.levels and self.grain.get(name, len(self.levels) + 1) <= depth

    @property
    def row_name(self) -> str | None:
        """The level naming one row of this table (e.g. `step`), usable as a whole-row input."""
        return self.levels[-1] if self.levels else None
