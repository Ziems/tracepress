"""Table schemas: the key levels (grain) and the known columns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Schema:
    levels: tuple[str, ...]
    # Column name -> type. None means the columns are unknown (e.g. after a plain `map`),
    # so name checks are skipped downstream.
    columns: dict[str, Any] | None = field(default_factory=dict)

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
        return Schema(self.levels, {**self.columns, **new})
