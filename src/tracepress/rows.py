"""Rows, keys and the events that flow between operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A key is a tuple of (level, id) pairs, e.g. (("task", "cad-model"), ("trial", "cad-model__084252e2")).
Key = tuple[tuple[str, Any], ...]


def key_levels(key: Key) -> tuple[str, ...]:
    return tuple(level for level, _ in key)


def is_prefix(prefix: Key, key: Key) -> bool:
    return key[: len(prefix)] == prefix


def sort_key(key: Key) -> tuple:
    """Order keys deterministically, numbers numerically and everything else as strings."""
    return tuple(
        (level, 0, ident, "") if isinstance(ident, (int, float)) else (level, 1, 0, str(ident))
        for level, ident in key
    )


@dataclass(frozen=True)
class Row:
    key: Key
    values: dict[str, Any] = field(default_factory=dict)
    final: bool = True
    version: int = 0
    # The rows this one was built from, recorded only by `collect(lineage=True)`: a reduce's
    # children, or () for a source row. Row-by-row operators pass their input's through.
    _sources: tuple[Row, ...] | None = field(default=None, compare=False, repr=False)

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.
        try:
            return self.get(name)
        except KeyError:
            raise AttributeError(name) from None

    def get(self, name: str, default: Any = KeyError) -> Any:
        """Look up a column, falling back to a key level of the same name."""
        if name in self.values:
            return self.values[name]
        for level, ident in self.key:
            if level == name:
                return ident
        if default is KeyError:
            raise KeyError(name)
        return default

    def inputs(self) -> list[Row]:
        """The rows this row was built from: a reduce's children, and theirs in turn.

        Row-by-row steps (maps, joins, filters) are transparent, so a per-task row's inputs
        are its trace summaries, and a trace summary's inputs are its steps. Source rows
        have none. Joins follow their left side. Needs `collect(lineage=True)`.
        """
        if self._sources is None:
            raise ValueError("this row has no lineage; collect with `collect(lineage=True)` to record it")
        return list(self._sources)

    def to_dict(self) -> dict[str, Any]:
        return {**dict(self.key), **self.values}

    def __repr__(self) -> str:
        key = "/".join(str(ident) for _, ident in self.key)
        cols = ", ".join(f"{k}={_short(v)}" for k, v in self.values.items())
        return f"Row({key}: {cols})"


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass(frozen=True)
class RowAdded:
    row: Row


@dataclass(frozen=True)
class RowUpdated:
    row: Row


@dataclass(frozen=True)
class Closed:
    """No more rows will ever arrive under this key prefix."""

    prefix: Key


Event = RowAdded | RowUpdated | Closed
