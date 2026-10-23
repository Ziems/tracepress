"""TracePress: LLM-powered map/reduce over agent traces, for batch and streaming data."""

from . import pipelines, read
from .adapters import Adapter, from_pandas, from_rows, replay
from .lm import configure
from .plan import GroupedTable, Table, acollect, collect, every, rows_to_pandas
from .rows import Closed, Row, RowAdded, RowUpdated
from .schema import Schema


def last_run():
    """Stats (LM calls, tokens, errors, time) for the most recent run."""
    from . import runtime

    return runtime.last_run


__all__ = [
    "Adapter",
    "Closed",
    "GroupedTable",
    "Row",
    "RowAdded",
    "RowUpdated",
    "Schema",
    "Table",
    "acollect",
    "collect",
    "configure",
    "every",
    "from_pandas",
    "from_rows",
    "last_run",
    "pipelines",
    "read",
    "replay",
    "rows_to_pandas",
]
