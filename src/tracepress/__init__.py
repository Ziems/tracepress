"""TracePress: LLM-powered map/reduce over agent traces, for batch and streaming data."""

from . import pipelines, read, watch
from .adapters import Adapter, from_pandas, from_rows, replay
from .lm import configure, prices
from .plan import GroupedTable, Table, acollect, collect, every, rows_to_pandas
from .rows import Closed, Row, RowAdded, RowUpdated
from .schema import Schema
from .server import connect


def last_run():
    """Stats (LM calls, tokens, errors, time) for the most recent run, even one still going.

    `.status` is "running", "completed", "failed" or "cancelled"; `.error` is what ended it.
    """
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
    "connect",
    "every",
    "from_pandas",
    "from_rows",
    "last_run",
    "pipelines",
    "prices",
    "read",
    "replay",
    "rows_to_pandas",
    "watch",
]
