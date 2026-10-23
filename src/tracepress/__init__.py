"""TracePress: LLM-powered map/reduce over agent traces, for batch and streaming data."""

from . import read
from .adapters import Adapter, from_pandas, from_rows, replay
from .lm import configure
from .plan import GroupedTable, Table, collect, rows_to_pandas
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
    "collect",
    "configure",
    "from_pandas",
    "from_rows",
    "last_run",
    "read",
    "replay",
    "rows_to_pandas",
]
