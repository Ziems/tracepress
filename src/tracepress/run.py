"""The active pipeline run, if any, and how operators report their outputs to it.

While a workspace runs a pipeline, every ``df.lm_*`` operator hands its output DataFrame to
``record``, which saves it as an artifact. Outside a run, ``record`` does not save. Lineage
rides on ``DataFrame.attrs["tracepress_artifact"]``: an operator reads it from its input to
name the parent, then stamps its own output.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from .telemetry import note_lm_op

if TYPE_CHECKING:
    from .model import Usage
    from .workspace import Run

ATTR = "tracepress_artifact"

_current: ContextVar["Run | None"] = ContextVar("tracepress_run", default=None)


def current_run() -> "Run | None":
    return _current.get()


def artifact_of(df: pd.DataFrame) -> str | None:
    """The ``run_id/name`` of the artifact this frame is (or was derived from), if any."""
    return df.attrs.get(ATTR) if isinstance(df, pd.DataFrame) else None


def record(
    df: pd.DataFrame,
    *,
    op: str,
    name: str | None,
    inputs: Iterable[pd.DataFrame] = (),
    instruction: str | None = None,
    model: str | None = None,
    usage: "Usage | None" = None,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Save an operator's output as an artifact of the active run. Returns ``df``.

    Outside a run, nothing is written. Row counts and usage are noted either way, for
    telemetry; the instruction and artifact name are not.
    """
    frames = tuple(inputs)
    note_lm_op(
        model=model,
        usage=usage,
        n_input_rows=len(frames[0]) if frames else 0,
        n_output_rows=len(df),
    )
    run = _current.get()
    if run is None:
        return df
    parents = [p for p in (artifact_of(d) for d in frames) if p]
    run.save(
        df,
        name=name,
        op=op,
        parents=parents,
        instruction=instruction,
        model=model,
        usage=usage.as_dict() if usage is not None else None,
        params=params,
    )
    return df
