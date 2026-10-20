"""Global defaults: which model the operators use.

    tracepress.configure(lm=Model("openrouter:openai/gpt-6-luna"))
    with tracepress.using(lm=Model("anthropic/claude-haiku-4-5")):
        df.lm_map(...)

``using`` overrides are context-local, so they nest and don't leak across threads.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

import dspy

from .model import cache_dir

if TYPE_CHECKING:
    from .model import Model

_defaults: dict[str, "Model | None"] = {"lm": None}
_overrides: ContextVar[dict[str, "Model | None"]] = ContextVar("tracepress_overrides", default={})


def configure(*, lm: "Model | None" = None) -> None:
    """Set the process-wide default model. Left as None, it is unchanged.

    Also points DSPy's response cache at ``~/.tracepress/cache`` for this session.
    """
    dspy.configure_cache(disk_cache_dir=str(cache_dir()))
    if lm is not None:
        _defaults["lm"] = lm


@contextmanager
def using(*, lm: "Model | None" = None) -> Iterator[None]:
    """Temporarily override the default model inside a ``with`` block."""
    changes = {"lm": lm} if lm is not None else {}
    token = _overrides.set({**_overrides.get(), **changes})
    try:
        yield
    finally:
        _overrides.reset(token)


def _get(key: str) -> "Model | None":
    return _overrides.get().get(key) or _defaults[key]


def get_lm(lm: "Model | None" = None) -> "Model":
    """The model an operator should use: the explicit one, else the configured default."""
    lm = lm or _get("lm")
    if lm is None:
        raise RuntimeError("No model configured. Call tracepress.configure(lm=tracepress.Model(...)) or pass lm=.")
    return lm
