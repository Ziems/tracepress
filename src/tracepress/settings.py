"""Global defaults: which model the operators use, and which one the agent uses.

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

_defaults: dict[str, "Model | None"] = {"lm": None, "agent_lm": None}
_overrides: ContextVar[dict[str, "Model | None"]] = ContextVar("tracepress_overrides", default={})


def configure(*, lm: "Model | None" = None, agent_lm: "Model | None" = None) -> None:
    """Set the process-wide default models. Arguments left as None are unchanged.

    Also points DSPy's response cache at ``~/.tracepress/cache`` for this session.
    """
    dspy.configure_cache(disk_cache_dir=str(cache_dir()))
    if lm is not None:
        _defaults["lm"] = lm
    if agent_lm is not None:
        _defaults["agent_lm"] = agent_lm


@contextmanager
def using(*, lm: "Model | None" = None, agent_lm: "Model | None" = None) -> Iterator[None]:
    """Temporarily override the default models inside a ``with`` block."""
    changes = {k: v for k, v in {"lm": lm, "agent_lm": agent_lm}.items() if v is not None}
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


def get_agent_lm(lm: "Model | None" = None) -> "Model":
    """The model the agent should use: explicit, else agent_lm, else lm."""
    lm = lm or _get("agent_lm") or _get("lm")
    if lm is None:
        raise RuntimeError("No agent model configured. Call tracepress.configure(agent_lm=tracepress.Model(...)).")
    return lm
