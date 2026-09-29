"""Small text helpers shared by signatures and operators."""

from __future__ import annotations

from typing import Any


def as_text(value: Any, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    if limit is not None and len(text) > limit:
        text = text[:limit] + " …[truncated]"
    return text
