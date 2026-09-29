"""Pieces shared by the operators."""

from __future__ import annotations

from typing import Any

import dspy
import pandas as pd

from ..signature import render


def rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [dict(zip(map(str, df.columns), values)) for values in df.itertuples(index=False, name=None)]


def expand(df: pd.DataFrame, records: list[dict[str, Any]], names: list[str]) -> pd.DataFrame:
    """A copy of ``df`` with one column per name, filled from ``records`` (one per row)."""
    out = df.copy()
    for name in names:
        out[name] = [r.get(name) for r in records]
    return out


def numbered(docs: list[str], label: str = "") -> str:
    """Records for ops that show many rows in one call: ``[1]`` then its fields, and so on."""
    return "\n\n".join(f"[{label}{i}]\n{doc}" for i, doc in enumerate(docs, 1))


def many_rows_signature(outputs: dict[str, tuple[Any, Any]], instructions: str, **extra_inputs: str) -> type[dspy.Signature]:
    """A signature whose inputs are numbered ``records`` (plus any ``extra_inputs``, by description)."""
    fields: dict[str, tuple[Any, Any]] = {name: (str, dspy.InputField(desc=desc)) for name, desc in extra_inputs.items()}
    fields["records"] = (str, dspy.InputField(desc="numbered records, each introduced by [n]"))
    fields.update(outputs)
    return dspy.Signature(fields, instructions)


def rendered(sig: type[dspy.Signature], df: pd.DataFrame) -> list[str]:
    return [render(sig, r) for r in rows(df)]
