"""``lm_filter``: keep the rows for which a condition holds.

    df.lm_filter("review", "Describes a leak or drip")
"""

from __future__ import annotations

from typing import Any

import dspy
import pandas as pd

from ..model import Model, total_usage
from ..run import record
from ..settings import get_lm
from ..signature import check, describe, inputs, to_signature
from ._common import rows


def lm_filter(
    df: pd.DataFrame,
    signature: Any,
    condition: str,
    *,
    explain: bool = False,
    lm: Model | None = None,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Keep the rows where ``condition`` holds. The index is preserved.

    Args:
        signature: the columns to read, e.g. ``"product, review"``.
        condition: a statement about the record, e.g. ``"Describes a leak or drip"``.
        explain: ask for a one-sentence explanation first, kept as an ``explanation``
            column. Costs more tokens and usually judges better.
    """
    sig = to_signature(signature, outputs=False)
    check(sig, df.columns)
    lm = get_lm(lm)
    judge = sig.with_instructions(
        f"Decide whether this condition holds for the record.\nCondition: {condition}"
    )
    if explain:
        judge = judge.append("explanation", dspy.OutputField(desc="one sentence on whether the condition holds"), type_=str)
    judge = judge.append("keep", dspy.OutputField(desc="true only if the condition holds"), type_=bool)

    preds = lm.predict(judge, [inputs(sig, r) for r in rows(df)], desc=save_as or "lm_filter")
    mask = [bool(p.outputs["keep"]) for p in preds]
    result = df.copy()
    if explain:
        result["explanation"] = [p.outputs["explanation"] for p in preds]
    result = result[mask]

    return record(
        result,
        op="filter",
        name=save_as,
        inputs=[df],
        instruction=condition,
        model=lm.model,
        usage=total_usage([p.usage for p in preds]),
        params={"signature": describe(sig), "explain": explain, "kept": int(sum(mask)), "of": len(mask)},
    )
