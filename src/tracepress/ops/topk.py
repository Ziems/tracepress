"""``lm_topk``: the K rows that best match a criterion.

    reviews.lm_topk("review", "Most likely to discourage a buyer", k=3)

Listwise ranking in rounds: rows are split into batches, each batch is ranked in one
call, the top K of every batch go on to the next round, and the last round ranks the
survivors. Calls per round run concurrently.
"""

from __future__ import annotations

from typing import Any

import dspy
import pandas as pd

from ..model import Model, total_usage
from ..run import record
from ..settings import get_lm
from ..signature import check, describe, to_signature
from ._common import many_rows_signature, numbered, rendered

RANK = many_rows_signature(
    {"ranking": (list[int], dspy.OutputField(desc="record numbers, best first, each exactly once"))},
    "Rank the numbered records from best to worst match for the criterion. Compare them with each other first.",
    criterion="what makes a record a better match",
)


def _ranked(order: list[int], n: int) -> list[int]:
    """Clean a model's ranking of 1..n: drop invalid and repeated numbers, append any missed."""
    seen: list[int] = []
    for i in order:
        if isinstance(i, int) and 1 <= i <= n and i not in seen:
            seen.append(i)
    return seen + [i for i in range(1, n + 1) if i not in seen]


def lm_topk(
    df: pd.DataFrame,
    signature: Any,
    criterion: str,
    k: int,
    *,
    batch_size: int = 20,
    lm: Model | None = None,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Return the ``k`` best rows by ``criterion``, best first, with a ``rank`` column (1 = best).

    Args:
        signature: the columns to compare, e.g. ``"product, review"``.
        criterion: what "best" means, e.g. ``"Most severe complaint"``.
    """
    sig = to_signature(signature, outputs=False)
    check(sig, df.columns)
    lm = get_lm(lm)
    if k < 1:
        raise ValueError("k must be at least 1")
    batch_size = max(batch_size, k + 1, 2)
    docs = rendered(sig, df)

    candidates = list(range(len(df)))  # positions into df
    usages = []
    rounds = 0
    while True:
        batches = [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]
        preds = lm.predict(
            RANK,
            [{"criterion": criterion, "records": numbered([docs[pos] for pos in b])} for b in batches],
            desc=f"{save_as or 'lm_topk'} round {rounds}",
        )
        usages += [p.usage for p in preds]
        rounds += 1
        ranked = [[b[j - 1] for j in _ranked(p.outputs["ranking"], len(b))] for b, p in zip(batches, preds)]
        if len(batches) == 1:
            final = ranked[0][:k]
            break
        candidates = [pos for r in ranked for pos in r[:k]]

    result = df.iloc[final].copy()
    result["rank"] = range(1, len(final) + 1)

    return record(
        result,
        op="topk",
        name=save_as,
        inputs=[df],
        instruction=criterion,
        model=lm.model,
        usage=total_usage(usages),
        params={"signature": describe(sig), "k": k, "rounds": rounds},
    )
