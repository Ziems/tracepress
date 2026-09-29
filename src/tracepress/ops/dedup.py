"""``lm_dedup``: merge rows that the model judges to be the same thing. No embeddings.

    issues.lm_dedup("product, part, review", "Same root cause")

Rows are grouped into batches, and one call per batch lists the clusters of duplicates in
it. Clusters are merged with union-find. A later pass reshuffles the surviving
representatives into new batches, so duplicates that started in different batches can
still meet. With ``passes`` rounds this is not a guarantee of finding every duplicate
pair, but it is linear in calls rather than quadratic.
"""

from __future__ import annotations

import random
from typing import Any

import dspy
import pandas as pd

from ..model import Model, total_usage
from ..run import record
from ..settings import get_lm
from ..signature import check, describe, to_signature
from ._common import many_rows_signature, numbered, rendered

CLUSTER = many_rows_signature(
    {
        "clusters": (
            list[list[int]],
            dspy.OutputField(desc="groups of record numbers that are duplicates of each other; only groups of 2 or more"),
        )
    },
    "Find the numbered records that are duplicates of each other under the criterion: records that describe "
    "the same underlying thing, even if worded differently.",
    criterion="what counts as a duplicate",
)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # the earliest row represents the cluster


def lm_dedup(
    df: pd.DataFrame,
    signature: Any,
    criterion: str | None = None,
    *,
    batch_size: int = 40,
    passes: int = 2,
    seed: int = 0,
    lm: Model | None = None,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Keep one row per cluster of duplicates.

    The kept row is the earliest in the cluster. It gains ``n_duplicates`` (how many rows
    it stands for, including itself) and ``duplicate_ids`` (the index labels of the others).

    Args:
        signature: the columns to compare, e.g. ``"product, review"``.
        criterion: what counts as a duplicate, e.g. ``"Same root cause"``.
        passes: rounds of batching; later rounds reshuffle survivors.
    """
    criterion = criterion or "They describe the same thing."
    sig = to_signature(signature, outputs=False)
    check(sig, df.columns)
    lm = get_lm(lm)
    docs = rendered(sig, df)
    uf = _UnionFind(len(docs))
    usages = []
    rng = random.Random(seed)

    for p in range(passes):
        reps = sorted({uf.find(i) for i in range(len(docs))})
        if len(reps) < 2:
            break
        if p > 0:
            rng.shuffle(reps)
        batches = [reps[i : i + batch_size] for i in range(0, len(reps), batch_size)]
        batches = [b for b in batches if len(b) > 1]
        preds = lm.predict(
            CLUSTER,
            [{"criterion": criterion, "records": numbered([docs[pos] for pos in b])} for b in batches],
            desc=f"{save_as or 'lm_dedup'} pass {p}",
        )
        usages += [pr.usage for pr in preds]
        for b, pr in zip(batches, preds):
            for cluster in pr.outputs["clusters"] or []:
                members = [b[j - 1] for j in cluster if isinstance(j, int) and 1 <= j <= len(b)]
                for m in members[1:]:
                    uf.union(members[0], m)
        if len(batches) <= 1:
            break  # everything was compared in one call; another pass adds nothing

    clusters: dict[int, list[int]] = {}
    for i in range(len(docs)):
        clusters.setdefault(uf.find(i), []).append(i)
    keep = sorted(clusters)
    result = df.iloc[keep].copy()
    labels = list(df.index)
    result["n_duplicates"] = [len(clusters[r]) for r in keep]
    result["duplicate_ids"] = [[labels[i] for i in clusters[r] if i != r] for r in keep]

    return record(
        result,
        op="dedup",
        name=save_as,
        inputs=[df],
        instruction=criterion,
        model=lm.model,
        usage=total_usage(usages),
        params={"signature": describe(sig), "passes": passes, "batch_size": batch_size, "kept": len(keep), "of": len(docs)},
    )
