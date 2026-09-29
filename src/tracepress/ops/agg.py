"""``lm_agg``: reduce many rows to one answer (per group), however many rows there are.

    issues.lm_agg("part, review -> top_defect", "Most common defect, with a count", by="product")

Rows are packed into batches under a character budget. If a group fits in one batch, one
call answers it. Otherwise each batch gets a partial answer, and the partial answers are
packed and merged again, level by level, until one call can see everything left; that
last call fills the signature's outputs. All groups advance a level together, so every
level is one concurrent batch of calls.

The output is an ordinary DataFrame (one row per group), so aggregations chain.
"""

from __future__ import annotations

from typing import Any, Sequence

import dspy
import pandas as pd

from .. import prompt
from ..model import Model, total_usage
from ..run import record
from ..settings import get_lm
from ..signature import check, describe, has_instruction, output_schema, to_signature
from ._common import many_rows_signature, numbered, rendered

DEFAULT_MAX_CHARS = 60_000

RULES = (
    "Ground every claim in the records and keep counts and identifiers exact. "
    "The [n] labels only number the records within this message: never cite them; "
    "refer to records by their own fields (ids, names) instead. "
    "The scope says whether the records are the whole input, one part of it, or partial answers to merge."
)


def pack(docs: Sequence[str], max_chars: int, min_per_batch: int = 1) -> list[list[str]]:
    """Greedily group ``docs`` into batches of at most ``max_chars`` characters.

    A single doc longer than the budget is truncated. ``min_per_batch`` lets a batch run
    over budget so that a merge level always makes progress.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for doc in docs:
        doc = prompt.as_text(doc, max_chars)
        if current and size + len(doc) > max_chars and len(current) >= min_per_batch:
            batches.append(current)
            current, size = [], 0
        current.append(doc)
        size += len(doc)
    if current:
        if batches and len(current) < min_per_batch:
            batches[-1].extend(current)
        else:
            batches.append(current)
    return batches


def _group_label(by: list[str], key: tuple) -> str:
    if not by:
        return ""
    return " for the group where " + ", ".join(f"{c} = {prompt.as_text(v)}" for c, v in zip(by, key))


def _scope(n: int, *, merging: bool, part: int, parts: int, label: str) -> str:
    if merging:
        what = f"{n} partial answers, each covering a different subset of the records{label}"
    elif parts == 1:
        what = f"all {n} records{label}"
    else:
        what = f"{n} records{label}, part {part} of {parts} of the input"
    if parts == 1:
        how = "merge duplicates and add up counts, as if you had read every record" if merging else "answer from these records"
        return f"These are {what}. This is the final answer: {how}."
    return (
        f"These are {what}. Write a partial answer for this part only. It will be merged with the answers for the "
        "other parts, so keep every count, identifier, and example the final answer might need."
    )


def lm_agg(
    df: pd.DataFrame,
    signature: Any,
    instruction: str | None = None,
    *,
    by: str | Sequence[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    lm: Model | None = None,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Aggregate rows into one answer, or one per group.

    Args:
        signature: ``"inputs -> outputs"``: the columns each record shows, and the columns
            of the answer, e.g. ``"reason -> common_modes"``.
        instruction: what to produce.
        by: column(s) to group on. The result has one row per group: the group columns,
            the outputs, and ``n_rows``.
        max_chars: the per-call budget of record text; more rows than fit are reduced
            hierarchically.
    """
    sig = to_signature(signature, instruction, outputs=True)
    if not has_instruction(sig):
        raise ValueError("lm_agg needs an instruction: what should the answer say?")
    check(sig, df.columns)
    lm = get_lm(lm)
    by = [by] if isinstance(by, str) else list(by or [])
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise KeyError(f"`by` columns {missing} are not in {list(df.columns)}")

    instructions = f"{sig.instructions}\n\n{RULES}"
    scope = "which records these are, and what kind of answer to write"
    final_sig = many_rows_signature(
        {name: (f.annotation, f) for name, f in sig.output_fields.items()}, instructions, scope=scope
    )
    partial_sig = many_rows_signature(
        {"partial_answer": (str, dspy.OutputField(desc="the answer for this part"))}, instructions, scope=scope
    )

    grouped = list(df.groupby(by, sort=True, dropna=False)) if by else [((), df)]
    states: list[dict[str, Any]] = [
        {"key": key if isinstance(key, tuple) else (key,), "docs": rendered(sig, sub), "n_rows": len(sub), "result": None}
        for key, sub in grouped
    ]
    usages = []
    level = 0
    pending = [s for s in states if s["docs"]]
    for s in states:
        if not s["docs"]:
            s["result"] = {name: None for name in sig.output_fields}

    while pending:
        finals: list[tuple[dict[str, Any], dict[str, str]]] = []
        partials: list[tuple[dict[str, Any], dict[str, str]]] = []
        for s in pending:
            merging = level > 0
            batches = pack(s["docs"], max_chars, min_per_batch=2 if merging else 1)
            label = _group_label(by, s["key"])
            for i, batch in enumerate(batches, 1):
                item = {
                    "scope": _scope(len(batch), merging=merging, part=i, parts=len(batches), label=label),
                    "records": numbered(batch, "Partial answer " if merging else ""),
                }
                (finals if len(batches) == 1 else partials).append((s, item))
            s["docs"] = []

        desc = f"{save_as or 'lm_agg'} level {level}"
        done = lm.predict(final_sig, [item for _, item in finals], desc=desc)
        partial = lm.predict(partial_sig, [item for _, item in partials], desc=desc)
        usages += [p.usage for p in done] + [p.usage for p in partial]
        for (s, _), p in zip(finals, done):
            s["result"] = p.outputs
        for (s, _), p in zip(partials, partial):
            s["docs"].append(str(p.outputs["partial_answer"]).strip())
        pending = [s for s in pending if s["result"] is None]
        level += 1

    result = pd.DataFrame(
        [{**dict(zip(by, s["key"])), **s["result"], "n_rows": s["n_rows"]} for s in states],
        columns=[*by, *sig.output_fields, "n_rows"],
    )

    return record(
        result,
        op="agg",
        name=save_as,
        inputs=[df],
        instruction=sig.instructions,
        model=lm.model,
        usage=total_usage(usages),
        params={"signature": describe(sig), "by": by, "levels": level, "json_schema": output_schema(sig)},
    )
