"""``lm_map``: one model call per row, writing the signature's outputs as columns.

    df.lm_map("review -> gist", "Summarize in five words or fewer")
    df.lm_map("product, review -> has_defect: bool, part", "Does it report a defect?")
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import dspy
import pandas as pd

from ..model import Model, total_usage
from ..run import record
from ..settings import get_lm
from ..signature import check, describe, has_instruction, inputs, output_schema, to_signature
from ._common import expand, rows


def lm_map(
    df: pd.DataFrame,
    signature: Any,
    instruction: str | None = None,
    *,
    examples: Sequence[Mapping[str, Any]] | None = None,
    lm: Model | None = None,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Run one call per row and add the signature's outputs as columns.

    Args:
        signature: ``"inputs -> outputs"``, e.g. ``"review -> sentiment: Literal['pos', 'neg']"``,
            or a ``tracepress.Signature`` class.
        instruction: what to do, in plain words. Overrides a class signature's docstring.
        examples: few-shot rows, each a mapping with the input and output fields.
        lm: the model; defaults to ``tracepress.configure(lm=...)``.
        save_as: the artifact name when this runs inside a pipeline.
    """
    sig = to_signature(signature, instruction, outputs=True)
    check(sig, df.columns)
    lm = get_lm(lm)
    demos = [dspy.Example(**ex) for ex in examples] if examples else None
    preds = lm.predict(sig, [inputs(sig, r) for r in rows(df)], examples=demos, desc=save_as or "lm_map")
    result = expand(df, [p.outputs for p in preds], list(sig.output_fields))

    return record(
        result,
        op="map",
        name=save_as,
        inputs=[df],
        instruction=sig.instructions if has_instruction(sig) else None,
        model=lm.model,
        usage=total_usage([p.usage for p in preds]),
        params={"signature": describe(sig), "json_schema": output_schema(sig)},
    )
