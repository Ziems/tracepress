"""The operators, as DataFrame methods.

    df.lm_map("review -> gist", "Summarize in five words or fewer")
    df.lm_filter("review", "Describes a leak or drip")
    df.lm_agg("part, review -> top_defect", "Most common defect, with a count", by="product")
    df.lm_topk("review", "Most likely to discourage a buyer", k=3)
    df.lm_dedup("product, part, review", "Same root cause")

Each takes a DSPy signature (which columns go in, and for map and agg which come out) and
a plain-language instruction, and returns a new DataFrame. Importing this module
registers them as pandas accessors.
"""

from __future__ import annotations

import warnings

import pandas as pd

from ..telemetry import track_op
from .agg import lm_agg
from .dedup import lm_dedup
from .filter import lm_filter
from .join import lm_join
from .map import lm_map
from .topk import lm_topk

lm_map = track_op("map")(lm_map)
lm_filter = track_op("filter")(lm_filter)
lm_agg = track_op("agg")(lm_agg)
lm_topk = track_op("topk")(lm_topk)
lm_dedup = track_op("dedup")(lm_dedup)
lm_join = track_op("join")(lm_join)

OPS = {"map": lm_map, "filter": lm_filter, "agg": lm_agg, "topk": lm_topk, "dedup": lm_dedup}


def _register(fn) -> None:
    class Accessor:
        __doc__ = fn.__doc__

        def __init__(self, df: pd.DataFrame):
            self._df = df

        def __call__(self, *args, **kwargs):
            return fn(self._df, *args, **kwargs)

    Accessor.__call__.__doc__ = fn.__doc__
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # re-registering on module reload is fine
        pd.api.extensions.register_dataframe_accessor(fn.__name__)(Accessor)


for _fn in (lm_map, lm_filter, lm_agg, lm_topk, lm_dedup, lm_join):
    _register(_fn)

__all__ = ["lm_map", "lm_filter", "lm_agg", "lm_topk", "lm_dedup", "lm_join", "OPS"]
