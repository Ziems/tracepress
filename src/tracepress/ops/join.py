"""``lm_join``: not built yet.

TODO: a semantic join (rows of two frames paired when a condition holds). The naive form
is one call per pair; a usable one needs blocking or batched comparisons first.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def lm_join(left: pd.DataFrame, right: pd.DataFrame, signature: Any, condition: str, **kwargs: Any) -> pd.DataFrame:
    raise NotImplementedError("lm_join is planned but not implemented yet.")
