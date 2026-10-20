"""Calls real models through OpenRouter. Skipped unless OPENROUTER_API_KEY is set (or in ../.env)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import tracepress


def _load_env() -> None:
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() and value and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_env()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="needs OPENROUTER_API_KEY"),
]

OPS_MODEL = os.environ.get("TRACEPRESS_LIVE_MODEL", "openrouter:openai/gpt-6-luna")

REVIEWS = pd.DataFrame(
    {
        "product": ["kettle", "kettle", "kettle", "toaster", "toaster"],
        "review": [
            "Boils fast but the lid broke after a week.",
            "Lid hinge snapped in the first month. Otherwise fine.",
            "Great kettle, quiet and quick.",
            "Burns one side of the bread every time.",
            "Uneven toasting; left slot is much hotter.",
        ],
    }
)


@pytest.fixture(scope="module")
def lm():
    return tracepress.Model(OPS_MODEL, temperature=0, max_tokens=600, reasoning="off", cache=False, progress=False)


def test_map_filter_agg(lm):
    mapped = REVIEWS.lm_map("review -> gist", "The main point in five words or fewer", lm=lm)
    assert mapped["gist"].str.len().gt(0).all()

    issues = REVIEWS.lm_map("review -> has_defect: bool, part", "Does the review report a defect, and in which part?", lm=lm)
    assert issues["has_defect"].tolist() == [True, True, False, True, True]

    kept = REVIEWS.lm_filter("review", "Reports a problem with the lid", lm=lm)
    assert set(kept.index) == {0, 1}

    assert lm.usage.calls > 0 and lm.usage.cost_usd > 0  # DSPy reports cost

    per_product = REVIEWS.lm_agg("review -> complaint", "The most common complaint, in one sentence", by="product", lm=lm)
    assert list(per_product["product"]) == ["kettle", "toaster"]
    assert "lid" in per_product["complaint"].iloc[0].lower()
