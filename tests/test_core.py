from __future__ import annotations

from pathlib import Path
from typing import Literal

import dspy
import pytest

import tracepress
from tracepress.signature import check, describe, has_instruction, output_schema, to_signature

from .conftest import field_value


def test_string_signature():
    sig = to_signature("product, review -> has_defect: bool, part, severity: Literal['none', 'bad']", "Find defects")
    assert list(sig.input_fields) == ["product", "review"]
    assert list(sig.output_fields) == ["has_defect", "part", "severity"]
    assert sig.instructions == "Find defects" and has_instruction(sig)
    assert describe(sig) == "product, review -> has_defect: bool, part, severity: Literal['none', 'bad']"
    assert output_schema(sig)["properties"]["severity"]["enum"] == ["none", "bad"]


def test_inputs_only_signature():
    sig = to_signature("product, review", outputs=False)
    assert list(sig.input_fields) == ["product", "review"] and not sig.output_fields
    assert describe(sig) == "product, review"


def test_class_signature():
    class FindIssue(tracepress.Signature):
        """Does the review report a defect?"""

        review: str = tracepress.In()
        part: str = tracepress.Out(desc="the part at fault")
        severity: Literal["low", "high"] = tracepress.Out()

    sig = to_signature(FindIssue)
    assert sig.instructions == "Does the review report a defect?" and has_instruction(sig)
    assert describe(sig) == "review -> part, severity: Literal['low', 'high']"
    assert to_signature(FindIssue, "Override").instructions == "Override"


def test_default_instruction_is_not_counted():
    assert not has_instruction(to_signature("a -> b"))


@pytest.mark.parametrize(
    "spec, kwargs, error",
    [
        ("review", {"outputs": True}, "needs outputs"),
        ("review -> x", {"outputs": False}, "only reads rows"),
    ],
)
def test_signature_errors(spec, kwargs, error):
    with pytest.raises(ValueError, match=error):
        to_signature(spec, **kwargs)


def test_signature_checks_columns():
    with pytest.raises(KeyError, match="missing"):
        check(to_signature("missing -> x"), ["a"])


def test_predict_returns_typed_outputs_and_usage(make_lm):
    lm, engine = make_lm(lambda r: {"shout": field_value(r, "text").upper(), "n": 3})
    sig = dspy.Signature("text -> shout, n: int", "Shout")
    preds = lm.predict(sig, [{"text": "a"}, {"text": "b"}])
    assert [p.outputs for p in preds] == [{"shout": "A", "n": 3}, {"shout": "B", "n": 3}]
    assert preds[0].usage.calls == 1 and preds[0].usage.input_tokens == 10
    assert lm.usage.calls == 2 and lm.usage.output_tokens == 10


def test_model_provider_is_separated_with_a_colon():
    class Engine:
        def complete(self, request):
            raise AssertionError("not called")

    lm = tracepress.Model("openrouter:anthropic/claude-opus-5.5", engine=Engine(), reasoning="off", cache=False)
    assert lm.model == "openrouter:anthropic/claude-opus-5.5"
    assert lm.lm.model == "openrouter/anthropic/claude-opus-5.5"
    assert lm.lm.kwargs["reasoning"] == {"effort": "off"}


def test_response_cache_is_under_tracepress_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEPRESS_HOME", str(tmp_path))
    previous = dspy.cache
    try:
        tracepress.configure()
        assert Path(dspy.cache.disk_cache_dir) == tmp_path / "cache"
        assert (tmp_path / "cache").is_dir()
    finally:
        dspy.cache = previous


def test_cache_hits_are_counted_as_cached(make_lm):
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    lm, engine = make_lm(lambda r: {"out": "x"}, cache=True)
    sig = dspy.Signature("text -> out", "Copy")
    lm.predict(sig, [{"text": "cache-test-unique"}])
    again = lm.predict(sig, [{"text": "cache-test-unique"}])[0]
    assert len(engine.requests) == 1
    assert again.usage.cached_calls == 1 and again.usage.calls == 0


def test_budget_refuses_before_calling(make_lm):
    lm, engine = make_lm(lambda r: {"out": "x"})
    sig = dspy.Signature("text -> out", "Copy")
    with tracepress.budget(max_calls=2):
        lm.predict(sig, [{"text": "a"}, {"text": "b"}])
        with pytest.raises(tracepress.BudgetExceeded):
            lm.predict(sig, [{"text": "c"}])
    assert len(engine.requests) == 2


def test_dollar_budget_uses_prices(make_lm):
    lm, engine = make_lm(lambda r: {"out": "x"}, price=(1000.0, 1000.0), max_tokens=100)
    with tracepress.budget(max_usd=0.05), pytest.raises(tracepress.BudgetExceeded):
        lm.predict(dspy.Signature("text -> out"), [{"text": "a"}] * 10)
    assert engine.requests == []


def test_using_overrides_the_default(make_lm):
    a, _ = make_lm(lambda r: "{}")
    b, _ = make_lm(lambda r: "{}")
    tracepress.configure(lm=a)
    from tracepress.settings import get_lm

    with tracepress.using(lm=b):
        assert get_lm() is b
    assert get_lm() is a
