from __future__ import annotations

from pathlib import Path
from typing import Literal

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

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


def test_litellm_model_string_is_kept():
    class Engine:
        def complete(self, request):
            raise AssertionError("not called")

    lm = tracepress.Model("openai/gpt-5-nano", engine=Engine(), cache=False)
    assert lm.model == "openai/gpt-5-nano"
    assert lm.lm.model == "openai/gpt-5-nano"
    kept = tracepress.Model("bedrock/anthropic.claude-v2:0", engine=Engine(), cache=False)
    assert kept.lm.model == "bedrock/anthropic.claude-v2:0"


def test_colon_is_not_a_substitute_for_the_litellm_slash():
    with pytest.raises(ValueError, match="openai/gpt-5-nano"):
        tracepress.Model("openai:gpt-5-nano", cache=False)


@pytest.mark.parametrize("name", ["portkey:openai/gpt-5-nano", "portkey:@openai/gpt-5-nano"])
def test_portkey_prices_the_litellm_id(monkeypatch, name):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-test")
    monkeypatch.delenv("PORTKEY_BASE_URL", raising=False)
    lm = tracepress.Model(name, cache=False)
    assert lm.model == name
    assert lm.lm.model == "openai/@openai/gpt-5-nano"
    assert lm.lm.kwargs["api_key"] == "pk-test"
    assert lm.lm.kwargs["api_base"] == "https://api.portkey.ai/v1"
    assert lm.price == pytest.approx((0.05, 0.4))


def test_portkey_warns_when_the_litellm_rate_is_missing(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-test")
    with pytest.warns(UserWarning, match="no LiteLLM price"):
        lm = tracepress.Model("portkey:acme/not-a-model", cache=False)
    assert lm.price is None


def test_portkey_price_argument_is_kept(monkeypatch):
    monkeypatch.setenv("PORTKEY_BASE_URL", "https://gateway.example/v1")
    lm = tracepress.Model("portkey:openai/gpt-5-nano", api_key="pk-explicit", price=(1, 5), cache=False)
    assert lm.price == (1, 5)
    assert lm.lm.kwargs["api_key"] == "pk-explicit"
    assert lm.lm.kwargs["api_base"] == "https://gateway.example/v1"


def _price_table(table):
    def info(provider, model, **kwargs):
        return table.get((provider, model), {})

    return info


def test_gateway_price_falls_back_to_the_upstream_id(monkeypatch):
    class Engine:
        def complete(self, request):
            raise AssertionError("not called")

    monkeypatch.setattr(
        "dspy.clients.model_metadata.model_info",
        _price_table({("openai", "gpt-6-luna"): {"input_cost_per_token": 1e-07, "output_cost_per_token": 5e-07}}),
    )
    lm = tracepress.Model("openrouter:openai/gpt-6-luna", engine=Engine(), cache=False)
    assert lm.price == pytest.approx((0.1, 0.5))
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    filled = tracepress.model._usage_of([{"usage": usage, "cost": None}], lm._recorded_price())
    assert filled.cost_usd == pytest.approx(0.6)
    kept = tracepress.model._usage_of([{"usage": usage, "cost": 1.25}], lm._recorded_price())
    assert kept.cost_usd == 1.25
    direct = tracepress.Model("openai/gpt-6-luna", engine=Engine(), cache=False)
    assert direct.price is None and direct._recorded_price() is None


def test_gateway_entry_wins_over_the_upstream_rate(monkeypatch):
    class Engine:
        def complete(self, request):
            raise AssertionError("not called")

    monkeypatch.setattr(
        "dspy.clients.model_metadata.model_info",
        _price_table({
            ("openrouter", "openai/gpt-6-luna"): {"input_cost_per_token": 3e-06, "output_cost_per_token": 9e-06},
            ("openai", "gpt-6-luna"): {"input_cost_per_token": 1e-07, "output_cost_per_token": 5e-07},
        }),
    )
    lm = tracepress.Model("openrouter:openai/gpt-6-luna", engine=Engine(), cache=False)
    assert lm.price == pytest.approx((3, 9))


def test_gateway_warns_when_neither_rate_exists(monkeypatch):
    class Engine:
        def complete(self, request):
            raise AssertionError("not called")

    monkeypatch.setattr("dspy.clients.model_metadata.model_info", lambda *args, **kwargs: {})
    with pytest.warns(UserWarning, match="no LiteLLM price"):
        lm = tracepress.Model("openrouter:openai/gpt-6-luna", engine=Engine(), cache=False)
    assert lm.price is None


def test_model_provider_is_separated_with_a_colon():
    class Engine:
        def complete(self, request):
            raise AssertionError("not called")

    lm = tracepress.Model("openrouter:anthropic/claude-opus-5.5", engine=Engine(), reasoning="off", cache=False, price=(1, 5))
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


@pytest.mark.parametrize(
    "value",
    ["ran `coqc Main.v 2>&1 | head`", "make && make test", "checked n < 4 and a<b", "kept &amp; and &#38;"],
)
def test_bare_ampersand_and_angle_bracket_parse_in_one_call(make_lm, value):
    lm, engine = make_lm(lambda r: f"<out>\n{value}\n</out>")
    pred = lm.predict(dspy.Signature("text -> out", "Copy"), [{"text": "a"}])[0]
    expected = value.replace("&amp;", "&").replace("&#38;", "&")
    assert pred.outputs == {"out": expected}
    assert len(engine.requests) == 1


def test_list_items_keep_their_markup_when_text_is_escaped(make_lm):
    lm, engine = make_lm(lambda r: "<cmds><item>a && b</item><item>x < y</item></cmds>")
    pred = lm.predict(dspy.Signature("text -> cmds: list[str]", "List"), [{"text": "a"}])[0]
    assert pred.outputs == {"cmds": ["a && b", "x < y"]}
    assert len(engine.requests) == 1


def _prose_then(n_bad: int):
    """Answer the first ``n_bad`` requests with untagged prose, then with XML."""
    seen = []

    def fn(request):
        seen.append(request)
        return "Just a sentence, no tags." if len(seen) <= n_bad else {"out": "ok"}

    return fn


def test_unparseable_response_is_retried(make_lm):
    # XML and DSPy's JSON fallback both fail on attempt 0; attempt 1 parses.
    lm, engine = make_lm(_prose_then(2))
    pred = lm.predict(dspy.Signature("text -> out", "Copy"), [{"text": "a"}])[0]
    assert pred.outputs == {"out": "ok"}
    assert len(engine.requests) == 3
    assert pred.usage.calls == 3 and lm.usage.calls == 3


def test_parse_retry_is_replayed_from_the_cache(make_lm):
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    lm, engine = make_lm(_prose_then(2), cache=True)
    sig = dspy.Signature("text -> out", "Copy")
    lm.predict(sig, [{"text": "parse-retry-cache-unique"}])
    again = lm.predict(sig, [{"text": "parse-retry-cache-unique"}])[0]
    assert again.outputs == {"out": "ok"}
    assert len(engine.requests) == 3
    assert again.usage.calls == 0


@pytest.mark.parametrize("parse_retries", [0, 2])
def test_parse_error_is_raised_after_the_last_retry(make_lm, parse_retries):
    lm, engine = make_lm(_prose_then(100), parse_retries=parse_retries)
    with pytest.raises(AdapterParseError):
        lm.predict(dspy.Signature("text -> out", "Copy"), [{"text": "a"}])
    assert len(engine.requests) == 2 * (parse_retries + 1)
    assert lm.usage.calls == 2 * (parse_retries + 1)


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
