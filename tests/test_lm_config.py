import pytest

import tracepress as tp
from tracepress import lm as lm_config


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No configured LM, no keys, no .env, and a fresh LM cache for every test."""
    monkeypatch.chdir(tmp_path)
    for name in ("TRACEPRESS_LM", "OPENROUTER_API_KEY", "PORTKEY_API_KEY", "PORTKEY_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(lm_config, "_env_loaded", False)
    monkeypatch.setattr(lm_config, "_lm_cache", {})
    previous = tp.configure().lm
    tp.configure(lm=None)
    yield
    tp.configure(lm=previous)


def test_no_lm_configured_fails_before_any_call():
    table = tp.from_rows([tp.Row((("row", 0),), {"text": "hi"})]).lm_map("text -> topic")
    with pytest.raises(RuntimeError, match=r"No LM configured.*openrouter:openai/gpt-6-luna"):
        table.collect()


def test_openrouter_prefix_maps_to_litellm(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    assert lm_config.make_lm("openrouter:openai/gpt-6-luna").model == "openrouter/openai/gpt-6-luna"


def test_openrouter_asks_for_cost_only_when_the_price_is_unknown(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    assert "extra_body" not in lm_config.make_lm("openrouter:openai/gpt-6-luna").kwargs  # priced: cache keys unchanged
    unpriced = lm_config.make_lm("openrouter:acme/unreleased-model")
    assert unpriced.kwargs["extra_body"] == {"usage": {"include": True}}


def test_typesafe_prefix_builds_a_decision_client(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    lm = lm_config.make_lm("typesafe:jev-latest")
    assert lm.supports_decision_requests and lm.model == "typesafe:jev-latest"
    assert lm_config.model_label(lm) == ("typesafe", "jev-latest")
    assert lm_config.call_cost(lm.model, 1_000_000, 0) == 0.042


def test_configuring_a_price_keeps_the_built_in_prices(monkeypatch):
    monkeypatch.setattr(tp.configure(), "prices", dict(lm_config.PRICES))  # restored after the test
    tp.configure(prices={"openai/my-model": (0.50, 2.00)})
    assert lm_config.call_cost("openai/my-model", 1_000_000, 0) == 0.50
    assert lm_config.call_cost("openrouter/openai/gpt-6-luna", 1_000_000, 0) == 0.10
    assert lm_config.call_cost("typesafe:jev-latest", 1_000_000, 0) == 0.042


def test_prices_show_what_a_run_charges_and_where_it_comes_from(monkeypatch):
    monkeypatch.setattr(tp.configure(), "prices", dict(lm_config.PRICES))  # restored after the test
    tp.configure(prices={"openai/my-model": (0.50, 2.00)})
    configured = tp.prices().set_index("model")
    assert configured.loc["openrouter/openai/gpt-6-luna"].tolist() == [0.10, 0.50, "built-in"]
    assert configured.loc["openai/my-model"].tolist() == [0.50, 2.00, "configured"]
    looked_up = tp.prices("openai/gpt-4o-mini", "openai/acme-unreleased-model", "ollama/llama3").set_index("model")
    gpt = looked_up.loc["openai/gpt-4o-mini"]
    assert gpt.source == "dspy" and gpt.usd_per_1m_input == lm_config.call_cost("openai/gpt-4o-mini", 1_000_000, 0)
    for unpriced in ("openai/acme-unreleased-model", "ollama/llama3"):  # DSPy lists Ollama as free; runs don't price it
        assert looked_up.loc[unpriced].isna().all() and lm_config.call_cost(unpriced, 100, 10) is None


def test_prices_take_the_model_strings_people_configure_without_api_keys():
    luna = tp.prices("openrouter:openai/gpt-6-luna").iloc[0]
    assert luna.model == "openrouter/openai/gpt-6-luna" and luna.source == "built-in"
    assert (luna.usd_per_1m_input, luna.usd_per_1m_output) == lm_config.PRICES["openrouter/openai/gpt-6-luna"]
    assert luna.usd_per_1m_input == lm_config.call_cost(luna.model, 1_000_000, 0)
    specs = [
        "portkey:openai-prod/gpt-5",
        "portkey:@openai-prod/gpt-5",
        "typesafe:jev-latest",
        "openrouter/openai/gpt-6-luna",
        "anthropic/claude-opus-5",
    ]
    assert tp.prices(*specs).model.tolist() == [
        "openai/@openai-prod/gpt-5",
        "openai/@openai-prod/gpt-5",
        "typesafe:jev-latest",
        "openrouter/openai/gpt-6-luna",
        "anthropic/claude-opus-5",
    ]


def test_model_name_is_the_name_make_lm_gives_its_lm(monkeypatch):
    for key in ("OPENROUTER_API_KEY", "PORTKEY_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.setenv(key, "test")
    for spec in (
        "openrouter:openai/gpt-6-luna",
        "openrouter:acme/unreleased-model",
        "portkey:openai-prod/gpt-5",
        "portkey:@openai-prod/gpt-5",
        "typesafe:jev-latest",
        "ollama_chat/llama3:8b",
    ):
        assert lm_config.make_lm(spec).model == lm_config.model_name(spec)


def test_portkey_prefix_targets_the_gateway_with_a_provider_slug(monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-test")
    for spec in ("portkey:openai-prod/gpt-5", "portkey:@openai-prod/gpt-5"):
        lm = lm_config.make_lm(spec)
        assert lm.model == "openai/@openai-prod/gpt-5"
        assert lm.kwargs["api_base"] == lm_config.PORTKEY_BASE_URL
        assert lm.kwargs["api_key"] == "pk-test"
        assert lm.kwargs["extra_headers"] == {"x-portkey-api-key": "pk-test"}
    monkeypatch.setenv("PORTKEY_BASE_URL", "http://gateway.internal/v1")
    assert lm_config.make_lm("portkey:openai-prod/gpt-5").kwargs["api_base"] == "http://gateway.internal/v1"
    with pytest.raises(ValueError, match="provider slug"):
        lm_config.make_lm("portkey:gpt-5")


def test_missing_key_names_the_variable():
    with pytest.raises(RuntimeError, match="PORTKEY_API_KEY"):
        lm_config.make_lm("portkey:openai-prod/gpt-5")


def test_keys_and_model_come_from_dotenv(tmp_path):
    (tmp_path / ".env").write_text("TRACEPRESS_LM=openrouter:openai/gpt-6-luna\nOPENROUTER_API_KEY=or-test\n")
    assert lm_config.resolve_lm(None).model == "openrouter/openai/gpt-6-luna"


def test_plain_litellm_strings_pass_through():
    assert lm_config.make_lm("anthropic/claude-opus-5").model == "anthropic/claude-opus-5"
    # A colon after a slash is part of the model name (an Ollama tag), not a provider.
    assert lm_config.make_lm("ollama_chat/llama3:8b").model == "ollama_chat/llama3:8b"


def test_unknown_provider_lists_known_ones():
    with pytest.raises(ValueError, match="openrouter, portkey"):
        lm_config.make_lm("portky:openai/gpt-5")
