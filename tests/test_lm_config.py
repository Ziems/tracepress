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
    assert lm.supports_decision_requests and lm.model == "jev-latest"
    assert lm_config.model_label(lm) == ("typesafe", "jev-latest")
    assert lm_config.resolve_lm(lm) is lm  # not a dspy.BaseLM, but still an LM


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
