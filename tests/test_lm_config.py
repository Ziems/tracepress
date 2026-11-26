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


# --- The default JSON adapter's response format ----------------------------------------------


@pytest.fixture
def fresh_dspy_cache(tmp_path):
    import dspy

    previous = dspy.cache
    dspy.configure_cache(disk_cache_dir=str(tmp_path / "dspy-cache"))
    yield
    dspy.cache = previous


def _count_schema_builds(monkeypatch) -> list:
    from dspy.adapters import json_adapter

    builds = []
    real = json_adapter._get_structured_outputs_response_format
    monkeypatch.setattr(
        json_adapter,
        "_get_structured_outputs_response_format",
        lambda *args, **kwargs: builds.append(1) or real(*args, **kwargs),
    )
    return builds


def test_the_default_adapter_builds_a_signatures_response_format_once(monkeypatch, fresh_dspy_cache):
    import dspy
    from fakes import structured_lm

    builds = _count_schema_builds(monkeypatch)
    lm = structured_lm({"topic": "files", "risky": False})
    predict = dspy.Predict("command -> topic, risky: bool")
    with dspy.context(lm=lm, adapter=lm_config.Config().adapter):  # the default (tests swap in ChatAdapter)
        answers = [predict(command=f"ls {i}") for i in range(5)]
    assert [(a.topic, a.risky) for a in answers] == [("files", False)] * 5
    assert len(lm.engine_requests) == 5 and len(builds) == 1
    schemas = [r.config.response_format for r in lm.engine_requests]
    assert all(s == schemas[0] for s in schemas)


def test_the_default_adapter_sends_the_request_dspys_own_adapter_sends(fresh_dspy_cache):
    """Same request, same cache key: answers cached before the change are still found."""
    import dspy
    from fakes import structured_lm

    lm = structured_lm({"topic": "files", "risky": False})
    predict = dspy.Predict("command -> topic, risky: bool")
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        predict(command="ls")
    with dspy.context(lm=lm, adapter=lm_config.JSONAdapter()):
        predict(command="ls")
    assert len(lm.engine_requests) == 1  # the second call was a cache hit


def test_calls_with_their_own_options_choose_their_format_every_time(monkeypatch, fresh_dspy_cache):
    import dspy
    from fakes import structured_lm

    builds = _count_schema_builds(monkeypatch)
    predict = dspy.Predict("command -> topic, risky: bool", temperature=0.5)
    with dspy.context(lm=structured_lm({"topic": "files", "risky": False}), adapter=lm_config.JSONAdapter()):
        for i in range(3):
            predict(command=f"ls {i}")
    assert len(builds) == 3


def test_each_call_gets_its_own_copy_of_the_schema():
    import dspy
    from fakes import structured_lm

    adapter, lm = lm_config.JSONAdapter(), structured_lm({})
    signature = dspy.make_signature("command -> topic, risky: bool")
    first, second = {}, {}
    adapter._prepare_response_format(lm, first, signature)
    first["response_format"].model_json_schema()["properties"].clear()  # an in-place edit, as LiteLLM has made
    adapter._prepare_response_format(lm, second, signature)
    assert set(second["response_format"].model_json_schema()["properties"]) == {"topic", "risky"}


def test_each_call_gets_its_own_json_mode_format():
    import dspy
    from fakes import structured_lm

    adapter, lm = lm_config.JSONAdapter(), structured_lm({}, schemas=False)
    signature = dspy.make_signature("command -> topic")
    first, second = {}, {}
    adapter._prepare_response_format(lm, first, signature)
    first["response_format"]["edited"] = True
    adapter._prepare_response_format(lm, second, signature)
    assert second["response_format"] == {"type": "json_object"}


def _signatures():
    """Output shapes, several of them for one field name, so a key that drops a type collides."""
    from typing import Any, Literal

    import dspy
    import pydantic

    class Finding(pydantic.BaseModel):
        title: str
        severity: int

    def x(annotation, **field):
        return dspy.make_signature(
            {"command": dspy.InputField(), "x": dspy.OutputField(**field)}, custom_types={"Any": Any}
        ).with_updated_fields("x", type_=annotation)

    return {
        "text": dspy.make_signature("command -> topic"),
        "bool": dspy.make_signature("command -> topic, risky: bool"),
        "x: int": x(int),
        "x: str": x(str),
        "x: list[str]": x(list[str]),
        "x: Finding": x(Finding),
        "x: Literal a, b": x(Literal["a", "b"]),
        "x: Literal b, a": x(Literal["b", "a"]),
        "x: int | str": x(int | str),
        "x: str | int": x(str | int),
        "x: float = 0.5": x(float, default=0.5),
        "x: float = 0.7": x(float, default=0.7),
        "x: dict[str, Any]": x(dict[str, Any]),  # an open mapping: plain JSON mode
        "x: ToolCalls": x(dspy.ToolCalls),
        "reasoning": dspy.ChainOfThought("command -> topic").predict.signature,
    }


def test_the_default_adapter_chooses_the_format_dspy_chooses():
    """One adapter, as a pipeline shares it: across output types, LMs, settings and repeated calls."""
    import dspy
    from fakes import structured_lm

    ours = lm_config.JSONAdapter()
    lms = {"structured": structured_lm({}), "json mode": structured_lm({}, schemas=False)}

    def chosen(adapter, lm, signature):
        kwargs = {}
        adapter._prepare_response_format(lm, kwargs, signature)
        return dspy.cache.cache_key(kwargs)  # the format as DSPy's cache key sees it

    signatures = _signatures()
    for _ in range(2):  # the second pass answers from the memo
        for native in (True, False):
            ours.use_native_function_calling = native
            stock = dspy.JSONAdapter(use_native_function_calling=native)
            for lm_name, lm in lms.items():
                for name, signature in signatures.items():
                    assert chosen(ours, lm, signature) == chosen(stock, lm, signature), (native, lm_name, name)


def test_the_default_adapter_pickles_and_copies():
    import copy
    import pickle

    import dspy
    from fakes import structured_lm

    adapter = lm_config.JSONAdapter()
    adapter._prepare_response_format(structured_lm({}), {}, dspy.make_signature("command -> topic"))
    for clone in (pickle.loads(pickle.dumps(adapter)), copy.deepcopy(adapter)):
        assert isinstance(clone, lm_config.JSONAdapter) and len(clone._formats) == 0
        kwargs = {}
        clone._prepare_response_format(structured_lm({}), kwargs, dspy.make_signature("command -> topic"))
        assert "response_format" in kwargs  # and it works after
