"""The prediction cache: a call's answer looked up before DSPy renders its prompt."""

import dspy
import pytest
from conftest import _step, write_trial
from fakes import FuncLM, _AsyncEngine, _Engine

import tracepress as tp
from tracepress import lm as lm_config


@pytest.fixture(autouse=True)
def fresh_dspy_cache(tmp_path):
    previous = dspy.cache
    dspy.configure_cache(disk_cache_dir=str(tmp_path / "dspy-cache"))
    yield
    dspy.cache = previous


class CountingLM(dspy.LM):
    requests: FuncLM  # answers the calls that reach it; `requests.calls` counts them


def counting_lm(**outputs) -> CountingLM:
    """A dspy.LM (so DSPy caches it) answering `outputs`; `lm.requests.calls` counts what reached it."""
    owner = FuncLM(lambda fields: outputs)
    lm = CountingLM("openai/func", engine=_Engine(owner), async_engine=_AsyncEngine(owner))
    lm.requests = owner
    return lm


def renders(monkeypatch) -> list:
    """Counts prompt renders by the configured adapter (DSPy wraps `format` on each adapter class)."""
    calls = []
    adapter = type(tp.configure().adapter)
    real = adapter.format
    monkeypatch.setattr(adapter, "format", lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    return calls


@pytest.fixture
def job(tmp_path):
    """Three trials of two steps, every step different, so a run never hits its own calls."""
    trials = tmp_path / "job" / "trials"
    trials.mkdir(parents=True)
    for n in range(3):
        write_trial(
            trials, "alpha", f"alpha__{n}", 1, steps=[_step(i, "agent", f"step {i} of trial {n}") for i in (1, 2)]
        )
    return tmp_path / "job"


def test_a_rerun_is_answered_without_rendering_a_prompt(job, monkeypatch):
    lm = counting_lm(what_happened="ran ls", is_mistake=False)
    table = tp.read.harbor(job).steps.lm_map("step -> what_happened, is_mistake: bool", lm=lm)
    first = table.collect()
    assert lm.requests.calls == 6 and tp.last_run().cached_calls == 0

    rendered = renders(monkeypatch)
    second = table.collect()
    assert [r.values for r in second] == [r.values for r in first]
    assert tp.last_run().cached_calls == tp.last_run().calls == 6
    assert lm.requests.calls == 6 and rendered == []


def test_changed_instructions_miss(job):
    lm = counting_lm(what_happened="ran ls", is_mistake=False)
    src = tp.read.harbor(job)
    sig = "step -> what_happened, is_mistake: bool"
    src.steps.lm_map(sig, lm=lm, instructions="Summarize.").collect()
    src.steps.lm_map(sig, lm=lm, instructions="Summarize in five words.").collect()
    assert tp.last_run().cached_calls == 0 and lm.requests.calls == 12
    src.steps.lm_map(sig, lm=lm, instructions="Summarize.").collect()
    assert tp.last_run().cached_calls == 6 and lm.requests.calls == 12


def test_answers_dspy_cached_before_are_still_hits(job, monkeypatch):
    """With the prediction cache off, calls go through DSPy's cache; turning it on finds them."""
    lm = counting_lm(what_happened="ran ls", is_mistake=False)
    table = tp.read.harbor(job).steps.lm_map("step -> what_happened, is_mistake: bool", lm=lm)
    monkeypatch.setattr(lm_config.config(), "prediction_cache", False)
    table.collect()
    rendered = renders(monkeypatch)
    table.collect()
    assert len(rendered) == 6 and tp.last_run().cached_calls == 6  # DSPy's cache, after rendering

    monkeypatch.setattr(lm_config.config(), "prediction_cache", True)
    rendered.clear()
    table.collect()  # answered by DSPy's cache, and remembered
    table.collect()
    assert tp.last_run().cached_calls == 6 and len(rendered) == 6 and lm.requests.calls == 6


def test_what_the_key_holds():
    lm = counting_lm(topic="t")
    predict = dspy.Predict("summary -> topic")
    key = lambda module=predict, lm=lm, inputs={"summary": "s"}: dspy.cache.cache_key(
        lm_config.prediction_request(module, lm, inputs)
    )
    base = key()

    assert key(inputs={"summary": "other"}) != base
    assert key(module=dspy.Predict("summary -> topic: bool")) != base
    assert key(module=dspy.Predict(dspy.make_signature("summary -> topic", "Be brief."))) != base
    assert key(module=dspy.Predict("summary -> topic", temperature=0.9)) != base
    assert key(module=dspy.ChainOfThought("summary -> topic")) != base
    with_demos = dspy.Predict("summary -> topic")
    with_demos.demos = [dspy.Example(summary="a", topic="b")]
    assert key(module=with_demos) != base
    other = counting_lm(topic="t")
    other.kwargs["temperature"] = 0.9
    assert key(lm=other) != base

    rekeyed = counting_lm(topic="t")
    rekeyed.kwargs["api_key"] = "sk-rotated"  # credentials aren't part of what's asked
    assert key(lm=rekeyed) == base


def test_calls_that_cant_be_cached_go_to_dspy(monkeypatch):
    predict, inputs = dspy.Predict("summary -> topic"), {"summary": "s"}
    assert lm_config.prediction_request(predict, FuncLM(lambda f: {}), inputs) is None  # not a dspy.LM
    uncached = counting_lm(topic="t")
    uncached.cache = False
    assert lm_config.prediction_request(predict, uncached, inputs) is None
    monkeypatch.setattr(lm_config.config(), "prediction_cache", False)
    assert lm_config.prediction_request(predict, counting_lm(topic="t"), inputs) is None


def _key(module, inputs, lm=None):
    return dspy.cache.cache_key(lm_config.prediction_request(module, lm or counting_lm(topic="t"), inputs))


def test_inputs_dspy_renders_differently_key_differently():
    import datetime

    predict, lm = dspy.Predict("summary -> topic"), counting_lm(topic="t")
    alike = [
        float("nan"),
        float("inf"),
        None,
        "None",
        1,
        1.0,
        True,
        [1, 2],
        (1, 2),
        {"a": 1, "b": 2},
        {"b": 2, "a": 1},
        datetime.date(2026, 1, 1),
        "2026-01-01",
    ]
    keys = [_key(predict, {"summary": v}, lm) for v in alike]
    assert len(set(keys)) == len(alike)
    assert _key(predict, {"summary": float("nan")}, lm) == keys[0]  # and the same input keys the same


def test_output_types_are_in_the_key():
    import enum

    small, large = enum.Enum("Kind", ["edit", "test"]), enum.Enum("Kind", ["edit", "test", "explore"])
    sig = lambda kind: dspy.make_signature("summary -> kind").with_updated_fields("kind", type_=kind)
    assert _key(dspy.Predict(sig(small)), {"summary": "s"}) != _key(dspy.Predict(sig(large)), {"summary": "s"})
    with_default = dspy.make_signature({"summary": dspy.InputField(), "topic": dspy.OutputField(default="x")})
    assert _key(dspy.Predict(with_default), {"summary": "s"}) != _key(
        dspy.Predict("summary -> topic"), {"summary": "s"}
    )


@pytest.mark.parametrize(
    "module, sig, outputs, expected",
    [
        (
            dspy.Predict,
            "step -> tags: list[str], kind: Literal['edit', 'test']",
            {"tags": '["a", "b"]', "kind": "edit"},
            {"tags": ["a", "b"], "kind": "edit"},
        ),
        # Without native reasoning, DSPy asks for the reasoning as a `plan`.
        (dspy.ChainOfThought, "step -> topic", {"plan": "because", "topic": "files"}, {"topic": "files"}),
    ],
    ids=["typed", "chain of thought"],
)
def test_typed_answers_are_read_back_from_disk(job, monkeypatch, module, sig, outputs, expected):
    """As in a new process: the in-memory cache is gone and the answer comes from disk."""
    lm = counting_lm(**outputs)
    table = tp.read.harbor(job).steps.lm_map(sig, module=module, lm=lm)
    first = table.collect()
    assert len(first) == 6 and tp.last_run().errors == 0
    dspy.cache.reset_memory_cache()
    rendered = renders(monkeypatch)
    second = table.collect()
    assert [r.values for r in second] == [r.values for r in first]
    assert rendered == [] and tp.last_run().cached_calls == 6 and lm.requests.calls == 6
    assert {k: first[0].values[k] for k in expected} == expected


def test_a_changed_answer_doesnt_change_the_cache():
    import asyncio

    lm, predict = counting_lm(tags='["a"]'), dspy.Predict("step -> tags: list[str]")
    executor = lm_config.Executor()
    first = asyncio.run(executor.call(predict, lm, {"step": "ls"}))
    first.tags.append("edited")
    again = asyncio.run(executor.call(predict, lm, {"step": "ls"}))
    assert again.tags == ["a"] and lm.requests.calls == 1


def test_answers_are_stored_only_if_they_read_back_exactly():
    import pydantic

    class Finding(pydantic.BaseModel):
        title: str

    predict = dspy.Predict(
        dspy.make_signature("step -> tags: list[str], finding").with_updated_fields("finding", type_=Finding)
    )
    types = lm_config._output_types(predict)
    answer = dspy.Prediction(tags=["a"], finding=Finding(title="x"))
    stored = lm_config._stored_outputs(predict, answer)
    assert stored == {"tags": ["plain", ["a"]], "finding": ["typed", {"title": "x"}]}
    assert lm_config._read_outputs(types, stored) == dict(answer.items())
    assert (
        lm_config._stored_outputs(predict, dspy.Prediction(tags=("a",), finding=Finding(title="x"))) is None
    )  # a tuple reads back a list
    assert lm_config._stored_outputs(predict, dspy.Prediction(other=1)) is None  # not one of its outputs


def test_a_failed_lookup_asks_dspy(job, monkeypatch):
    lm = counting_lm(what_happened="ran ls", is_mistake=False)
    table = tp.read.harbor(job).steps.lm_map("step -> what_happened, is_mistake: bool", lm=lm)
    table.collect()

    real = dspy.cache.get

    def busy_for_ours(request, *args, **kwargs):  # DSPy's own lookups still work
        if request.get("_fn_identifier") == "tracepress.prediction":
            raise TimeoutError("the disk cache is busy")
        return real(request, *args, **kwargs)

    monkeypatch.setattr(dspy.cache, "get", busy_for_ours)
    rendered = renders(monkeypatch)
    rows = table.collect()
    assert len(rows) == 6 and tp.last_run().errors == 0
    assert len(rendered) == 6  # every call went to DSPy, which answered from its own cache


def test_a_failed_store_doesnt_fail_the_call(job, monkeypatch):
    real = dspy.cache.put

    def full_for_ours(request, *args, **kwargs):
        if request.get("_fn_identifier") == "tracepress.prediction":
            raise OSError("disk full")
        return real(request, *args, **kwargs)

    monkeypatch.setattr(dspy.cache, "put", full_for_ours)
    lm = counting_lm(what_happened="ran ls", is_mistake=False)
    rows = tp.read.harbor(job).steps.lm_map("step -> what_happened, is_mistake: bool", lm=lm).collect()
    assert len(rows) == 6 and tp.last_run().errors == 0 and lm.requests.calls == 6


def test_an_enum_answer_reads_back_in_a_new_process(monkeypatch):
    import asyncio
    import enum

    class Kind(str, enum.Enum):
        edit = "edit"
        test = "test"

    predict = dspy.Predict(dspy.make_signature("step -> kind").with_updated_fields("kind", type_=Kind))
    lm = counting_lm(kind="edit")
    first = asyncio.run(lm_config.Executor().call(predict, lm, {"step": "ls"}))
    assert first.kind is Kind.edit
    dspy.cache.reset_memory_cache()  # as in a new process: the answer comes from disk
    rendered = renders(monkeypatch)
    again = asyncio.run(lm_config.Executor().call(predict, lm, {"step": "ls"}))
    assert again.kind is Kind.edit and rendered == []  # ours, not DSPy's cache after a render


class _Scaled(dspy.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.predict = dspy.Predict("summary -> topic")

    def forward(self, **kwargs):
        return self.predict(**kwargs)


class _Base(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("summary -> topic")


class _FromBase(_Base):
    pass


def test_a_modules_settings_and_code_are_in_the_key():
    inputs = {"summary": "s"}
    assert _key(_Scaled(1), inputs) == _key(_Scaled(1), inputs)  # DSPy's random stage id isn't keyed
    assert _key(_Scaled(1), inputs) != _key(_Scaled(2), inputs)
    assert lm_config._source_id(_FromBase) != lm_config._source_id(_Base)  # a subclass's key holds its bases' code
    assert lm_config._source_id(dspy.ChainOfThought) == lm_config._source_id(dspy.Predict)  # DSPy's: its version

    unsourced = type("Unsourced", (dspy.Module,), {})()  # made at run time: no source to read
    vars(unsourced)["predict"] = dspy.Predict("summary -> topic")
    assert lm_config.prediction_request(unsourced, counting_lm(topic="t"), inputs) is None


def test_containers_and_lookalikes_key_apart():
    predict, lm = dspy.Predict("summary -> topic"), counting_lm(topic="t")
    alike = [(1,), [1], ["<tuple>", [1]], {}, ["<dict>", []], {1, 2}, frozenset({1, 2}), [1, 2]]
    keys = [_key(predict, {"summary": v}, lm) for v in alike]
    assert len(set(keys)) == len(alike)


def test_headers_are_keyed_without_their_values_in_the_request():
    predict, inputs = dspy.Predict("summary -> topic"), {"summary": "s"}
    one, two = counting_lm(topic="t"), counting_lm(topic="t")
    one.kwargs["extra_headers"] = {"x-portkey-api-key": "pk-secret-1"}
    two.kwargs["extra_headers"] = {"x-portkey-api-key": "pk-secret-2"}
    assert _key(predict, inputs, one) != _key(predict, inputs, two)
    assert "pk-secret-1" not in repr(lm_config.prediction_request(predict, one, inputs))


def test_a_predictors_own_lm_is_in_the_key():
    inputs = {"summary": "s"}
    own, other = dspy.Predict("summary -> topic"), dspy.Predict("summary -> topic")
    own.lm, other.lm = counting_lm(topic="a"), counting_lm(topic="a")
    other.lm.kwargs["temperature"] = 0.9
    assert _key(own, inputs) != _key(other, inputs) != _key(dspy.Predict("summary -> topic"), inputs)
    criteria = dspy.Predict("summary -> topic")
    criteria.fields = {"topic": "must name a file"}
    assert _key(criteria, inputs) != _key(dspy.Predict("summary -> topic"), inputs)
    custom = dspy.Predict("summary -> topic")
    custom.lm = FuncLM(lambda f: {})  # an LM we can't key on
    assert lm_config.prediction_request(custom, counting_lm(topic="t"), inputs) is None


def test_a_swapped_predictor_is_keyed_by_its_new_signature():
    class Held(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("summary -> topic")
            self.steps = [dspy.Predict("summary -> topic")]
            self.by_name = {"judge": dspy.Predict("summary -> topic")}
            self.inner = dspy.ChainOfThought("summary -> topic")

    inputs = {"summary": "s"}
    swaps = [
        lambda m: setattr(m, "predict", dspy.Predict("summary -> topic: bool")),
        lambda m: m.steps.__setitem__(0, dspy.Predict("summary -> topic: bool")),
        lambda m: m.steps.append(dspy.Predict("summary -> other")),
        lambda m: m.by_name.__setitem__("judge", dspy.Predict("summary -> topic: bool")),
        lambda m: setattr(m.inner, "_compiled", True),
    ]  # DSPy stops walking a compiled module
    for swap in swaps:
        module = Held()
        before = _key(module, inputs)
        swap(module)
        assert _key(module, inputs) != before


def test_predictors_are_found_as_dspy_finds_them():
    class Steps(dspy.Module):
        def __init__(self):
            super().__init__()
            self.steps = [dspy.Predict("a -> b"), dspy.ChainOfThought("b -> c")]
            self.by_name = {"judge": dspy.Predict("c -> ok: bool")}
            self.inner = dspy.ChainOfThought("c -> d")
            self.again = self.steps[0]  # the same predictor twice is found once

    def tool(x: str) -> str:
        return x

    class Named(dspy.Module):  # its own predictors under the names DSPy uses for bookkeeping
        def __init__(self):
            super().__init__()
            self.history = dspy.Predict("a -> summary")
            self.demos = [dspy.Predict("a -> b")]
            self.traces = {"judge": dspy.Predict("b -> ok: bool")}

    modules = [
        Named(),
        dspy.Predict("a -> b"),
        dspy.ChainOfThought("a -> b"),
        Steps(),
        dspy.ReAct(dspy.make_signature("a -> b"), tools=[tool]),
        dspy.BestOfN(dspy.Predict("a -> b"), N=2, reward_fn=lambda *a: 1.0, threshold=1.0),
    ]
    for module in modules:
        found = [(name, id(p)) for name, p in lm_config._predictors(module)]
        assert found == [(name, id(p)) for name, p in module.named_predictors()], type(module).__name__


def test_keying_a_call_doesnt_grow_with_the_modules_history():
    import sys

    predict, lm = dspy.Predict("summary -> topic"), counting_lm(topic="t")

    def calls_to_key():
        calls = []
        sys.setprofile(lambda frame, event, arg: calls.append(1) if event == "call" else None)
        try:
            lm_config.prediction_request(predict, lm, {"summary": "s"})
        finally:
            sys.setprofile(None)
        return len(calls)

    lm_config.prediction_request(predict, lm, {"summary": "s"})
    small = calls_to_key()
    predict.history.extend({"call": i} for i in range(5000))  # what thousands of calls leave behind
    assert calls_to_key() <= small + 5
