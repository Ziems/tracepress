import json

import dspy
import pytest

import tracepress as tp
from tracepress import lm as lm_config
from tracepress import telemetry
from tracepress.lm import RunStats
from tracepress.pipelines import harbor as pipes

from fakes import FuncLM

SECRET = "zz_secret_zz"


@pytest.fixture
def sent(monkeypatch, tmp_path):
    """Telemetry switched on, with sends captured instead of hitting the network."""
    events = []
    tp.configure(telemetry=True)
    monkeypatch.setattr(telemetry, "POSTHOG_API_KEY", "phc_test")
    monkeypatch.setattr(telemetry, "_automated", lambda: False)
    monkeypatch.setattr(telemetry, "_send", events.append)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("TRACEPRESS_TELEMETRY", "DO_NOT_TRACK"):
        monkeypatch.delenv(name, raising=False)
    yield events
    for thread in telemetry._pending:
        thread.join()


def _wait():
    for thread in telemetry._pending:
        thread.join()


def secret_table():
    rows = [tp.Row(((f"{SECRET}_task", f"{SECRET}_{i}"),), {f"{SECRET}_col": f"{SECRET} value {i}"}) for i in range(3)]
    return tp.from_rows(rows).lm_map(
        f"{SECRET}_col -> {SECRET}_out", instructions=f"Classify {SECRET}.",
        lm=FuncLM(lambda f: {f"{SECRET}_out": "x"}),
    )


def test_one_event_per_run_with_counts(sent, harbor_dir):
    def fn(fields):
        if "tool_calls" in fields:
            return {"digest": "d"}
        return {"matched": "False", "rationale": "r"}

    tp.pipelines.harbor.trajectory_taxonomy(tp.read.harbor(harbor_dir), lm=FuncLM(fn)).collect()
    _wait()
    (event,) = sent
    assert event["event"] == "run_completed" and event["api_key"] == "phc_test"
    props = event["properties"]
    # alpha__2 is the only failed trial with a trajectory: 3 digests, then 9 judges.
    assert props["calls"] == 12 and props["ok"] and props["error_type"] is None
    assert props["llm_calls_by_op"] == {"lm_map": 12}
    assert props["ops"] == {"filter": 1, "map": 3, "lm_map": 2, "reduce": 1, "explode": 1}
    assert props["sources"] == ["harbor"] and props["pipelines"] == ["harbor.trajectory_taxonomy"]
    assert props["models"][0]["gateway"] == "custom" and props["primary_model"] == "custom:custom"
    assert props["$process_person_profile"] is False and props["$geoip_disable"] is True


def test_nothing_private_is_sent(sent, monkeypatch):
    with pytest.raises(RuntimeError):
        tp.from_rows([tp.Row((("row", 0),), {"x": SECRET})]).map(
            lambda r: (_ for _ in ()).throw(RuntimeError(SECRET))).collect()
    secret_table().collect()
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-test")
    stats = RunStats()
    stats.record(lm_config.make_lm(f"portkey:{SECRET}/gpt-5"), "lm_map", 1.0, 10, 5, ok=True, cached=False)
    stats.record(lm_config.make_lm(f"openai/ft:gpt-4o-mini:{SECRET}::abc"), "lm_map", 1.0, 10, 5, ok=True, cached=False)
    telemetry.report(stats, [])
    _wait()
    assert len(sent) == 3
    assert SECRET not in json.dumps(sent)
    assert sent[0]["properties"]["error_type"] == "RuntimeError"
    models = {(m["gateway"], m["model"]) for m in sent[2]["properties"]["models"]}
    assert models == {("portkey", "gpt-5"), ("litellm", "openai/ft:gpt-4o-mini")}


def test_user_exception_types_are_not_named(sent):
    class AcmeInternalError(Exception):
        pass

    with pytest.raises(AcmeInternalError):
        tp.from_rows([tp.Row((("row", 0),), {})]).map(lambda r: (_ for _ in ()).throw(AcmeInternalError())).collect()
    _wait()
    assert sent[0]["properties"]["error_type"] == "other" and sent[0]["properties"]["ok"] is False


def test_cache_and_cost_accounting():
    lm = dspy.LM("openrouter/openai/gpt-6-luna")
    stats = RunStats()
    stats.record(lm, "lm_map", 1.0, 1_000_000, 1_000_000, ok=True, cached=False)
    stats.record(lm, "lm_map", 0.0, 0, 0, ok=True, cached=True)
    assert stats.cached_calls == 1 and stats.cost_usd == pytest.approx(0.60)
    unknown = dspy.LM("openai/acme-unreleased-model")
    stats.record(unknown, "lm_reduce", 1.0, 100, 10, ok=True, cached=False)
    assert stats.unpriced_calls == 1 and stats.by_model["openai/acme-unreleased-model"].cost_usd is None
    assert stats.by_op == {"lm_map": 2, "lm_reduce": 1}
    assert "cost=$0.600000 (+1 unpriced calls)" in repr(stats)


@pytest.mark.parametrize("env", [{"TRACEPRESS_TELEMETRY": "0"}, {"TRACEPRESS_TELEMETRY": "off"}, {"DO_NOT_TRACK": "1"}])
def test_opt_out_env(sent, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    secret_table().collect()
    _wait()
    assert sent == []


def test_opt_out_configure_and_automation(sent, monkeypatch):
    tp.configure(telemetry=False)
    secret_table().collect()
    tp.configure(telemetry=True)
    monkeypatch.setattr(telemetry, "_automated", lambda: True)
    secret_table().collect()
    monkeypatch.setattr(telemetry, "_automated", lambda: False)
    monkeypatch.setattr(telemetry, "POSTHOG_API_KEY", "")
    secret_table().collect()
    _wait()
    assert sent == []


def test_ci_and_pytest_are_detected(monkeypatch):
    assert telemetry._automated()  # pytest sets PYTEST_CURRENT_TEST
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    for name in ("CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)
    assert not telemetry._automated()
    monkeypatch.setenv("CI", "true")
    assert telemetry._automated()


def test_debug_prints_and_does_not_send(sent, monkeypatch, capsys):
    monkeypatch.setenv("TRACEPRESS_TELEMETRY", "debug")
    secret_table().collect()
    _wait()
    assert sent == []
    printed = capsys.readouterr().err
    assert "[tracepress telemetry, not sent]" in printed and '"calls": 3' in printed


def test_install_id_is_created_once_with_a_notice(sent, capsys, tmp_path):
    secret_table().collect()
    secret_table().collect()
    _wait()
    assert sent[0]["distinct_id"] == sent[1]["distinct_id"]
    assert capsys.readouterr().err.count("TRACEPRESS_TELEMETRY=0") == 1
    state = json.loads((tmp_path / "config" / "tracepress" / "telemetry.json").read_text())
    assert state["install_id"] == sent[0]["distinct_id"]


def test_send_never_raises(monkeypatch):
    monkeypatch.setenv("TRACEPRESS_TELEMETRY_HOST", "http://127.0.0.1:9")  # nothing listens here
    telemetry._send({"event": "x"})
    monkeypatch.setattr(telemetry, "payload", lambda *a: 1 / 0)
    monkeypatch.setattr(telemetry, "mode", lambda: "on")
    telemetry.report(RunStats(), [])
