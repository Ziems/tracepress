from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import dspy
import pytest

from trace_miner.config import load_config
from trace_miner.errors import ConfigError, PipelineExecutionError
from trace_miner.jev import (
    DEFAULT_JEV_MODEL,
    JEV_INPUT_USD_PER_MILLION,
    OPENROUTER_JEV_BASE_URL,
    JevPredict,
    JevSystemOneResponse,
)
from trace_miner.runtime import run_config


def write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def read_jsonl(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_project(tmp_path: Path, stages: str, config: str, inputs: list[object] | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "stages.py").write_text(stages, encoding="utf-8")
    if inputs is not None:
        write_jsonl(tmp_path / "input.jsonl", inputs)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(config, encoding="utf-8")
    return config_path


class Triage(dspy.Signature):
    """Triage one note."""

    text: str = dspy.InputField()
    failed: bool = dspy.OutputField(desc="Whether it failed.")
    domain: Literal["systems", "other"] = dspy.OutputField(desc="Task domain.")
    severity: float = dspy.OutputField()


class FakeJev:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def system_one(self, state, questions, *, model=None, response_model=None):
        self.calls.append({"state": state, "questions": questions, "model": model, "response_model": response_model})
        return self.response


def _response(*, noul: float = 0.5, choice: str = "systems", score: float = 1.4, cost: float | None = 0.01, tokens: int = 1000):
    usage = {"input_tokens": tokens, "output_tokens": 3}
    if cost is not None:
        usage["cost"] = cost
    return SimpleNamespace(
        nouls={"failed": SimpleNamespace(noul=noul)},
        choices={"domain": SimpleNamespace(choice=choice)},
        scores={"severity": SimpleNamespace(score=score)},
        usage=SimpleNamespace(**usage),
    )


def test_jev_bool_uses_half_threshold() -> None:
    low = FakeJev(_response(noul=0.49))
    high = FakeJev(_response(noul=0.5))
    below = JevPredict(Triage, score_fields={"severity": ["low", "mid", "high"]}, client=low)
    above = JevPredict(Triage, score_fields={"severity": ["low", "mid", "high"]}, client=high)

    assert below(text="note").failed is False
    assert above(text="note").failed is True


def test_jev_literal_choice_and_shared_request() -> None:
    client = FakeJev(_response(choice="other"))
    module = JevPredict(Triage, score_fields={"severity": ["low", "mid", "high"]}, client=client)

    prediction = module(text="note")

    assert prediction.domain == "other"
    assert len(client.calls) == 1
    questions = client.calls[0]["questions"]
    assert set(questions) == {"failed", "domain", "severity"}
    assert questions["domain"].criteria == {"systems": "Task domain.", "other": "Task domain."}
    assert client.calls[0]["state"]["inputs"] == {"text": "note"}


def test_jev_score_rubric_returns_fractional_score() -> None:
    client = FakeJev(_response(score=1.6))
    module = JevPredict(Triage, score_fields={"severity": ["low", "mid", "high"]}, client=client)

    assert module(text="note").severity == 1.6
    assert client.calls[0]["questions"]["severity"].criteria == ["low", "mid", "high"]


def test_jev_reported_cost_wins_over_token_price() -> None:
    client = FakeJev(_response(cost=0.2, tokens=1_000_000))
    module = JevPredict(Triage, score_fields={"severity": ["low", "high"]}, client=client)

    assert module(text="note")._jev_usage.cost_usd == 0.2


def test_jev_prices_input_tokens_when_cost_is_missing() -> None:
    client = FakeJev(_response(cost=None, tokens=1_000_000))
    module = JevPredict(Triage, score_fields={"severity": ["low", "high"]}, client=client)

    usage = module(text="note")._jev_usage
    assert usage.prompt_tokens == 1_000_000
    assert usage.cost_usd == pytest.approx(JEV_INPUT_USD_PER_MILLION)


def test_jev_rejects_a_string_output() -> None:
    class Summary(dspy.Signature):
        text: str = dspy.InputField()
        summary: str = dspy.OutputField()

    with pytest.raises(ValueError, match="summary"):
        JevPredict(Summary)


def test_jev_client_uses_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingClient:
        kwargs: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            type(self).kwargs = kwargs

        def __enter__(self) -> RecordingClient:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def system_one(self, state, questions, *, model=None, response_model=None):
            self.model = model
            self.response_model = response_model
            return _response()

    monkeypatch.setattr("trace_miner.jev.TypeSafeClient", RecordingClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class Gate(dspy.Signature):
        text: str = dspy.InputField()
        failed: bool = dspy.OutputField()

    JevPredict(Gate)(text="note")

    assert RecordingClient.kwargs["api_key"] == "test-key"
    assert RecordingClient.kwargs["base_url"] == OPENROUTER_JEV_BASE_URL
    assert RecordingClient.kwargs["model"] == DEFAULT_JEV_MODEL
    assert RecordingClient.kwargs["timeout"] == 60.0
    assert RecordingClient.kwargs["retry"].max_retries == 0
    assert DEFAULT_JEV_MODEL == "jev-1.13"


def test_jev_response_keeps_openrouter_cost() -> None:
    parsed = JevSystemOneResponse.model_validate(
        {
            "model": "jev-1.13",
            "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.003},
            "answers": {"failed": {"type": "noul", "noul": 0.8}},
        }
    )
    assert parsed.usage.cost == 0.003


def test_jev_stage_does_not_require_an_lm_provider(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
from typing import Literal

import dspy

from trace_miner.jev import JevPredict


class Gate(dspy.Signature):
    text: str = dspy.InputField()
    domain: Literal["systems", "other"] = dspy.OutputField()
    failed: bool = dspy.OutputField()


gate = JevPredict(Gate)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: gate
""",
        [{"text": "note"}],
    )

    stage = load_config(config).pipelines[0].steps[0].stage
    assert stage is not None
    assert stage.uses_jev is True
    assert stage.kind == "lm"
    assert stage.lm_provider is None
    assert stage.lm_model is None


def test_jev_stage_rejects_lm_options_and_expand(tmp_path: Path) -> None:
    stages = """
from typing import Literal

import dspy

from trace_miner.jev import JevPredict


class Gate(dspy.Signature):
    text: str = dspy.InputField()
    failed: bool = dspy.OutputField()


gate = JevPredict(Gate)
"""
    with_lm = write_project(
        tmp_path / "with-lm",
        stages,
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map:
          stage: gate
          lm:
            model: jev-1.13
""",
        [{"text": "note"}],
    )
    with pytest.raises(ConfigError, match="cannot declare `lm` options"):
        load_config(with_lm)

    expand = write_project(
        tmp_path / "expand",
        stages,
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: gate
          reduce: gate
""",
        [{"text": "note"}],
    )
    with pytest.raises(ConfigError, match="cannot be used as `expand`"):
        load_config(expand)


def test_jev_map_merges_outputs_without_constructing_dspy_lm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_project(
        tmp_path,
        """
from typing import Literal

import dspy

from trace_miner.jev import JevPredict


class Gate(dspy.Signature):
    text: str = dspy.InputField()
    domain: Literal["systems", "other"] = dspy.OutputField()
    failed: bool = dspy.OutputField()


gate = JevPredict(Gate)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: gate
""",
        [{"id": "a", "text": "note", "untouched": True}],
    )

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> RecordingClient:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def system_one(self, state, questions, *, model=None, response_model=None):
            self.model = model
            return SimpleNamespace(
                nouls={"failed": SimpleNamespace(noul=0.91)},
                choices={"domain": SimpleNamespace(choice="systems")},
                scores={},
                usage=SimpleNamespace(input_tokens=500, output_tokens=2, cost=0.004),
            )

    def reject_lm(*args: object, **kwargs: object) -> None:
        raise AssertionError("dspy.LM should not be constructed")

    monkeypatch.setattr("trace_miner.jev.TypeSafeClient", RecordingClient)
    monkeypatch.setattr(dspy, "LM", reject_lm)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"domain": "systems", "failed": True, "id": "a", "text": "note", "untouched": True}
    ]
    assert result.stats.lm_calls == 1
    assert result.stats.lm_prompt_tokens == 500
    assert result.stats.lm_cost_usd == 0.004


def test_jev_missing_openrouter_key_fails_at_run_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy

from trace_miner.jev import JevPredict


class Gate(dspy.Signature):
    text: str = dspy.InputField()
    failed: bool = dspy.OutputField()


gate = JevPredict(Gate)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: gate
""",
        [{"text": "note"}],
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(PipelineExecutionError, match="OPENROUTER_API_KEY is required for Jev stages"):
        run_config(config)
    assert not (tmp_path / "output.jsonl").exists()
