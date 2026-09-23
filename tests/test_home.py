from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_miner.cli import main
from trace_miner.home import (
    cache_dir,
    configure_lm_cache,
    display_home,
    ensure_home,
    format_usage,
    home_dir,
    load_stats,
    record_command,
    record_run,
    runs_path,
    stats_path,
)
from trace_miner.model import RunStats


def test_home_dir_defaults_to_dot_trace_miner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACE_MINER_HOME", raising=False)
    assert home_dir() == Path.home() / ".trace-miner"
    assert display_home() == "~/.trace-miner"


def test_home_dir_respects_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "custom-home"
    monkeypatch.setenv("TRACE_MINER_HOME", str(override))
    assert home_dir() == override.resolve()
    assert display_home() == str(override.resolve())


def test_ensure_home_creates_cache_directory(isolate_trace_miner_home: Path) -> None:
    assert not isolate_trace_miner_home.exists()
    root = ensure_home()
    assert root == isolate_trace_miner_home
    assert root.is_dir()
    assert (root / "cache").is_dir()
    assert cache_dir() == root / "cache"


def test_record_run_accumulates_usage(isolate_trace_miner_home: Path) -> None:
    record_run(
        "run",
        RunStats(
            records_read=3,
            emitted=4,
            retired=2,
            failed=1,
            elapsed_s=1.25,
            lm_calls=2,
            lm_prompt_tokens=10,
            lm_completion_tokens=5,
            lm_cost_usd=0.01,
        ),
        extra={"config": "/tmp/pipeline.yaml"},
    )
    record_run(
        "serve",
        RunStats(records_read=1, retired=1, elapsed_s=0.5, lm_calls=1, lm_prompt_tokens=4, lm_cost_usd=0.002),
    )

    stats = load_stats()
    assert stats["runs"] == 2
    assert stats["commands"] == {"run": 1, "serve": 1}
    assert stats["records_read"] == 4
    assert stats["emitted"] == 4
    assert stats["retired"] == 3
    assert stats["failed"] == 1
    assert stats["lm_calls"] == 3
    assert stats["lm_prompt_tokens"] == 14
    assert stats["lm_completion_tokens"] == 5
    assert stats["lm_cost_usd"] == pytest.approx(0.012)
    assert stats["elapsed_s"] == pytest.approx(1.75)
    assert stats["created_at"]
    assert stats["updated_at"]

    events = [json.loads(line) for line in runs_path().read_text(encoding="utf-8").splitlines()]
    assert [event["command"] for event in events] == ["run", "serve"]
    assert events[0]["config"] == "/tmp/pipeline.yaml"
    assert events[0]["lm_calls"] == 2


def test_record_command_counts_without_a_run(isolate_trace_miner_home: Path) -> None:
    record_command("init", extra={"directory": "/tmp/demo"})
    record_command("serve")

    stats = load_stats()
    assert stats["runs"] == 0
    assert stats["commands"] == {"init": 1, "serve": 1}
    events = [json.loads(line) for line in runs_path().read_text(encoding="utf-8").splitlines()]
    assert events[0]["directory"] == "/tmp/demo"


def test_format_usage_empty_and_with_totals(isolate_trace_miner_home: Path) -> None:
    assert format_usage() == f"No usage recorded yet. Local data lives in {isolate_trace_miner_home}"
    record_run(
        "run",
        RunStats(elapsed_s=1.0, lm_calls=3, lm_prompt_tokens=4000, lm_completion_tokens=210, lm_cost_usd=0.0021),
    )
    text = format_usage()
    assert str(isolate_trace_miner_home) in text
    assert "1 runs  3 LM calls  4,210 tokens  $0.0021" in text
    assert "run 1" in text


def test_configure_lm_cache_points_dspy_at_home(isolate_trace_miner_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_configure_cache(**kwargs: object) -> None:
        seen.update(kwargs)

    monkeypatch.setattr("dspy.configure_cache", fake_configure_cache)
    path = configure_lm_cache()
    assert path == isolate_trace_miner_home / "cache"
    assert path.is_dir()
    assert seen["disk_cache_dir"] == str(path)


def test_cli_creates_home_and_records_batch_usage(
    tmp_path: Path,
    isolate_trace_miner_home: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "stages.py").write_text("def keep(record):\n    return record\n", encoding="utf-8")
    (tmp_path / "input.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
    (tmp_path / "pipeline.yaml").write_text(
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output/findings.jsonl
    pipeline:
      - map: keep
""",
        encoding="utf-8",
    )

    assert not isolate_trace_miner_home.exists()
    assert main(["run", "pipeline.yaml"]) == 0
    capsys.readouterr()
    assert isolate_trace_miner_home.is_dir()
    assert (isolate_trace_miner_home / "cache").is_dir()
    stats = json.loads(stats_path().read_text(encoding="utf-8"))
    assert stats["commands"]["run"] == 1
    assert stats["runs"] == 1
    assert stats["records_read"] == 2
    assert stats["retired"] == 2

    assert main(["stats"]) == 0
    out = capsys.readouterr().out
    assert str(isolate_trace_miner_home) in out
    assert "1 runs" in out
    assert "run 1" in out
