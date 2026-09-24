from __future__ import annotations

import json
from pathlib import Path

import pytest

from hello_world_checks import (
    ATRX,
    FREECAD,
    PHOTONIC,
    check_alerts,
    check_categories,
    check_doom_loop,
    check_frustration,
    check_pii,
    check_project,
    check_summaries,
    main,
)

ATRX_SESSION = "atrx-session"
PHOTONIC_SESSION = "photonic-session"
FREECAD_SESSION = "freecad-session"
LOOP_SESSION = "loop-session"


def _write_jsonl(path: Path, rows: list[dict[object, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _summary_rows() -> list[dict[object, object]]:
    return [
        {
            "task_name": ATRX,
            "session_id": ATRX_SESSION,
            "reported_passed": False,
            "failed": True,
            "failure_cause": "HGVS annotations did not match the verifier.",
            "trajectory_summary": "The agent catalogued ATRX variants but failed checks.",
        },
        {
            "task_name": PHOTONIC,
            "session_id": PHOTONIC_SESSION,
            "reported_passed": False,
            "failed": True,
            "failure_cause": "Routing never finished after the user asked it to stop delaying.",
            "trajectory_summary": "Waveguide routing hit an output-token limit.",
        },
        {
            "task_name": FREECAD,
            "session_id": FREECAD_SESSION,
            "reported_passed": True,
            "failed": False,
            "failure_cause": "none",
            "trajectory_summary": "The agent built a parametric spring clip and passed.",
        },
    ]


def _pii_rows() -> list[dict[object, object]]:
    return [
        {
            "task_name": ATRX,
            "has_pii": True,
            "findings": [{"kind": "ssn", "evidence": "123-45-6789"}],
        },
        {"task_name": PHOTONIC, "has_pii": False, "findings": []},
        {
            "task_name": FREECAD,
            "has_pii": True,
            "findings": [{"kind": "phone", "evidence": "(555) 010-1234"}],
        },
    ]


def _category_rows() -> list[dict[object, object]]:
    return [
        {
            "task_name": ATRX,
            "domain": "bioinformatics",
            "outcome": "wrong_output",
            "trajectory_summary": "The agent catalogued ATRX variants but failed checks.",
        },
        {
            "task_name": PHOTONIC,
            "domain": "systems",
            "outcome": "incomplete",
            "trajectory_summary": "Waveguide routing hit an output-token limit.",
        },
        {
            "task_name": FREECAD,
            "domain": "other",
            "outcome": "passed",
            "trajectory_summary": "The agent built a parametric spring clip and passed.",
        },
    ]


def _alert_rows() -> list[dict[object, object]]:
    return [
        {
            "name": "flag_sensitive_step",
            "reason": "ssn",
            "session_id": "5927034d-40de-4882-8ac2-3fe2c7dbfc7d",
        }
    ]


def _doom_loop_rows() -> list[dict[object, object]]:
    return [
        {"name": "flag_doom_loop", "reason": "restarting the same service", "session_id": LOOP_SESSION},
    ]


def _frustration_rows() -> list[dict[object, object]]:
    return [
        {"task_name": ATRX, "has_frustration": False, "findings": []},
        {
            "task_name": PHOTONIC,
            "has_frustration": True,
            "findings": [{"evidence": "This is taking forever"}],
        },
        {"task_name": FREECAD, "has_frustration": False, "findings": []},
    ]


def test_check_summaries_accepts_planted_labels(tmp_path: Path) -> None:
    check_summaries(_write_jsonl(tmp_path / "summaries.jsonl", _summary_rows()))


def test_check_summaries_rejects_wrong_failed_flag(tmp_path: Path) -> None:
    rows = _summary_rows()
    rows[2]["failed"] = True
    with pytest.raises(AssertionError, match="freecad-spring-clip"):
        check_summaries(_write_jsonl(tmp_path / "summaries.jsonl", rows))


def test_check_pii_accepts_planted_spans(tmp_path: Path) -> None:
    check_pii(_write_jsonl(tmp_path / "pii.jsonl", _pii_rows()))


def test_check_pii_identifies_rows_from_trajectory_path(tmp_path: Path) -> None:
    rows = [
        {
            "trajectory_path": "Terminal-Bench/atrx-vep-crispr__76064f07/agent/trajectory.eval.json",
            "has_pii": True,
            "findings": [{"kind": "ssn", "evidence": "123-45-6789"}],
        },
        {
            "trajectory_path": "Terminal-Bench/photonic-waveguide-routing__3890296f/agent/trajectory.eval.json",
            "has_pii": False,
            "findings": [],
        },
        {
            "trajectory_path": "Terminal-Bench/freecad-spring-clip__df396b0f/agent/trajectory.eval.json",
            "has_pii": True,
            "findings": [{"kind": "phone", "evidence": "(555) 010-1234"}],
        },
    ]
    check_pii(_write_jsonl(tmp_path / "pii.jsonl", rows))


def test_check_pii_rejects_missing_ssn(tmp_path: Path) -> None:
    rows = _pii_rows()
    rows[0]["has_pii"] = False
    rows[0]["findings"] = []
    with pytest.raises(AssertionError, match="planted SSN"):
        check_pii(_write_jsonl(tmp_path / "pii.jsonl", rows))


def test_check_frustration_accepts_planted_user_turn(tmp_path: Path) -> None:
    check_frustration(_write_jsonl(tmp_path / "frustration.jsonl", _frustration_rows()))


def test_check_frustration_rejects_quiet_photonic(tmp_path: Path) -> None:
    rows = _frustration_rows()
    rows[1]["has_frustration"] = False
    rows[1]["findings"] = []
    with pytest.raises(AssertionError, match="planted user turn"):
        check_frustration(_write_jsonl(tmp_path / "frustration.jsonl", rows))


def test_check_categories_accepts_planted_labels(tmp_path: Path) -> None:
    check_categories(_write_jsonl(tmp_path / "categories.jsonl", _category_rows()))


def test_check_categories_rejects_passed_failure(tmp_path: Path) -> None:
    rows = _category_rows()
    rows[0]["outcome"] = "passed"
    with pytest.raises(AssertionError, match="atrx-vep-crispr"):
        check_categories(_write_jsonl(tmp_path / "categories.jsonl", rows))


def test_check_alerts_accepts_the_planted_ssn(tmp_path: Path) -> None:
    check_alerts(_write_jsonl(tmp_path / "alerts.jsonl", _alert_rows()))


def test_check_alerts_rejects_a_quiet_run(tmp_path: Path) -> None:
    rows = _alert_rows()
    rows[0]["reason"] = "phone number"
    with pytest.raises(AssertionError, match="alert reason"):
        check_alerts(_write_jsonl(tmp_path / "alerts.jsonl", rows))


def test_check_doom_loop_accepts_planted_loop(tmp_path: Path) -> None:
    summaries = _write_jsonl(tmp_path / "summaries.jsonl", _summary_rows())
    check_doom_loop(_write_jsonl(tmp_path / "doom_loop.jsonl", _doom_loop_rows()), summaries)


def test_check_doom_loop_rejects_missing_loop_alert(tmp_path: Path) -> None:
    summaries = _write_jsonl(tmp_path / "summaries.jsonl", _summary_rows())
    rows = [{"name": "flag_sensitive_step", "reason": "ssn", "session_id": LOOP_SESSION}]
    with pytest.raises(AssertionError, match="never fired flag_doom_loop"):
        check_doom_loop(_write_jsonl(tmp_path / "doom_loop.jsonl", rows), summaries)


def test_check_project_and_cli(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    alerts = tmp_path / "cookbook" / ".trace-miner" / "alerts"
    alerts.mkdir(parents=True)
    _write_jsonl(output / "summaries.jsonl", _summary_rows())
    _write_jsonl(output / "pii.jsonl", _pii_rows())
    _write_jsonl(output / "frustration.jsonl", _frustration_rows())
    _write_jsonl(output / "categories.jsonl", _category_rows())
    _write_jsonl(alerts / "alerts.jsonl", _alert_rows())
    _write_jsonl(alerts / "doom_loop.jsonl", _doom_loop_rows())
    check_project(tmp_path)
    assert main([str(tmp_path)]) == 0


def test_cli_reports_missing_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path)]) == 1
    assert "hello-world checks failed" in capsys.readouterr().err
