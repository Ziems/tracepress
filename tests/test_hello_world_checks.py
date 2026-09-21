from __future__ import annotations

import json
from pathlib import Path

import pytest

from hello_world_checks import (
    ATRX,
    FREECAD,
    PHOTONIC,
    check_frustration,
    check_pii,
    check_project,
    check_summaries,
    main,
)


def _write_jsonl(path: Path, rows: list[dict[object, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _summary_rows() -> list[dict[object, object]]:
    return [
        {
            "task_name": ATRX,
            "reported_passed": False,
            "failed": True,
            "failure_cause": "HGVS annotations did not match the verifier.",
            "trajectory_summary": "The agent catalogued ATRX variants but failed checks.",
        },
        {
            "task_name": PHOTONIC,
            "reported_passed": False,
            "failed": True,
            "failure_cause": "Routing never finished after the user asked it to stop delaying.",
            "trajectory_summary": "Waveguide routing hit an output-token limit.",
        },
        {
            "task_name": FREECAD,
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


def test_check_project_and_cli(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_jsonl(output / "summaries.jsonl", _summary_rows())
    _write_jsonl(output / "pii.jsonl", _pii_rows())
    _write_jsonl(output / "frustration.jsonl", _frustration_rows())
    check_project(tmp_path)
    assert main([str(tmp_path)]) == 0


def test_cli_reports_missing_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path)]) == 1
    assert "hello-world checks failed" in capsys.readouterr().err
