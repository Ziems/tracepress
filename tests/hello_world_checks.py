from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ATRX = "terminal-bench/atrx-vep-crispr"
PHOTONIC = "terminal-bench/photonic-waveguide-routing"
FREECAD = "terminal-bench/freecad-spring-clip"
EXPECTED_TASKS = {ATRX, PHOTONIC, FREECAD}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AssertionError(f"missing {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise AssertionError(f"{path} is empty")
    return rows


def row_task(row: dict[str, Any]) -> str | None:
    name = row.get("task_name")
    if isinstance(name, str) and name in EXPECTED_TASKS:
        return name
    blob = " ".join(str(row.get(key) or "") for key in ("task_name", "trajectory_path"))
    matches = [task for task in EXPECTED_TASKS if task.rsplit("/", 1)[-1] in blob]
    if len(matches) == 1:
        return matches[0]
    if isinstance(name, str) and name:
        return name
    return None


def by_task(rows: list[dict[str, Any]], *, path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AssertionError(f"{path} row {index} is not an object")
        name = row_task(row)
        if not name:
            raise AssertionError(
                f"{path} row {index} missing task_name (keys: {sorted(row)})"
            )
        if name in tasks:
            raise AssertionError(f"{path} has duplicate task_name {name}")
        tasks[name] = row
    missing = EXPECTED_TASKS - tasks.keys()
    extra = tasks.keys() - EXPECTED_TASKS
    if missing or extra:
        raise AssertionError(
            f"{path} expected tasks {sorted(EXPECTED_TASKS)}, "
            f"missing {sorted(missing)}, extra {sorted(extra)}"
        )
    return tasks


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _findings_blob(row: dict[str, Any]) -> str:
    findings = row.get("findings") or []
    parts = []
    for finding in findings:
        if isinstance(finding, dict):
            parts.append(str(finding.get("evidence") or ""))
            parts.append(str(finding.get("kind") or ""))
    return " ".join(parts)


def _cause(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("failure_cause") or "").split()).strip().rstrip(".").lower()


def check_summaries(path: Path) -> None:
    tasks = by_task(load_jsonl(path), path=path)

    atrx = tasks[ATRX]
    assert atrx["reported_passed"] is False, f"{ATRX} reported_passed"
    assert atrx["failed"] is True, f"{ATRX} should be diagnosed as failed"
    assert _cause(atrx) not in {"", "none"}, f"{ATRX} needs a failure cause"

    photonic = tasks[PHOTONIC]
    assert photonic["reported_passed"] is False, f"{PHOTONIC} reported_passed"
    assert photonic["failed"] is True, f"{PHOTONIC} should be diagnosed as failed"
    assert _cause(photonic) not in {"", "none"}, f"{PHOTONIC} needs a failure cause"

    freecad = tasks[FREECAD]
    assert freecad["reported_passed"] is True, f"{FREECAD} reported_passed"
    assert freecad["failed"] is False, f"{FREECAD} should be diagnosed as passed"
    assert str(freecad.get("trajectory_summary") or "").strip(), f"{FREECAD} missing summary"


def check_pii(path: Path) -> None:
    tasks = by_task(load_jsonl(path), path=path)

    atrx = tasks[ATRX]
    assert atrx["has_pii"] is True, f"{ATRX} should find planted SSN"
    kinds = {finding.get("kind") for finding in atrx.get("findings") or []}
    assert "ssn" in kinds, f"{ATRX} findings should include kind=ssn, got {sorted(kinds)}"
    assert "123456789" in _digits(_findings_blob(atrx)), f"{ATRX} evidence should quote 123-45-6789"

    photonic = tasks[PHOTONIC]
    assert photonic["has_pii"] is False, f"{PHOTONIC} should not report PII"
    assert not photonic.get("findings"), f"{PHOTONIC} should have empty findings"

    freecad = tasks[FREECAD]
    assert freecad["has_pii"] is True, f"{FREECAD} should find planted phone"
    kinds = {finding.get("kind") for finding in freecad.get("findings") or []}
    assert "phone" in kinds, f"{FREECAD} findings should include kind=phone, got {sorted(kinds)}"
    assert "5550101234" in _digits(_findings_blob(freecad)), (
        f"{FREECAD} evidence should quote (555) 010-1234"
    )


def check_frustration(path: Path) -> None:
    tasks = by_task(load_jsonl(path), path=path)

    atrx = tasks[ATRX]
    assert atrx["has_frustration"] is False, f"{ATRX} should not flag user frustration"

    photonic = tasks[PHOTONIC]
    assert photonic["has_frustration"] is True, f"{PHOTONIC} should flag the planted user turn"
    blob = _findings_blob(photonic).lower()
    assert "forever" in blob or "taking" in blob, (
        f"{PHOTONIC} evidence should quote the impatient user turn"
    )

    freecad = tasks[FREECAD]
    assert freecad["has_frustration"] is False, f"{FREECAD} should not flag user frustration"


def check_project(project: Path) -> None:
    output = project / "output"
    check_summaries(output / "summaries.jsonl")
    check_pii(output / "pii.jsonl")
    check_frustration(output / "frustration.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assert planted hello-world pipeline outcomes.")
    parser.add_argument("project", type=Path, help="Directory created by trace-miner init")
    args = parser.parse_args(argv)
    try:
        check_project(args.project.expanduser().resolve())
    except AssertionError as exc:
        print(f"hello-world checks failed: {exc}", file=sys.stderr)
        return 1
    print("ok: planted hello-world outcomes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
