from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from trace_miner.client import Client
from trace_miner.config import load_config
from trace_miner.errors import ConfigError, PipelineExecutionError, TraceMinerError
from trace_miner.init import init_project
from trace_miner.serve import start_serve


def write_project(tmp_path: Path, stages: str, config: str, inputs: dict[str, list[object]] | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "stages.py").write_text(stages, encoding="utf-8")
    for name, records in (inputs or {}).items():
        write_jsonl(tmp_path / name, records)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(config, encoding="utf-8")
    return config_path


def write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def read_jsonl(path: Path) -> list[object]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


STAGES = """
def doom_loop(record):
    if not record.get("flag"):
        return None
    return record.get("reason") or None
"""

CONFIG = """
from: ./stages
id_field: rollout_id
pipelines:
  - name: alerts
    in: input.jsonl
    out: alerts.jsonl
    pipeline:
      - alert: doom_loop
"""


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def open_events(base_url: str, *, rollout_id: str = "r1", last_event_id: str | None = None):
    query = urlencode({"output": "alerts", "field": "rollout_id", "rollout_id": rollout_id})
    headers = {"Accept": "text/event-stream"}
    if last_event_id is not None:
        headers["Last-Event-ID"] = last_event_id
    return urlopen(Request(f"{base_url}/v1/events?{query}", headers=headers), timeout=5)


def read_event(response) -> tuple[str | None, dict]:
    event_id = None
    data: list[str] = []
    while True:
        raw = response.readline()
        if not raw:
            raise AssertionError("alert stream closed")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                return event_id, json.loads("\n".join(data))
            continue
        if line.startswith("id:"):
            event_id = line[3:].lstrip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


def test_alert_sets_latest_for_the_matching_id(tmp_path: Path) -> None:
    config = write_project(tmp_path, STAGES, CONFIG, {"input.jsonl": []})
    server = start_serve(config, port=0, poll_interval=30, progress=False)
    try:
        client = Client(server.url)
        with client.watch("alerts", rollout_id="r1") as alerts:
            client.post_step({"rollout_id": "r1", "tool_calls": ["bash"]})
            assert wait_until(lambda: {"rollout_id": "r1", "tool_calls": ["bash"]} in read_jsonl(tmp_path / "alerts.jsonl"))
            assert alerts.latest is None

            client.post_step({"rollout_id": "r2", "flag": True, "reason": "other"})
            assert wait_until(lambda: any(record.get("rollout_id") == "r2" for record in read_jsonl(tmp_path / ".trace-miner" / "alerts" / "alerts.jsonl")))
            assert alerts.latest is None

            client.post_step({"rollout_id": "r1", "flag": True, "reason": "repeated bash"})
            assert wait_until(lambda: alerts.latest is not None)
            assert alerts.latest is not None
            assert alerts.latest.name == "doom_loop"
            assert alerts.latest.reason == "repeated bash"
    finally:
        server.close()


def test_last_event_id_resumes_after_the_cursor(tmp_path: Path) -> None:
    config = write_project(tmp_path, STAGES, CONFIG, {"input.jsonl": []})
    server = start_serve(config, port=0, poll_interval=30, progress=False)
    response = None
    resume = None
    try:
        response = open_events(server.url)
        post(server.url, json.dumps({"rollout_id": "r1", "flag": True, "reason": "slow"}).encode("utf-8"))
        event_id, record = read_event(response)
        assert record == {"name": "doom_loop", "reason": "slow", "rollout_id": "r1"}
        assert event_id

        post(server.url, json.dumps({"rollout_id": "r1", "flag": True, "reason": "repeated bash"}).encode("utf-8"))
        _, second = read_event(response)
        assert second["reason"] == "repeated bash"
        response.close()
        response = None

        resume = open_events(server.url, last_event_id=event_id)
        _, resumed = read_event(resume)
        assert resumed == {"name": "doom_loop", "reason": "repeated bash", "rollout_id": "r1"}
    finally:
        if response is not None:
            response.close()
        if resume is not None:
            resume.close()
        server.close()


def test_unknown_output_name_is_rejected(tmp_path: Path) -> None:
    config = write_project(tmp_path, STAGES, CONFIG, {"input.jsonl": []})
    server = start_serve(config, port=0, poll_interval=30, progress=False)
    try:
        with pytest.raises(TraceMinerError, match="404"):
            with Client(server.url).watch("missing", rollout_id="r1"):
                pass
    finally:
        server.close()


def test_hello_client_demo_stops_on_the_planted_ssn(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    init_project(dest)
    server = start_serve(dest / "alerts.yaml", port=0, poll_interval=30, progress=False)
    try:
        completed = subprocess.run(
            [sys.executable, str(dest / "client_demo.py"), server.url],
            cwd=dest,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 1
        assert "aborted terminal-bench/atrx-vep-crispr step 2: ssn" in completed.stderr
        assert "replay terminal-bench/atrx-vep-crispr step 1" in completed.stdout
        assert "photonic" not in completed.stdout
        alerts = read_jsonl(dest / ".trace-miner" / "alerts" / "alerts.jsonl")
        assert len(alerts) == 1
        assert alerts[0]["name"] == "flag_sensitive_step"
        assert alerts[0]["reason"] == "ssn"
        assert alerts[0]["session_id"]
    finally:
        server.close()


def test_serve_rejects_multiple_root_inputs(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        STAGES,
        """
from: ./stages
pipelines:
  - name: left
    in: a.jsonl
    out: a-out.jsonl
    pipeline: []
  - name: right
    in: b.jsonl
    out: b-out.jsonl
    pipeline: []
""",
        {"a.jsonl": [], "b.jsonl": []},
    )
    with pytest.raises(PipelineExecutionError, match="exactly one root input"):
        start_serve(config, port=0, progress=False)


def test_duplicate_pipeline_name_fails_at_load(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        STAGES,
        """
from: ./stages
pipelines:
  - name: alerts
    in: input.jsonl
    out: one.jsonl
    pipeline: []
  - name: alerts
    in: input.jsonl
    out: two.jsonl
    pipeline: []
""",
        {"input.jsonl": []},
    )
    with pytest.raises(ConfigError, match="Duplicate pipeline name `alerts`"):
        load_config(config)


def post(base_url: str, body: bytes) -> None:
    headers = {"Content-Type": "application/json"}
    request = Request(base_url + "/v1/records", data=body, method="POST", headers=headers)
    try:
        with urlopen(request, timeout=5) as response:
            assert response.status == 202
    except HTTPError as exc:
        raise AssertionError(exc.read().decode()) from exc
