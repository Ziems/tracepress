from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import re
import threading
import time

import dspy
import pytest

from trace_miner.config import load_config
from trace_miner.errors import ConfigError, PipelineExecutionError
from trace_miner.model import DEFAULT_LM_MODEL, DEFAULT_LM_PROVIDER, RunStats, format_run_profile
from trace_miner.runtime import run_config, run_config_async, run_stream, run_stream_async


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def progress_text(stream: io.StringIO) -> str:
    return _ANSI.sub("", stream.getvalue())


def chat_completion(**fields: object) -> str:
    parts = []
    for name, value in fields.items():
        rendered = value if isinstance(value, str) else json.dumps(value)
        parts.append(f"[[ ## {name} ## ]]\n{rendered}")
    parts.append("[[ ## completed ## ]]")
    return "\n\n".join(parts)


def install_fake_lm(
    monkeypatch: pytest.MonkeyPatch,
    completions: dict[str, str] | None = None,
    *,
    sleep: float = 0.0,
    prompt_tokens: int = 12,
    completion_tokens: int = 4,
    cost: float | list[float] = 0.0021,
    note_usage: bool = True,
) -> type[dspy.BaseLM]:
    from trace_miner.model import LmUsage
    from trace_miner.runtime import _note_lm_usage

    costs = list(cost) if isinstance(cost, list) else None
    cost_lock = threading.Lock()

    class FakeLM(dspy.BaseLM):
        instances: list[FakeLM] = []
        context_lms: list[object] = []
        active = 0
        max_active = 0
        _gate = threading.Lock()

        def __init__(self, model: str, api_key=None, cache=False, extra_body=None, **kwargs):
            super().__init__(model=model, cache=cache)
            self.api_key = api_key
            self.extra_body = extra_body
            self.init_kwargs = kwargs
            type(self).instances.append(self)

        def __call__(self, *args, **kwargs):
            type(self).context_lms.append(dspy.settings.lm)
            with type(self)._gate:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            try:
                if sleep:
                    time.sleep(sleep)
                content = (completions or {}).get(self.model)
                if content is None:
                    content = (completions or {}).get("*", "")
                if note_usage:
                    with cost_lock:
                        this_cost = costs.pop(0) if costs else cost
                    _note_lm_usage(
                        LmUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=this_cost)
                    )
                    self.history.append(
                        {
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                            "cost": this_cost,
                        }
                    )
                return [content]
            finally:
                with type(self)._gate:
                    type(self).active -= 1

    monkeypatch.setattr(dspy, "LM", FakeLM)
    return FakeLM


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


def test_linear_pipeline_map_filter_stage_and_tee(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def keep(record):
    return record if record["keep"] else None

def label(record, ctx):
    ctx.emit({"id": record["id"], "value": record["value"] * 2})

def add_one(record):
    return {**record, "value": record["value"] + 1}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - filter: keep
      - tee: kept.jsonl
      - map: label
      - stage: add_one
""",
        [
            {"id": "a", "keep": True, "value": 2},
            {"id": "b", "keep": False, "value": 9},
        ],
    )

    result = run_config(config)

    assert read_jsonl(tmp_path / "kept.jsonl") == [{"id": "a", "keep": True, "value": 2}]
    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "value": 5}]
    assert result.stats.elapsed_s > 0
    assert result.stats.lm_calls == 0
    assert result.stats.lm_prompt_tokens == 0
    assert result.stats.lm_completion_tokens == 0
    assert result.stats.lm_cost_usd == 0


def test_alert_passes_records_through_and_copies_the_parent_id(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def split(record, ctx):
    for step in record["steps"]:
        ctx.emit(step)

def frustrated_user(record):
    if record.get("has_frustration"):
        return record["evidence"]
    return None

def collect(steps, ctx):
    return {"session_id": ctx.parent["session_id"], "texts": [step["text"] for step in steps]}

def not_a_reason(record):
    return {"reason": "nope"}
""",
        """
from: ./stages
id_field: session_id
pipelines:
  - name: findings
    in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: split
          pipeline:
            - alert: frustrated_user
          reduce: collect
""",
        [
            {
                "session_id": "s1",
                "steps": [
                    {"text": "ok"},
                    {"text": "slow", "has_frustration": True, "evidence": "taking forever"},
                ],
            }
        ],
    )

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"session_id": "s1", "texts": ["ok", "slow"]}]
    assert read_jsonl(tmp_path / ".trace-miner" / "alerts" / "findings.jsonl") == [
        {"name": "frustrated_user", "reason": "taking forever", "session_id": "s1"}
    ]

    broken = write_project(
        tmp_path / "broken",
        """
def not_a_reason(record):
    return {"reason": "nope"}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - alert: not_a_reason
""",
        [{"id": "a"}],
    )
    with pytest.raises(PipelineExecutionError, match="reason string"):
        run_config(broken)


def test_lm_alert_sends_module_outputs_when_the_bool_is_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Frustration(dspy.Signature):
    message: str = dspy.InputField()
    has_frustration: bool = dspy.OutputField()
    evidence: str = dspy.OutputField()


class DetectFrustration(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = Frustration

    def forward(self, message):
        frustrated = "forever" in message
        return dspy.Prediction(
            has_frustration=frustrated,
            evidence="taking forever" if frustrated else "",
        )


detect_frustration = DetectFrustration()
""",
        """
from: ./stages
id_field: session_id
pipelines:
  - name: alerts
    in: input.jsonl
    out: output.jsonl
    pipeline:
      - alert: detect_frustration
""",
        [
            {"session_id": "s1", "message": "looks good"},
            {"session_id": "s1", "message": "this is taking forever"},
        ],
    )
    install_fake_lm(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"session_id": "s1", "message": "looks good"},
        {"session_id": "s1", "message": "this is taking forever"},
    ]
    assert read_jsonl(tmp_path / ".trace-miner" / "alerts" / "alerts.jsonl") == [
        {
            "name": "detect_frustration",
            "session_id": "s1",
            "has_frustration": True,
            "evidence": "taking forever",
        }
    ]


def test_lm_alert_requires_one_bool_output(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Quote(dspy.Signature):
    message: str = dspy.InputField()
    evidence: str = dspy.OutputField()


quote = dspy.Predict(Quote)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - alert: quote
""",
        [{"message": "hi"}],
    )

    with pytest.raises(ConfigError, match="exactly one bool output"):
        load_config(config)


def test_async_python_stages_and_async_entrypoint(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
async def keep(record):
    return record if record["keep"] else None

async def label(record, ctx):
    ctx.emit({"id": record["id"], "value": record["value"] * 2})

async def add_one(record):
    return {**record, "value": record["value"] + 1}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - filter: keep
      - map: label
      - stage: add_one
""",
        [
            {"id": "a", "keep": True, "value": 2},
            {"id": "b", "keep": False, "value": 9},
        ],
    )

    result = asyncio.run(run_config_async(config))

    assert result.stats.emitted == 3
    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "value": 5}]


def test_progress_bar_reports_stage_progress(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def double(record):
    return {**record, "value": record["value"] * 2}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: double
""",
        [{"id": "a", "value": 2}, {"id": "b", "value": 3}],
    )
    stream = io.StringIO()

    run_config(config, progress=True, progress_stream=stream)

    rendered = progress_text(stream)
    assert "[trace-miner] map: double" in rendered
    assert "2/2" in rendered
    lines = [line for chunk in rendered.split("\r") for line in chunk.splitlines() if line]
    assert max(len(line) for line in lines) <= 120


def test_progress_bar_reports_reduce_progress(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def merge(children):
    return {"count": len(children)}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          every: 2
          pipeline: []
          reduce: merge
""",
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    )
    stream = io.StringIO()

    run_config(config, progress=True, progress_stream=stream)

    rendered = progress_text(stream)
    assert "[trace-miner] reduce: merge" in rendered
    assert "2/2" in rendered


def test_stream_processes_appended_lines_and_writes_checkpoint(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def double(record):
    return {**record, "value": record["value"] * 2}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: double
""",
        [],
    )

    async def run():
        task = asyncio.create_task(run_stream_async(config, max_records=2, poll_interval=0.01, progress=False))
        await asyncio.sleep(0.02)
        with (tmp_path / "input.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "a", "value": 2}) + "\n")
            handle.write(json.dumps({"id": "b", "value": 3}) + "\n")
        return await task

    result = asyncio.run(run())
    assert result.stats.elapsed_s > 0
    assert result.stats.lm_calls == 0

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"id": "a", "value": 4},
        {"id": "b", "value": 6},
    ]
    checkpoints = list((tmp_path / ".trace-miner" / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 1
    checkpoint = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert checkpoint["inputs"][str(tmp_path / "input.jsonl")]["offset"] == (tmp_path / "input.jsonl").stat().st_size


def test_stream_recreates_missing_journal_parent_before_commit(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
from pathlib import Path
import shutil

def remove_journal_parent(record):
    shutil.rmtree(Path(__file__).parent / ".trace-miner" / "runs")
    return record
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: remove_journal_parent
""",
        [{"id": "a"}],
    )

    run_stream(config, max_records=1, poll_interval=0.01, progress=False)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a"}]
    assert list((tmp_path / ".trace-miner" / "runs").glob("*/journal.jsonl"))


def test_stream_ignores_partial_line_until_newline(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def double(record):
    return {**record, "value": record["value"] * 2}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: double
""",
        [],
    )
    (tmp_path / "input.jsonl").write_text('{"id":"a","value":2}', encoding="utf-8")

    run_stream(config, stop_when_idle=True, poll_interval=0.01, progress=False)

    assert not (tmp_path / "output.jsonl").exists()

    with (tmp_path / "input.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    run_stream(config, max_records=1, poll_interval=0.01, progress=False)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "value": 4}]


def test_stream_resumes_from_checkpoint(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def double(record):
    return {**record, "value": record["value"] * 2}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: double
""",
        [{"id": "a", "value": 2}],
    )

    run_stream(config, max_records=1, poll_interval=0.01, progress=False)
    with (tmp_path / "input.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "b", "value": 3}) + "\n")
    run_stream(config, max_records=1, poll_interval=0.01, progress=False)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"id": "a", "value": 4},
        {"id": "b", "value": 6},
    ]


def test_stream_from_end_skips_existing_lines(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def double(record):
    return {**record, "value": record["value"] * 2}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: double
""",
        [{"id": "old", "value": 2}],
    )

    run_stream(config, from_end=True, stop_when_idle=True, poll_interval=0.01, progress=False)
    with (tmp_path / "input.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "new", "value": 3}) + "\n")
    run_stream(config, max_records=1, poll_interval=0.01, progress=False)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "new", "value": 6}]


def test_stream_skips_blank_lines(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def double(record):
    return {**record, "value": record["value"] * 2}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: double
""",
        [],
    )
    (tmp_path / "input.jsonl").write_text("\n\n" + json.dumps({"id": "a", "value": 2}) + "\n", encoding="utf-8")

    run_stream(config, max_records=3, poll_interval=0.01, progress=False)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "value": 4}]


def test_stream_routes_records_through_pipeline_dag(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def double(record):
    return {**record, "value": record["value"] * 2}

def add_one(record):
    return {**record, "value": record["value"] + 1}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: intermediate.jsonl
    pipeline:
      - map: double
  - in: intermediate.jsonl
    out: output.jsonl
    pipeline:
      - map: add_one
""",
        [{"id": "a", "value": 2}],
    )

    run_stream(config, max_records=1, poll_interval=0.01, progress=False)

    assert read_jsonl(tmp_path / "intermediate.jsonl") == [{"id": "a", "value": 4}]
    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "value": 5}]


def test_stream_rejects_by_groups(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def total(children):
    return {"total": len(children)}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          by: kind
          reduce: total
""",
        [{"id": "a", "kind": "x"}],
    )

    with pytest.raises(PipelineExecutionError, match="close condition"):
        run_stream(config, max_records=1, progress=False)


def test_stream_flushes_incomplete_every_on_shutdown(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def merge(children, ctx):
    return {"window": ctx.key, "ids": [child["id"] for child in children]}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          every: 2
          reduce: merge
""",
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    )

    run_stream(config, max_records=3, poll_interval=0.01, progress=False)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"ids": ["a", "b"], "window": 1},
        {"ids": ["c"], "window": 2},
    ]


def test_expand_group_handles_varying_sizes_and_empty_groups(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def split(record, ctx):
    for value in record["values"]:
        ctx.emit({"value": value})

def even(record):
    return record if record["value"] % 2 == 0 else None

def merge(children, ctx):
    return {"parent": ctx.parent["id"], "values": [child["value"] for child in children]}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: split
          pipeline:
            - filter: even
          reduce: merge
""",
        [
            {"id": "a", "values": list(range(10))},
            {"id": "b", "values": []},
            {"id": "c", "values": list(range(30))},
        ],
    )

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"parent": "a", "values": [0, 2, 4, 6, 8]},
        {"parent": "b", "values": []},
        {"parent": "c", "values": list(range(0, 30, 2))},
    ]


def test_nested_groups_and_reduce_fanout_adjust_outer_count(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def docs(record, ctx):
    for doc in record["docs"]:
        ctx.emit(doc)

def paragraphs(doc, ctx):
    for paragraph in doc["paragraphs"]:
        ctx.emit({"text": paragraph})

def paragraph_pair(children, ctx):
    ctx.emit({"doc": ctx.parent["id"], "count": len(children), "copy": 1})
    ctx.emit({"doc": ctx.parent["id"], "count": len(children), "copy": 2})

def merge_docs(children, ctx):
    return {
        "trace": ctx.parent["id"],
        "items": [f"{child['doc']}:{child['copy']}:{child['count']}" for child in children],
    }
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: docs
          pipeline:
            - group:
                expand: paragraphs
                reduce: paragraph_pair
          reduce: merge_docs
""",
        [
            {
                "id": "t1",
                "docs": [
                    {"id": "d1", "paragraphs": ["a", "b"]},
                    {"id": "d2", "paragraphs": ["c"]},
                ],
            }
        ],
    )

    result = run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"trace": "t1", "items": ["d1:1:2", "d1:2:2", "d2:1:1", "d2:2:1"]}
    ]
    assert result.stats.groups_opened == 3
    assert result.stats.groups_reduced == 3


def test_by_groups_reduce_at_end_of_input(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def total(children, ctx):
    return {"key": ctx.key, "total": sum(child["value"] for child in children)}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          by: kind
          reduce: total
""",
        [
            {"id": 1, "kind": "x", "value": 2},
            {"id": 2, "kind": "y", "value": 5},
            {"id": 3, "kind": "x", "value": 7},
        ],
    )

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"key": "x", "total": 9},
        {"key": "y", "total": 5},
    ]


def test_nested_by_group_replaces_outer_obligations_as_a_batch(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def split(record, ctx):
    for item in record["items"]:
        ctx.emit(item)

def merge_kind(children, ctx):
    return {"kind": ctx.key, "total": sum(child["value"] for child in children)}

def merge_parent(children, ctx):
    return {"parent": ctx.parent["id"], "children": children}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: split
          pipeline:
            - group:
                by: kind
                reduce: merge_kind
          reduce: merge_parent
""",
        [
            {
                "id": "p",
                "items": [
                    {"kind": "same", "value": 2},
                    {"kind": "same", "value": 3},
                ],
            }
        ],
    )

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"children": [{"kind": "same", "total": 5}], "parent": "p"}
    ]


def test_every_groups_final_short_window(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def merge(children, ctx):
    return {"window": ctx.key, "ids": [child["id"] for child in children]}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          every: 2
          reduce: merge
""",
        [{"id": 1}, {"id": 2}, {"id": 3}],
    )

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"ids": [1, 2], "window": 1},
        {"ids": [3], "window": 2},
    ]


def test_stage_contract_errors(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def duplicate(record, ctx):
    ctx.emit(record)
    ctx.emit(record)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: duplicate
""",
        [{"id": "a"}],
    )

    with pytest.raises(PipelineExecutionError, match="expected exactly 1"):
        run_config(config)


def test_return_and_emit_raises(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def bad(record, ctx):
    ctx.emit(record)
    return record
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - stage: bad
""",
        [{"id": "a"}],
    )

    with pytest.raises(PipelineExecutionError, match="both returned"):
        run_config(config)


def test_returned_list_is_one_record(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def as_list(record):
    return [record["id"]]
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: as_list
""",
        [{"id": "a"}],
    )

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [["a"]]


def test_child_error_partial_and_fail_and_skip(tmp_path: Path) -> None:
    stages = """
def split(record, ctx):
    for value in [1, 2, 3]:
        ctx.emit({"id": f"{record['id']}-{value}", "value": value})

def sometimes_bad(record):
    if record["value"] == 2:
        raise ValueError("boom")
    return record

def merge(children, ctx):
    return {"parent": ctx.parent["id"], "values": [child["value"] for child in children], "failures": len(ctx.failures)}
"""
    base_config = """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: split
          on_child_error: {policy}
          pipeline:
            - map: sometimes_bad
          reduce: merge
"""
    partial = write_project(tmp_path / "partial", stages, base_config.format(policy="partial"), [{"id": "p"}])
    result = run_config(partial)
    assert read_jsonl(tmp_path / "partial" / "output.jsonl") == [{"failures": 1, "parent": "p", "values": [1, 3]}]
    assert result.stats.failed == 1

    fail_dir = tmp_path / "fail"
    fail_dir.mkdir()
    fail = write_project(fail_dir, stages, base_config.format(policy="fail"), [{"id": "f"}])
    with pytest.raises(PipelineExecutionError, match="f-2"):
        run_config(fail)

    skip_dir = tmp_path / "skip"
    skip_dir.mkdir()
    skip = write_project(skip_dir, stages, base_config.format(policy="skip"), [{"id": "s"}])
    result = run_config(skip)
    assert read_jsonl(skip_dir / "output.jsonl") == []
    assert result.manifest.skipped_groups[0]["failing_child"] == "s-2"


def test_load_time_validation_unknown_stage_and_bad_signature(tmp_path: Path) -> None:
    unknown = write_project(
        tmp_path / "unknown",
        "def ok(record): return record\n",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: missing
""",
        [{"id": 1}],
    )
    with pytest.raises(ConfigError, match="Unknown stage"):
        load_config(unknown)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad = write_project(
        bad_dir,
        "def bad(a, b, c): return a\n",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: bad
""",
        [{"id": 1}],
    )
    with pytest.raises(ConfigError, match="cannot be called"):
        load_config(bad)


def test_path_dag_duplicate_out_and_tee_dependency(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def tag(record):
    return {**record, "tag": True}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: first.jsonl
    pipeline:
      - tee: intermediate.jsonl
  - in: ./intermediate.jsonl
    out: second.jsonl
    pipeline:
      - map: tag
""",
        [{"id": 1}],
    )

    run_config(config)

    assert read_jsonl(tmp_path / "second.jsonl") == [{"id": 1, "tag": True}]

    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    duplicate = write_project(
        duplicate_dir,
        "def ok(record): return record\n",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: ./same.jsonl
    pipeline: []
  - in: input.jsonl
    out: sub/../same.jsonl
    pipeline: []
""",
        [{"id": 1}],
    )
    with pytest.raises(ConfigError, match="Multiple pipeline"):
        load_config(duplicate)


def test_post_mvp_features_are_rejected(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
def ok(record):
    return record
""",
        """
from: ./stages
concurrency:
  max_in_flight: 5
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: ok
""",
        [{"id": 1}],
    )

    with pytest.raises(ConfigError, match="post-MVP"):
        load_config(config)


def test_lm_map_field_mapping_passthrough_and_model_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()
    confidence: float = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
lm:
  default: deepseek/deepseek-v4.1-flash
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map:
          stage: label
          lm:
            model: openai/gpt-4o
""",
        [{"id": "a", "text": "alpha", "untouched": True}],
    )
    FakeLM = install_fake_lm(
        monkeypatch,
        {"*": chat_completion(label="ALPHA", confidence=0.8)},
        cost=0,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    loaded = load_config(config)
    assert loaded.pipelines[0].steps[0].stage.lm_model == "openai/gpt-4o"
    result = run_config(loaded)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"confidence": 0.8, "id": "a", "label": "ALPHA", "text": "alpha", "untouched": True}
    ]
    assert FakeLM.instances[0].model == "openrouter/openai/gpt-4o"
    assert result.stats.lm_calls == 1
    assert result.stats.lm_cost_usd == 0


def test_lm_config_validation_default_model_python_lm_options_and_filter_shape(tmp_path: Path) -> None:
    stages = """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


def py_stage(record):
    return record


label = dspy.Predict(Label)
"""
    default_model = write_project(
        tmp_path / "default-model",
        stages,
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "hello"}],
    )
    loaded = load_config(default_model)
    stage = loaded.pipelines[0].steps[0].stage
    assert stage is not None
    assert stage.lm_provider == DEFAULT_LM_PROVIDER
    assert stage.lm_model == DEFAULT_LM_MODEL
    assert loaded.lm_cache is True

    python_lm = write_project(
        tmp_path / "python-lm",
        stages,
        """
from: ./stages
lm:
  default: deepseek/deepseek-v4.1-flash
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map:
          stage: py_stage
          lm:
            model: model
""",
        [{"id": "a", "text": "hello"}],
    )
    with pytest.raises(ConfigError, match="Python stage"):
        load_config(python_lm)

    bad_filter = write_project(
        tmp_path / "bad-filter",
        stages,
        """
from: ./stages
lm:
  default: deepseek/deepseek-v4.1-flash
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - filter: label
""",
        [{"id": "a", "text": "hello"}],
    )
    with pytest.raises(ConfigError, match="LM filter"):
        load_config(bad_filter)

    bad_provider = write_project(
        tmp_path / "bad-provider",
        stages,
        """
from: ./stages
lm:
  provider: "open ai"
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "hello"}],
    )
    with pytest.raises(ConfigError, match="one of: anthropic, openai, openrouter"):
        load_config(bad_provider)

    unknown_model = write_project(
        tmp_path / "unknown-model",
        stages,
        """
from: ./stages
lm:
  provider: openai
  default: gpt-4o-mini
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map:
          stage: label
          lm:
            model: gpt-4o-mini-not-a-model
""",
        [{"id": "a", "text": "hello"}],
    )
    with pytest.raises(ConfigError, match="Unknown model `gpt-4o-mini-not-a-model` for `openai`"):
        load_config(unknown_model)


def test_lm_openrouter_default_provider_invocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    \"\"\"Return a concise material label.\"\"\"

    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}],
    )
    FakeLM = install_fake_lm(
        monkeypatch,
        {"*": chat_completion(label="ORE")},
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "label": "ORE", "text": "ore"}]
    assert len(FakeLM.instances) == 1
    lm = FakeLM.instances[0]
    assert lm.model == f"openrouter/{DEFAULT_LM_MODEL}"
    assert lm.api_key == "test-key"
    assert lm.cache is True
    assert lm.extra_body == {"usage": {"include": True}}
    from trace_miner.home import cache_dir

    assert Path(os.environ["DSPY_CACHEDIR"]) == cache_dir()
    assert cache_dir().is_dir()
    assert result.stats.elapsed_s > 0
    assert result.stats.lm_calls == 1
    assert result.stats.lm_prompt_tokens == 12
    assert result.stats.lm_completion_tokens == 4
    assert result.stats.lm_cost_usd == 0.0021


def test_lm_provider_prefixes_the_model_and_reads_its_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
"""
    config = """
from: ./stages
lm:
  provider: openai
  default: gpt-4o-mini
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
"""
    path = write_project(tmp_path, stages, config, [{"id": "a", "text": "ore"}])
    FakeLM = install_fake_lm(monkeypatch, {"*": chat_completion(label="ORE")})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    loaded = load_config(path)
    assert loaded.lm_provider == "openai"
    result = run_config(loaded)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "label": "ORE", "text": "ore"}]
    lm = FakeLM.instances[0]
    assert lm.model == "openai/gpt-4o-mini"
    assert lm.api_key == "sk-test"
    assert lm.extra_body is None
    assert result.stats.lm_calls == 1

    missing = write_project(
        tmp_path / "missing-key",
        stages,
        config,
        [{"id": "a", "text": "ore"}],
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(PipelineExecutionError, match="OPENAI_API_KEY is required for `openai`"):
        run_config(missing)


def test_lm_cache_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
lm:
  cache: false
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}],
    )
    FakeLM = install_fake_lm(monkeypatch, {"*": chat_completion(label="ORE")})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    loaded = load_config(config)
    assert loaded.lm_cache is False
    run_config(loaded)
    assert FakeLM.instances[0].cache is False


def test_lm_stage_must_be_a_dspy_module(tmp_path: Path) -> None:
    duck = write_project(
        tmp_path / "duck",
        """
class Field:
    def __init__(self, annotation):
        self.annotation = annotation


class LabelSignature:
    input_fields = {"text": Field(str)}
    output_fields = {"label": Field(str)}


class LabelModule:
    signature = LabelSignature()


label = LabelModule()
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}],
    )
    with pytest.raises(ConfigError, match="not a DSPy module"):
        load_config(duck)

    bare_signature = write_project(
        tmp_path / "signature",
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = Label
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}],
    )
    with pytest.raises(ConfigError, match="DSPy signature"):
        load_config(bare_signature)

    unsigned = write_project(
        tmp_path / "unsigned",
        """
import dspy


class Program(dspy.Module):
    def forward(self, text):
        return dspy.Prediction(label=text)


label = Program()
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}],
    )
    with pytest.raises(ConfigError, match="input_fields"):
        load_config(unsigned)


def test_lm_cache_must_be_boolean(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
lm:
  cache: 1
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}],
    )
    with pytest.raises(ConfigError, match="lm.cache"):
        load_config(config)


def test_lm_pydantic_output_is_structured_and_jsonl_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_project(
        tmp_path,
        """
from typing import Literal

import dspy
from pydantic import BaseModel, Field as PydField


class Finding(BaseModel):
    kind: Literal["ssn", "phone"]
    evidence: str = PydField(description="Quoted span.")


class Detect(dspy.Signature):
    \"\"\"Detect identifiers.\"\"\"

    text: str = dspy.InputField()
    findings: list[Finding] = dspy.OutputField()


detect = dspy.Predict(Detect)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: detect
""",
        [{"id": "a", "text": "ssn 123-45-6789"}],
    )
    install_fake_lm(
        monkeypatch,
        {"*": chat_completion(findings=[{"kind": "ssn", "evidence": "123-45-6789"}])},
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"findings": [{"evidence": "123-45-6789", "kind": "ssn"}], "id": "a", "text": "ssn 123-45-6789"}
    ]


def test_lm_openrouter_missing_api_key_fails_before_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}],
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(PipelineExecutionError, match="OPENROUTER_API_KEY is required"):
        run_config(config)


def test_lm_per_step_models_use_separate_context_lms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


class Score(dspy.Signature):
    text: str = dspy.InputField()
    score: float = dspy.OutputField()


label = dspy.Predict(Label)
score = dspy.Predict(Score)
""",
        """
from: ./stages
lm:
  max_concurrent: 2
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map:
          stage: label
          lm:
            model: openai/gpt-4o
      - map:
          stage: score
          lm:
            model: anthropic/claude-sonnet-4
""",
        [{"id": "a", "text": "ore"}, {"id": "b", "text": "coal"}],
    )
    FakeLM = install_fake_lm(
        monkeypatch,
        {
            "openrouter/openai/gpt-4o": chat_completion(label="ORE"),
            "openrouter/anthropic/claude-sonnet-4": chat_completion(score=0.9),
        },
        sleep=0.02,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    run_config(config)

    models = sorted(lm.model for lm in FakeLM.instances)
    assert models == ["openrouter/anthropic/claude-sonnet-4", "openrouter/openai/gpt-4o"]
    assert len({id(lm) for lm in FakeLM.instances}) == 2
    assert all(isinstance(observed, FakeLM) for observed in FakeLM.context_lms)
    assert {observed.model for observed in FakeLM.context_lms} == {
        "openrouter/anthropic/claude-sonnet-4",
        "openrouter/openai/gpt-4o",
    }
    from dspy.dsp.utils.settings import main_thread_config

    assert main_thread_config.get("lm") is None


def test_lm_overlapping_same_model_costs_are_summed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
lm:
  max_concurrent: 2
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}, {"id": "b", "text": "coal"}],
    )
    install_fake_lm(
        monkeypatch,
        {"*": chat_completion(label="ORE")},
        sleep=0.03,
        cost=[0.001, 0.004],
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = run_config(config)

    assert result.stats.lm_calls == 2
    assert result.stats.lm_cost_usd == pytest.approx(0.005)


def test_usage_from_lm_response_reads_openrouter_cost() -> None:
    from types import SimpleNamespace

    from trace_miner.runtime import _usage_from_lm_response

    usage = _usage_from_lm_response(
        SimpleNamespace(
            cost=None,
            usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0015},
        )
    )
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.cost_usd == 0.0015


def test_python_stage_that_invokes_dspy_names_the_stage(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


_label = dspy.Predict(Label)


def split(record, ctx):
    ctx.emit({"id": record["id"], "text": record["text"]})


def wrap_label(record):
    pred = _label(text=record["text"])
    return {**record, "label": pred.label}


def merge(children, ctx):
    return {"labels": [child.get("label") for child in children]}
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: split
          pipeline:
            - map: wrap_label
          reduce: merge
""",
        [{"id": "a", "text": "ore"}],
    )

    with pytest.raises(PipelineExecutionError, match="Stage `wrap_label` invoked DSPy") as exc_info:
        run_config(config)
    assert "expand stage" not in str(exc_info.value)
    assert "split" not in str(exc_info.value)


def test_lm_openrouter_missing_usage_still_counts_the_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "ore"}],
    )
    install_fake_lm(monkeypatch, {"*": chat_completion(label="ORE")}, note_usage=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "label": "ORE", "text": "ore"}]
    assert result.stats.lm_calls == 1
    assert result.stats.lm_prompt_tokens == 0
    assert result.stats.lm_completion_tokens == 0
    assert result.stats.lm_cost_usd == 0


def test_format_run_profile_totals() -> None:
    assert format_run_profile(RunStats(elapsed_s=0.12)) == "0.12s"
    assert (
        format_run_profile(
            RunStats(
                elapsed_s=12.4,
                lm_calls=3,
                lm_prompt_tokens=4000,
                lm_completion_tokens=210,
                lm_cost_usd=0.0021,
            )
        )
        == "12.4s  3 LM calls  4,210 tokens  $0.0021"
    )


def test_lm_map_runs_concurrently_with_configured_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
lm:
  max_concurrent: 2
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": str(i), "text": f"item-{i}"} for i in range(5)],
    )
    FakeLM = install_fake_lm(monkeypatch, {"*": chat_completion(label="ITEM")}, sleep=0.02)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    run_config(config)

    assert FakeLM.max_active == 2
    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"id": str(i), "label": "ITEM", "text": f"item-{i}"}
        for i in range(5)
    ]


def test_lm_missing_input_field_has_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class Label(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


label = dspy.Predict(Label)
""",
        """
from: ./stages
lm:
  default: deepseek/deepseek-v4.1-flash
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "body": "hello"}],
    )
    install_fake_lm(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(PipelineExecutionError, match="missing input field `text`"):
        run_config(config)


def test_lm_filter_expand_and_reduce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class SplitSig(dspy.Signature):
    text: str = dspy.InputField()
    sessions: list[dict] = dspy.OutputField()


class KeepSig(dspy.Signature):
    score: int = dspy.InputField()
    keep: bool = dspy.OutputField()


class MergeSig(dspy.Signature):
    trace: str = dspy.InputField()
    label: list[str] = dspy.InputField()
    summary: str = dspy.OutputField()


class Split(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = SplitSig

    def forward(self, text):
        return dspy.Prediction(sessions=[
            {"id": f"{text}-1", "label": "hot", "score": 9},
            {"id": f"{text}-2", "label": "cold", "score": 2},
        ])


class Keep(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = KeepSig

    def forward(self, score):
        return dspy.Prediction(keep=score > 5)


class Merge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = MergeSig

    def forward(self, trace, label):
        return dspy.Prediction(summary=f"{trace}:{','.join(label)}")


split = Split()
keep = Keep()
merge = Merge()
""",
        """
from: ./stages
lm:
  default: deepseek/deepseek-v4.1-flash
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: split
          pipeline:
            - filter: keep
          reduce: merge
""",
        [{"id": "trace-a", "trace": "A", "text": "session"}],
    )
    install_fake_lm(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"summary": "A:hot", "text": "session", "trace": "A", "id": "trace-a"}
    ]


def test_lm_reduce_by_group_gets_synthetic_parent_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class MergeSig(dspy.Signature):
    kind: str = dspy.InputField()
    value: list[int] = dspy.InputField()
    summary: str = dspy.OutputField()


class Merge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = MergeSig

    def forward(self, kind, value):
        return dspy.Prediction(summary=f"{kind}={sum(value)}")


merge = Merge()
""",
        """
from: ./stages
lm:
  default: deepseek/deepseek-v4.1-flash
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          by: kind
          reduce: merge
""",
        [
            {"id": 1, "kind": "ore", "value": 2},
            {"id": 2, "kind": "ore", "value": 5},
        ],
    )
    install_fake_lm(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"kind": "ore", "summary": "ore=7"}]


def test_lm_invalid_output_flows_through_child_error_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_project(
        tmp_path,
        """
import dspy


class BadSig(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()


class Bad(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = BadSig

    def forward(self, text):
        if text == "bad":
            return {}
        return dspy.Prediction(label=text)


def split(record, ctx):
    for text in record["items"]:
        ctx.emit({"id": text, "text": text})


def merge(children, ctx):
    return {"labels": [child["label"] for child in children], "failures": len(ctx.failures)}


bad = Bad()
""",
        """
from: ./stages
lm:
  default: deepseek/deepseek-v4.1-flash
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - group:
          expand: split
          on_child_error: partial
          pipeline:
            - map: bad
          reduce: merge
""",
        [{"id": "parent", "items": ["good", "bad"]}],
    )
    install_fake_lm(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"failures": 1, "labels": ["good"]}]
