from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import threading
import time

import pytest

from mine_trace.config import load_config
from mine_trace.errors import ConfigError, PipelineExecutionError
from mine_trace.model import DEFAULT_LM_MODEL, DEFAULT_LM_PROVIDER
from mine_trace.runtime import run_config, run_config_async, run_stream, run_stream_async


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

    run_config(config)

    assert read_jsonl(tmp_path / "kept.jsonl") == [{"id": "a", "keep": True, "value": 2}]
    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "value": 5}]


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

    assert "[mine-trace] map: double" in stream.getvalue()
    assert "2/2" in stream.getvalue()
    lines = [line for chunk in stream.getvalue().split("\r") for line in chunk.splitlines() if line]
    assert max(len(line) for line in lines) <= 120
    assert "#" not in stream.getvalue()


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

    assert "[mine-trace] reduce: merge" in stream.getvalue()
    assert "2/2" in stream.getvalue()


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

    async def run() -> None:
        task = asyncio.create_task(run_stream_async(config, max_records=2, poll_interval=0.01, progress=False))
        await asyncio.sleep(0.02)
        with (tmp_path / "input.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "a", "value": 2}) + "\n")
            handle.write(json.dumps({"id": "b", "value": 3}) + "\n")
        await task

    asyncio.run(run())

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"id": "a", "value": 4},
        {"id": "b", "value": 6},
    ]
    checkpoints = list((tmp_path / ".mine-trace" / "checkpoints").glob("*.json"))
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
    shutil.rmtree(Path(__file__).parent / ".mine-trace" / "runs")
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
    assert list((tmp_path / ".mine-trace" / "runs").glob("*/journal.jsonl"))


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


def test_lm_map_field_mapping_passthrough_and_model_override(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
class Field:
    def __init__(self, annotation):
        self.annotation = annotation


class LabelSignature:
    input_fields = {"text": Field(str)}
    output_fields = {"label": Field(str), "confidence": Field(float)}


class LabelModule:
    signature = LabelSignature()

    def __init__(self):
        self.models = []

    def with_model(self, model):
        self.models.append(model)
        return self

    def __call__(self, **kwargs):
        assert "model" not in kwargs
        return {"label": kwargs["text"].upper(), "confidence": 0.8}


label = LabelModule()
""",
        """
from: ./stages
lm:
  default: strong-model
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map:
          stage: label
          lm:
            model: cheap-model
""",
        [{"id": "a", "text": "alpha", "untouched": True}],
    )

    loaded = load_config(config)
    assert loaded.pipelines[0].steps[0].stage.lm_model == "cheap-model"
    run_config(loaded)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"confidence": 0.8, "id": "a", "label": "ALPHA", "text": "alpha", "untouched": True}
    ]
    assert loaded.module.label.models == ["cheap-model"]


def test_lm_config_validation_default_model_python_lm_options_and_filter_shape(tmp_path: Path) -> None:
    stages = """
class Field:
    def __init__(self, annotation):
        self.annotation = annotation


class LabelSignature:
    input_fields = {"text": Field(str)}
    output_fields = {"label": Field(str)}


class LabelModule:
    signature = LabelSignature()
    def __call__(self, **kwargs):
        return {"label": kwargs["text"]}


def py_stage(record):
    return record


label = LabelModule()
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

    python_lm = write_project(
        tmp_path / "python-lm",
        stages,
        """
from: ./stages
lm:
  default: model
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
  default: model
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
  provider: local
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "text": "hello"}],
    )
    with pytest.raises(ConfigError, match="openrouter"):
        load_config(bad_provider)


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
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"label": "ORE"})}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("mine_trace.runtime.urlopen", fake_urlopen)

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"id": "a", "label": "ORE", "text": "ore"}]
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["timeout"] == 60
    assert captured["body"]["model"] == DEFAULT_LM_MODEL
    assert captured["body"]["messages"][1]["role"] == "user"
    assert json.loads(captured["body"]["messages"][1]["content"]) == {
        "inputs": {"text": "ore"},
        "instructions": "Return a concise material label.",
        "output_fields": ["label"],
        "stage": "label",
    }


def test_lm_map_runs_concurrently_with_configured_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = write_project(
        tmp_path,
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
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_openrouter(stage, kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"label": kwargs["text"].upper()}

    monkeypatch.setattr("mine_trace.runtime._call_openrouter", fake_openrouter)

    run_config(config)

    assert max_active == 2
    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"id": str(i), "label": f"ITEM-{i}", "text": f"item-{i}"}
        for i in range(5)
    ]


def test_lm_missing_input_field_has_clear_error(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
class Field:
    def __init__(self, annotation):
        self.annotation = annotation


class LabelSignature:
    input_fields = {"text": Field(str)}
    output_fields = {"label": Field(str)}


class LabelModule:
    signature = LabelSignature()
    def __call__(self, **kwargs):
        return {"label": kwargs["text"]}


label = LabelModule()
""",
        """
from: ./stages
lm:
  default: model
pipelines:
  - in: input.jsonl
    out: output.jsonl
    pipeline:
      - map: label
""",
        [{"id": "a", "body": "hello"}],
    )

    with pytest.raises(PipelineExecutionError, match="missing input field `text`"):
        run_config(config)


def test_lm_filter_expand_and_reduce(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
class Field:
    def __init__(self, annotation):
        self.annotation = annotation


class SplitSignature:
    input_fields = {"text": Field(str)}
    output_fields = {"sessions": Field(list[dict])}


class KeepSignature:
    input_fields = {"score": Field(int)}
    output_fields = {"keep": Field(bool)}


class MergeSignature:
    input_fields = {"trace": Field(str), "label": Field(list[str])}
    output_fields = {"summary": Field(str)}


class Module:
    def __init__(self, signature, fn):
        self.signature = signature
        self.fn = fn

    def with_model(self, model):
        return self

    def __call__(self, **kwargs):
        return self.fn(kwargs)


split = Module(SplitSignature(), lambda kwargs: {
    "sessions": [
        {"id": f"{kwargs['text']}-1", "label": "hot", "score": 9},
        {"id": f"{kwargs['text']}-2", "label": "cold", "score": 2},
    ]
})
keep = Module(KeepSignature(), lambda kwargs: {"keep": kwargs["score"] > 5})
merge = Module(MergeSignature(), lambda kwargs: {
    "summary": f"{kwargs['trace']}:{','.join(kwargs['label'])}"
})
""",
        """
from: ./stages
lm:
  default: model
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

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"summary": "A:hot", "text": "session", "trace": "A", "id": "trace-a"}
    ]


def test_lm_reduce_by_group_gets_synthetic_parent_key(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
class Field:
    def __init__(self, annotation):
        self.annotation = annotation


class MergeSignature:
    input_fields = {"kind": Field(str), "value": Field(list[int])}
    output_fields = {"summary": Field(str)}


class MergeModule:
    signature = MergeSignature()
    def with_model(self, model):
        return self
    def __call__(self, **kwargs):
        return {"summary": f"{kwargs['kind']}={sum(kwargs['value'])}"}


merge = MergeModule()
""",
        """
from: ./stages
lm:
  default: model
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

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"kind": "ore", "summary": "ore=7"}]


def test_lm_invalid_output_flows_through_child_error_policy(tmp_path: Path) -> None:
    config = write_project(
        tmp_path,
        """
class Field:
    def __init__(self, annotation):
        self.annotation = annotation


class BadSignature:
    input_fields = {"text": Field(str)}
    output_fields = {"label": Field(str)}


class BadModule:
    signature = BadSignature()
    def with_model(self, model):
        return self
    def __call__(self, **kwargs):
        if kwargs["text"] == "bad":
            return {}
        return {"label": kwargs["text"]}


def split(record, ctx):
    for text in record["items"]:
        ctx.emit({"id": text, "text": text})


def merge(children, ctx):
    return {"labels": [child["label"] for child in children], "failures": len(ctx.failures)}


bad = BadModule()
""",
        """
from: ./stages
lm:
  default: model
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

    run_config(config)

    assert read_jsonl(tmp_path / "output.jsonl") == [{"failures": 1, "labels": ["good"]}]
