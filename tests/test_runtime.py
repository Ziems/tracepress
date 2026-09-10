from __future__ import annotations

import json
from pathlib import Path

import pytest

from mine_trace.config import load_config
from mine_trace.errors import ConfigError, PipelineExecutionError
from mine_trace.runtime import run_config


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
