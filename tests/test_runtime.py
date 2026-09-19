from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import threading
import time

import dspy
import pytest

from trace_miner.config import load_config
from trace_miner.errors import ConfigError, PipelineExecutionError
from trace_miner.model import DEFAULT_LM_MODEL, DEFAULT_LM_PROVIDER, RunStats, format_run_profile
from trace_miner.runtime import run_config, run_config_async, run_stream, run_stream_async


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
) -> type[dspy.BaseLM]:
    from trace_miner.model import LmUsage
    from trace_miner.runtime import _note_lm_usage

    costs = list(cost) if isinstance(cost, list) else None
    cost_lock = threading.Lock()

    class FakeLM(dspy.BaseLM):
        instances: list[FakeLM] = []
        context_lms: list[object] = []

        def __init__(self, model: str, api_key=None, cache=False, extra_body=None, **kwargs):
            super().__init__(model=model, cache=cache)
            self.api_key = api_key
            self.extra_body = extra_body
            self.init_kwargs = kwargs
            type(self).instances.append(self)

        def __call__(self, *args, **kwargs):
            type(self).context_lms.append(dspy.settings.lm)
            if sleep:
                time.sleep(sleep)
            content = (completions or {}).get(self.model)
            if content is None:
                content = (completions or {}).get("*", "")
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

    assert "[trace-miner] map: double" in stream.getvalue()
    assert "2/2" in stream.getvalue()
    lines = [line for chunk in stream.getvalue().split("\r") for line in chunk.splitlines() if line]
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

    assert "[trace-miner] reduce: merge" in stream.getvalue()
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
    result = run_config(loaded)

    assert read_jsonl(tmp_path / "output.jsonl") == [
        {"confidence": 0.8, "id": "a", "label": "ALPHA", "text": "alpha", "untouched": True}
    ]
    assert loaded.module.label.models == ["cheap-model"]
    assert result.stats.lm_calls == 1
    assert result.stats.lm_cost_usd == 0


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
    assert loaded.lm_cache is False

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
    assert lm.cache is False
    assert lm.extra_body == {"usage": {"include": True}}
    assert result.stats.elapsed_s > 0
    assert result.stats.lm_calls == 1
    assert result.stats.lm_prompt_tokens == 12
    assert result.stats.lm_completion_tokens == 4
    assert result.stats.lm_cost_usd == 0.0021


def test_lm_cache_can_be_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
  cache: true
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
    assert loaded.lm_cache is True
    run_config(loaded)
    assert FakeLM.instances[0].cache is True


def test_lm_cache_must_be_boolean(tmp_path: Path) -> None:
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
            model: cheap-model
      - map:
          stage: score
          lm:
            model: strong-model
""",
        [{"id": "a", "text": "ore"}, {"id": "b", "text": "coal"}],
    )
    FakeLM = install_fake_lm(
        monkeypatch,
        {
            "openrouter/cheap-model": chat_completion(label="ORE"),
            "openrouter/strong-model": chat_completion(score=0.9),
        },
        sleep=0.02,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    run_config(config)

    models = sorted(lm.model for lm in FakeLM.instances)
    assert models == ["openrouter/cheap-model", "openrouter/strong-model"]
    assert len({id(lm) for lm in FakeLM.instances}) == 2
    assert all(isinstance(observed, FakeLM) for observed in FakeLM.context_lms)
    assert {observed.model for observed in FakeLM.context_lms} == {
        "openrouter/cheap-model",
        "openrouter/strong-model",
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

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"label": "ORE"})}}]}).encode("utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("trace_miner.runtime.urlopen", lambda request, timeout: Response())

    result = run_config(config)

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


def test_parse_lm_json_repairs_python_style_escapes() -> None:
    from trace_miner.runtime import _parse_lm_json

    content = r'{"has_frustration": false, "frustration_kind": null, "evidence": "No module named \'Bio\'"}'
    assert _parse_lm_json(content) == {
        "has_frustration": False,
        "frustration_kind": None,
        "evidence": "No module named 'Bio'",
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

    monkeypatch.setattr("trace_miner.runtime._call_openrouter", fake_openrouter)

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
