from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal


StepKind = Literal["map", "filter", "stage", "tee", "group"]
GroupKind = Literal["expand", "by", "every"]
StageKind = Literal["python", "lm"]
ChildErrorPolicy = Literal["fail", "partial", "skip"]
IncompletePolicy = Literal["fail", "partial"]

DEFAULT_LM_PROVIDER = "openrouter"
DEFAULT_LM_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_LM_MAX_CONCURRENT = 8


@dataclass(frozen=True)
class LmField:
    name: str
    annotation: Any = None


@dataclass(frozen=True)
class LmSignature:
    input_fields: tuple[LmField, ...]
    output_fields: tuple[LmField, ...]


@dataclass(frozen=True)
class StageRef:
    name: str
    target: Callable[..., Any] | Any
    kind: StageKind
    role: str
    accepts_ctx: bool = False
    is_async: bool = False
    lm_signature: LmSignature | None = None
    lm_model: str | None = None
    lm_provider: str | None = None


@dataclass(frozen=True)
class Defaults:
    on_child_error: ChildErrorPolicy = "fail"
    on_group_incomplete: IncompletePolicy = "fail"
    ordered: bool = True


@dataclass(frozen=True)
class Step:
    kind: StepKind
    stage: StageRef | None = None
    path: Path | None = None
    group: GroupSpec | None = None


@dataclass(frozen=True)
class GroupSpec:
    kind: GroupKind
    expand: StageRef | None
    by: str | StageRef | None
    every: int | None
    pipeline: tuple[Step, ...]
    reduce: StageRef
    on_child_error: ChildErrorPolicy
    on_group_incomplete: IncompletePolicy
    ordered: bool


@dataclass(frozen=True)
class PipelineSpec:
    inputs: tuple[Path, ...]
    output: Path
    steps: tuple[Step, ...]
    source_index: int


@dataclass(frozen=True)
class Config:
    config_path: Path
    base_dir: Path
    module: ModuleType
    defaults: Defaults
    id_field: str | None
    lm_provider: str
    lm_default_model: str
    lm_max_concurrent: int
    pipelines: tuple[PipelineSpec, ...]


@dataclass
class Failure:
    stage: str
    record_identity: str
    group_id: str
    error: str


@dataclass
class RunManifest:
    skipped_groups: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunStats:
    records_read: int = 0
    emitted: int = 0
    retired: int = 0
    dropped: int = 0
    failed: int = 0
    groups_opened: int = 0
    groups_reduced: int = 0
    groups_skipped: int = 0


@dataclass
class RunResult:
    stats: RunStats
    manifest: RunManifest
