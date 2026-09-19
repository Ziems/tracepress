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
DEFAULT_LM_MAX_CONCURRENT = 64


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


@dataclass(frozen=True)
class LmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


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
    elapsed_s: float = 0.0
    lm_calls: int = 0
    lm_prompt_tokens: int = 0
    lm_completion_tokens: int = 0
    lm_cost_usd: float = 0.0

    def record_lm(self, usage: LmUsage | None = None) -> None:
        self.lm_calls += 1
        if usage is None:
            return
        self.lm_prompt_tokens += usage.prompt_tokens
        self.lm_completion_tokens += usage.completion_tokens
        self.lm_cost_usd += usage.cost_usd


@dataclass
class RunResult:
    stats: RunStats
    manifest: RunManifest


def format_run_profile(stats: RunStats) -> str:
    elapsed = _format_elapsed(stats.elapsed_s)
    if stats.lm_calls == 0:
        return elapsed
    tokens = stats.lm_prompt_tokens + stats.lm_completion_tokens
    return f"{elapsed}  {stats.lm_calls} LM calls  {tokens:,} tokens  ${stats.lm_cost_usd:.4f}"


def _format_elapsed(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.2f}s"
    return f"{seconds:.1f}s"
