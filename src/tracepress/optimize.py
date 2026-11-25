"""Tune one LM step's instructions against labeled rows, with GEPA: `table.optimize(labels, ...)`.

The step's input rows are read once. Each candidate prompt then reruns only that step, over
the labeled examples GEPA asks for, and is scored against the labels. GEPA reflects on the
misses ("expected True, got False") to propose better instructions.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import runtime, telemetry
from .adapters.base import RowsAdapter
from .lm import resolve_lm
from .rows import Row

if TYPE_CHECKING:
    from .plan import Table

COMPONENT = "instructions"
log = logging.getLogger("tracepress")


class _Log:
    """GEPA's progress lines go to the `tracepress` logger at debug level, not stdout."""

    def log(self, message: str) -> None:
        log.debug("gepa: %s", message)
Metric = str | Callable[[Row | None, dict[str, Any]], Any]


@dataclass
class OptimizeReport:
    """How tuning went: scores are the fraction of examples the step got right (or the metric's mean)."""

    before: str
    after: str
    train_before: float
    train_after: float
    val_before: float
    val_after: float
    candidates: int  # prompts GEPA kept (each beat the one it came from), the starting one included
    tried: int  # distinct prompts GEPA ran, rejected ones included
    lm_calls: int  # calls made running the step (cache hits excluded); GEPA's own reflection calls aren't counted
    cost_usd: float
    seconds: float
    history: list[tuple[str, float]] = field(default_factory=list)  # each kept prompt and its held-out score, in order

    def __str__(self) -> str:
        return (f"val {self.val_before:.2f} -> {self.val_after:.2f}, train {self.train_before:.2f} -> "
                f"{self.train_after:.2f}, kept {self.candidates} of {self.tried} prompts tried, "
                f"{self.lm_calls} LM calls, ${self.cost_usd:.4f}, {self.seconds:.0f}s")


@dataclass
class Scored:
    score: float
    output: dict[str, Any] | None  # None when the step produced no row for the example
    feedback: str
    inputs: dict[str, Any] = field(default_factory=dict)


class LabeledTask:
    """Runs one LM step with given instructions on chosen labeled examples, and scores it."""

    def __init__(self, table: Table, labels: list[dict[str, Any]], metric: Metric, max_concurrency: int | None):
        spec = table._node.spec
        self.spec, self.metric, self.max_concurrency = spec, metric, max_concurrency
        self.parent_schema = table._node.parent.schema
        self.keys = [c for c in table.schema.levels if c in labels[0]]
        if not self.keys or self.keys[-1] != table.schema.levels[-1]:
            raise ValueError(f"optimize: labels need a {table.schema.levels[-1]!r} column to match rows to labels "
                             f"(the table is keyed by {list(table.schema.levels)})")
        self.labels = labels
        by_key: dict[tuple, list[Row]] = {}
        for row in runtime.run([table._node.parent], max_concurrency)[0]:
            by_key.setdefault(self.key(row), []).append(row)
        self.inputs = [by_key.get(self.label_key(lab), []) for lab in labels]
        missing = [self.label_key(lab) for lab, rows in zip(labels, self.inputs) if not rows]
        if missing:
            raise ValueError(f"optimize: no rows for labeled {dict(zip(self.keys, missing[0]))}")
        self.calls, self.cost = 0, 0.0
        self.tried: set[str] = set()

    def key(self, row: Row) -> tuple:
        return tuple(row.get(k, None) for k in self.keys)

    def label_key(self, label: dict[str, Any]) -> tuple:
        return tuple(label[k] for k in self.keys)

    def evaluate(self, instructions: str, examples: list[int]) -> list[Scored]:
        self.tried.add(instructions)
        rows = [r for i in examples for r in self.inputs[i]]
        source = RowsAdapter(rows, self.parent_schema).table("rows")
        out = runtime.run([self.spec.remake(source, instructions)._node], self.max_concurrency)[0]
        stats = runtime.last_run
        self.calls += stats.calls - stats.cached_calls
        self.cost += stats.cost_usd
        got = {self.key(r): r for r in out}
        return [self.score(got.get(self.label_key(self.labels[i])), self.labels[i]) for i in examples]

    def score(self, row: Row | None, label: dict[str, Any]) -> Scored:
        output = {k: row.get(k, None) for k in self.spec.outputs} if row is not None else None
        if callable(self.metric):
            result = self.metric(row, label)
            score, feedback = result if isinstance(result, tuple) else (result, "")
            return Scored(float(score), output, feedback or f"Scored {float(score):.2f}.")
        want = label[self.metric]
        if row is None:
            return Scored(0.0, None, f"The step produced no output; expected {self.metric}={want!r}.")
        have = row.get(self.metric, None)
        ok = _same(have, want)
        return Scored(1.0 if ok else 0.0, output,
                      f"Correct: {self.metric}={want!r}." if ok else f"Expected {self.metric}={want!r}, got {have!r}.")

    def mean(self, instructions: str, examples: list[int]) -> float:
        scored = self.evaluate(instructions, examples) if examples else []
        return sum(s.score for s in scored) / len(scored) if scored else 0.0


def _same(have: Any, want: Any) -> bool:
    if isinstance(want, str) or isinstance(have, str):
        return str(have).strip().lower() == str(want).strip().lower()
    return have == want


class GEPAAdapter:
    """Presents a `LabeledTask` to GEPA: one component, the step's instructions."""

    propose_new_texts = None  # GEPA's own reflective proposer

    def __init__(self, task: LabeledTask):
        self.task = task

    def evaluate(self, batch: list[int], candidate: dict[str, str], capture_traces: bool = False):
        from gepa import EvaluationBatch

        scored = self.task.evaluate(candidate[COMPONENT], batch)
        return EvaluationBatch(outputs=[s.output for s in scored], scores=[s.score for s in scored],
                               trajectories=scored if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [{"Generated Outputs": s.output if s.output is not None else "(no output)", "Feedback": s.feedback}
                   for s in eval_batch.trajectories]
        return {COMPONENT: records}


def lm_text(lm) -> Callable[[str], str]:
    """A dspy.LM as the `prompt -> text` callable GEPA expects."""

    def call(prompt: str) -> str:
        out = lm(prompt)[0]
        return out["text"] if isinstance(out, dict) else out

    return call


def optimize(table: Table, labels, metric: Metric, *, optimizer: str = "gepa", budget: int = 60,
             val_frac: float = 0.3, seed: int = 0, reflection_lm=None,
             max_concurrency: int | None = None) -> Table:
    spec = table._node.spec
    if spec is None:
        raise ValueError("optimize: this table isn't an LM step (lm_map, lm_filter, lm_extract, lm_reduce or "
                         "lm_fold); call it on the step whose prompt you want to tune")
    if optimizer != "gepa":
        raise ValueError(f"optimize: optimizer={optimizer!r} isn't supported; use 'gepa'")
    records = labels.to_dict("records") if hasattr(labels, "to_dict") else [dict(r) for r in labels]
    if not records:
        raise ValueError("optimize: labels are empty")
    if isinstance(metric, str):
        if metric not in records[0]:
            raise ValueError(f"optimize: labels have no {metric!r} column")
        if metric not in spec.outputs:
            raise ValueError(f"optimize: {metric!r} isn't an output of this step (outputs: {list(spec.outputs)})")
    # The runs GEPA makes don't report one by one; the whole search sends one event.
    clock, report, error, n_val = time.monotonic(), None, None, 0
    try:
        with telemetry.quiet():
            task = LabeledTask(table, records, metric, max_concurrency)

            ids = list(range(len(records)))
            random.Random(seed).shuffle(ids)
            n_val = max(1, round(len(ids) * val_frac)) if len(ids) > 1 else 0
            val, train = sorted(ids[:n_val]), sorted(ids[n_val:])

            import gepa

            start = time.monotonic()
            result = gepa.optimize(
                seed_candidate={COMPONENT: spec.instructions},
                trainset=train,
                valset=val or train,
                adapter=GEPAAdapter(task),
                reflection_lm=lm_text(resolve_lm(reflection_lm)),
                max_metric_calls=budget,
                seed=seed,
                display_progress_bar=False,
                logger=_Log(),
            )
            best = result.best_candidate[COMPONENT]
            seconds = time.monotonic() - start
            history = [(c[COMPONENT], float(v)) for c, v in zip(result.candidates, result.val_aggregate_scores)]
            report = OptimizeReport(
                before=spec.instructions, after=best,
                train_before=task.mean(spec.instructions, train), train_after=task.mean(best, train),
                val_before=task.mean(spec.instructions, val), val_after=task.mean(best, val),
                candidates=len(result.candidates), tried=len(task.tried), lm_calls=task.calls, cost_usd=task.cost,
                seconds=seconds, history=history)
    except BaseException as e:
        error = e
        raise
    finally:
        telemetry.report_optimize(
            optimizer=optimizer, step_op=spec.op, metric="column" if isinstance(metric, str) else "callable",
            n_labels=len(records), n_val=n_val, budget=budget, result=report,
            seconds=time.monotonic() - clock, error=error)
    from .plan import Table

    tuned = spec.remake(Table(table._node.parent), best)
    tuned.optimize_report = report
    return tuned
