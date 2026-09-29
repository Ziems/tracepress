"""The agent's tools: read the workspace, compute over it, and run more LLM steps on it.

Every tool returns text. Reads are free; ``run_op`` and ``run_pipeline`` call models, and
the active ``tracepress.model.budget`` refuses them before they start if they would overrun it.
Anything they produce is saved to the workspace like any other artifact, under a run
named ``agent_run`` (for ops) or under the pipeline's own run.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from dspy.lm15 import FunctionTool

from ..model import BudgetExceeded
from ..ops import OPS
from ..run import _current
from ..workspace import Run, Workspace

MAX_OUTPUT = 20_000


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[output truncated: {len(text) - limit} more characters]"


def _cell(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f" …[+{len(value) - limit} chars]"
    if isinstance(value, (list, tuple)):
        return [_cell(v, limit) for v in value]
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return _cell(value.tolist(), limit)
    return value


def frame_json(df: pd.DataFrame, cell_limit: int = 2000) -> str:
    """Rows as JSON records, each with its index label under ``_row``."""
    records = []
    for label, row in zip(df.index, df.to_dict(orient="records")):
        records.append({"_row": _cell(label, 100), **{str(k): _cell(v, cell_limit) for k, v in row.items()}})
    return json.dumps(records, ensure_ascii=False, default=str, indent=1)


def artifact_summary(rec: dict[str, Any], instruction_limit: int = 300) -> dict[str, Any]:
    return {
        "name": rec["name"],
        "ref": rec["ref"],
        "pipeline": rec["pipeline"],
        "op": rec["op"],
        "n_rows": rec["n_rows"],
        "columns": rec["columns"],
        "parents": rec["parents"],
        "instruction": _cell(rec.get("instruction"), instruction_limit),
        **({"description": rec["description"]} if rec.get("description") else {}),
    }


@dataclass
class Tool:
    spec: FunctionTool
    fn: Callable[..., str]


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


class Session:
    """The state one ``agent_run`` works in: the workspace, a Python namespace, and its own run."""

    def __init__(self, workspace: Workspace, question: str):
        self.ws = workspace
        self.question = question
        self._run: Run | None = None
        self.namespace: dict[str, Any] = {"pd": pd, "ws": workspace, "load": workspace.load}

    @property
    def run(self) -> Run:
        """The run that ad-hoc outputs are saved under, created on first use."""
        if self._run is None:
            self._run = Run(self.ws, "agent_run", {"question": self.question})
        return self._run

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()

    @contextlib.contextmanager
    def recording(self):
        token = _current.set(self.run)
        try:
            yield
        finally:
            _current.reset(token)

    # ------------------------------------------------------------------- tools

    def list_pipelines(self) -> str:
        if not self.ws.pipelines:
            return "No pipelines are registered in this session."
        return json.dumps([p.info() for p in self.ws.pipelines.values()], indent=1)

    def list_artifacts(self, all_versions: bool = False) -> str:
        records = self.ws.artifact_records()
        if not all_versions:
            records = list({r["name"]: r for r in records}.values())
        if not records:
            return "The workspace has no artifacts yet."
        return json.dumps([artifact_summary(r) for r in records], indent=1, default=str)

    def describe_artifact(self, name: str, samples: int = 3) -> str:
        rec = self.ws.resolve(name)
        df = self.ws.load(name)
        columns = []
        for col in df.columns:
            s = df[col]
            info: dict[str, Any] = {"name": str(col), "dtype": str(s.dtype), "non_null": int(s.notna().sum())}
            try:
                uniq = s.nunique(dropna=True)
                info["n_unique"] = int(uniq)
                if uniq <= 20:
                    info["value_counts"] = {str(k): int(v) for k, v in s.value_counts().head(20).items()}
            except TypeError:  # unhashable values (lists)
                pass
            if s.dtype == object or str(s.dtype).startswith("str"):
                lengths = s.dropna().astype(str).str.len()
                if len(lengths):
                    info["mean_chars"] = int(lengths.mean())
            columns.append(info)
        out = {
            **artifact_summary(rec, instruction_limit=4000),
            "model": rec.get("model"),
            "params": rec.get("params"),
            "created": rec.get("created"),
            "column_stats": columns,
            "sample_rows": json.loads(frame_json(df.head(samples), cell_limit=400)),
        }
        return json.dumps(out, indent=1, default=str)

    def read_rows(
        self,
        name: str,
        columns: list[str] | None = None,
        where: str | None = None,
        offset: int = 0,
        limit: int = 20,
        max_chars_per_cell: int = 2000,
    ) -> str:
        df = self.ws.load(name)
        if where:
            df = df.query(where)
        total = len(df)
        if columns:
            df = df[columns]
        page = df.iloc[offset : offset + min(limit, 200)]
        header = f"{name}: rows {offset}–{offset + len(page) - 1} of {total}" + (f" matching {where!r}" if where else "")
        return header + "\n" + frame_json(page, max_chars_per_cell)

    def python(self, code: str) -> str:
        """Run code in a persistent namespace; returns stdout and the last expression's value."""
        buffer = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(buffer), self.recording():
                exec(compile(tree, "<agent>", "exec"), self.namespace)
                if last is not None:
                    value = eval(compile(ast.Expression(last.value), "<agent>", "eval"), self.namespace)
                    if value is not None:
                        with pd.option_context("display.max_rows", 60, "display.max_columns", 30, "display.width", 200):
                            print(repr(value))
        except BudgetExceeded:
            raise
        except Exception:
            lines = traceback.format_exc().strip().splitlines()
            print("\n".join(lines[-12:]), file=buffer)
        return buffer.getvalue() or "(no output)"

    def run_op(
        self,
        op: str,
        artifact: str,
        signature: str,
        instruction: str | None = None,
        by: list[str] | None = None,
        k: int | None = None,
        where: str | None = None,
        limit: int | None = None,
        save_as: str | None = None,
    ) -> str:
        if op not in OPS:
            return f"Unknown op {op!r}; choose one of {sorted(OPS)}."
        df = self.ws.load(artifact)
        if where:
            df = df.query(where)
        if limit:
            df = df.head(limit)
        kwargs: dict[str, Any] = {"save_as": save_as}
        if op == "agg" and by:
            kwargs["by"] = by
        if op == "topk":
            if not k:
                return "topk needs `k`."
            kwargs["k"] = k
        if op in ("filter", "topk") and not instruction:
            return f"{op} needs an `instruction` (the condition or criterion)."
        with self.recording():
            result = OPS[op](df, signature, instruction, **kwargs)
        ref = result.attrs.get("tracepress_artifact")
        return (
            f"Saved as artifact {ref} ({len(result)} rows from {len(df)} input rows; "
            f"columns {[str(c) for c in result.columns]}). First rows:\n{frame_json(result.head(5), 1500)}"
        )

    def run_pipeline(
        self,
        pipeline: str,
        inputs: dict[str, str],
        where: dict[str, str] | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        if pipeline not in self.ws.pipelines:
            return f"No pipeline {pipeline!r}; registered: {sorted(self.ws.pipelines)}."
        kwargs: dict[str, Any] = dict(params or {})
        for param, artifact in inputs.items():
            df = self.ws.load(artifact)
            if where and param in where:
                df = df.query(where[param])
            if limit:
                df = df.head(limit)
            kwargs[param] = df
        run = self.ws.run(pipeline, **kwargs)
        arts = [{"name": a["name"], "op": a["op"], "n_rows": a["n_rows"]} for a in run.manifest["artifacts"]]
        return json.dumps({"run_id": run.id, "status": run.manifest["status"], "output": run.manifest["output"], "artifacts": arts}, indent=1)

    def tools(self) -> list[Tool]:
        s = _schema
        string = {"type": "string"}
        return [
            Tool(
                FunctionTool(
                    "list_pipelines",
                    "List the registered pipelines: name, signature, and what each is for.",
                    s({}, []),
                ),
                self.list_pipelines,
            ),
            Tool(
                FunctionTool(
                    "list_artifacts",
                    "List the workspace's artifacts (latest version of each name, unless all_versions): "
                    "the op and instruction that produced each, its parents, row count, and columns.",
                    s({"all_versions": {"type": "boolean"}}, []),
                ),
                self.list_artifacts,
            ),
            Tool(
                FunctionTool(
                    "describe_artifact",
                    "Describe one artifact: full producing instruction, model, per-column stats "
                    "(value counts for low-cardinality columns), and sample rows. `name` is a name or run_id/name.",
                    s({"name": string, "samples": {"type": "integer"}}, ["name"]),
                ),
                self.describe_artifact,
            ),
            Tool(
                FunctionTool(
                    "read_rows",
                    "Read rows of an artifact as JSON, with each row's id under _row. `where` is a pandas "
                    "query string (e.g. \"task_name == 'x'\"). At most 200 rows per call.",
                    s(
                        {
                            "name": string,
                            "columns": {"type": "array", "items": string},
                            "where": string,
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                            "max_chars_per_cell": {"type": "integer"},
                        },
                        ["name"],
                    ),
                ),
                self.read_rows,
            ),
            Tool(
                FunctionTool(
                    "python",
                    "Run Python in a persistent namespace with pandas as pd. load('name') returns an "
                    "artifact as a DataFrame. Prints stdout and the value of the last expression. Use "
                    "this for exact counts, filters, joins, and grouping.",
                    s({"code": string}, ["code"]),
                ),
                self.python,
            ),
            Tool(
                FunctionTool(
                    "run_op",
                    "Run one LLM operator over an artifact (optionally a subset) and save the result as a "
                    "new artifact. Costs model calls: one per row for map/filter, fewer for agg/topk/dedup.\n"
                    "`signature` names the input columns and, for map and agg, the output columns: "
                    "'product, review -> has_defect: bool, part'. Output types default to str; "
                    "bool, int, float, list[str], and Literal['a', 'b'] also work. filter, topk, and dedup "
                    "take only inputs: 'product, review'. `instruction` is plain prose, with no column "
                    "templating.\n"
                    "Ops: map (per-row outputs), filter (instruction = condition; keeps matching rows), "
                    "agg (one answer, or one per `by` group), topk (instruction = criterion, plus k), "
                    "dedup (instruction = what counts as a duplicate). Try a small `limit` first.",
                    s(
                        {
                            "op": {"type": "string", "enum": sorted(OPS)},
                            "artifact": string,
                            "signature": string,
                            "instruction": string,
                            "by": {"type": "array", "items": string},
                            "k": {"type": "integer"},
                            "where": string,
                            "limit": {"type": "integer"},
                            "save_as": string,
                        },
                        ["op", "artifact", "signature"],
                    ),
                ),
                self.run_op,
            ),
            Tool(
                FunctionTool(
                    "run_pipeline",
                    "Run a registered pipeline. `inputs` maps each DataFrame parameter to an artifact "
                    "name; `where` optionally maps a parameter to a pandas query that subsets it first; "
                    "`limit` keeps the first N rows of each input; `params` passes other arguments. "
                    "Costs model calls. Returns the new run's artifacts.",
                    s(
                        {
                            "pipeline": string,
                            "inputs": {"type": "object", "additionalProperties": string},
                            "where": {"type": "object", "additionalProperties": string},
                            "limit": {"type": "integer"},
                            "params": {"type": "object"},
                        },
                        ["pipeline", "inputs"],
                    ),
                ),
                self.run_pipeline,
            ),
        ]

    def dispatch(self, tools: dict[str, Tool], name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Call a tool; returns (output, is_error). Errors go back to the model as text."""
        tool = tools.get(name)
        if tool is None:
            return f"Unknown tool {name!r}.", True
        try:
            return _clip(tool.fn(**(arguments or {}))), False
        except BudgetExceeded as exc:
            return f"Refused by the budget: {exc}. Answer with what you have, or use a smaller subset.", True
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", True
