"""The workspace: registered pipelines and every artifact their runs produced.

    ws = tracepress.Workspace("ws")

    @ws.pipeline(description="Why each run failed, rolled up by task")
    def failure_modes(steps):
        reasons = steps.lm_agg("summary -> reason", "Why did this run fail?", by="session_id", save_as="run_reasons")
        return reasons.lm_agg("reason -> modes", "Common failure modes", by="task_name", save_as="task_modes")

    failure_modes(steps_df)          # runs it; every lm_* output is saved
    ws["task_modes"]                 # the latest artifact with that name

On disk, each run is a folder: ``<ws>/runs/<run_id>/manifest.json`` plus one parquet file
per artifact. The manifest records, for every artifact, the op and instruction that made
it, the model, its parent artifacts, and its shape. Artifacts are written as soon as
they are produced, so a run that fails midway keeps what it finished.

Pipelines are Python functions, so they're registered again each time the notebook cell
defining them runs; their artifacts persist on disk.
"""

from __future__ import annotations

import inspect
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .run import ATTR, _current, artifact_of

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe(name: str) -> str:
    return _SAFE.sub("_", name).strip("_") or "artifact"


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df``; columns pyarrow can't store as-is (mixed or nested values) become JSON text."""
    try:
        df.to_parquet(path)
        return
    except Exception:
        pass
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            safe[col] = [v if v is None or isinstance(v, str) else json.dumps(v, default=str) for v in safe[col]]
    safe.to_parquet(path)


@dataclass
class Pipeline:
    """A registered pipeline. Calling it runs it in its workspace and returns its output."""

    fn: Callable[..., Any]
    name: str
    description: str
    workspace: "Workspace"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.workspace.run(self.name, *args, **kwargs).output

    @property
    def signature(self) -> str:
        """The function's name and parameters, e.g. ``failure_modes(runs)``."""
        return f"{self.name}{inspect.signature(self.fn)}"

    def info(self) -> dict[str, Any]:
        """Name, signature, description, and docstring."""
        return {
            "name": self.name,
            "signature": self.signature,
            "description": self.description,
            "doc": inspect.getdoc(self.fn) or "",
        }


class Run:
    """One execution of a pipeline (or a group of ad-hoc steps) and the artifacts it wrote."""

    def __init__(self, workspace: "Workspace", pipeline: str, params: dict[str, Any] | None = None):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{_safe(pipeline)}"
        run_id, n = base, 1
        while (workspace.path / "runs" / run_id).exists():
            n += 1
            run_id = f"{base}-{n}"
        self.workspace = workspace
        self.id = run_id
        self.dir = workspace.path / "runs" / run_id
        self.dir.mkdir(parents=True)
        self.output: Any = None
        self.manifest: dict[str, Any] = {
            "run_id": run_id,
            "pipeline": pipeline,
            "params": params or {},
            "status": "running",
            "started": _now(),
            "finished": None,
            "error": None,
            "output": None,
            "artifacts": [],
        }
        self._step = 0
        self._frames: dict[str, pd.DataFrame] = {}  # what was saved, to tell it from derived frames
        self._write_manifest()

    def __repr__(self) -> str:
        names = [a["name"] for a in self.manifest["artifacts"]]
        return f"Run({self.id!r}, status={self.manifest['status']!r}, artifacts={names})"

    def _write_manifest(self) -> None:
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))

    def save(
        self,
        df: pd.DataFrame,
        *,
        name: str | None = None,
        op: str = "input",
        parents: list[str] | None = None,
        instruction: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> str:
        """Write ``df`` as an artifact of this run and stamp it; returns ``run_id/name``."""
        self._step += 1
        name = _safe(name or f"step{self._step}_{op}")
        taken = {a["name"] for a in self.manifest["artifacts"]}
        base, n = name, 1
        while name in taken:
            n += 1
            name = f"{base}_{n}"
        _write_parquet(df, self.dir / f"{name}.parquet")
        self.manifest["artifacts"].append(
            {
                "name": name,
                "op": op,
                "instruction": instruction,
                "model": model,
                "parents": parents or [],
                "columns": [str(c) for c in df.columns],
                "n_rows": len(df),
                "usage": usage,
                "params": params,
                "description": description,
                "created": _now(),
            }
        )
        self._write_manifest()
        ref = f"{self.id}/{name}"
        df.attrs[ATTR] = ref
        self._frames[ref] = df
        return ref

    def finish(self, output: Any = None, error: BaseException | None = None) -> None:
        """Mark the run done (or failed) and record which artifacts it returned."""
        self.output = output
        self.manifest["finished"] = _now()
        if error is not None:
            self.manifest["status"] = "error"
            self.manifest["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        else:
            self.manifest["status"] = "ok"
            self.manifest["output"] = self._output_refs(output)
        self._write_manifest()

    def _output_refs(self, output: Any) -> Any:
        """Names of the artifacts a pipeline returned, saving any the ops didn't already save."""
        if isinstance(output, pd.DataFrame):
            ref = artifact_of(output)
            if ref and self._frames.get(ref) is output:
                return ref.split("/", 1)[1]
            return self.save(output, name="output", op="output", parents=[ref] if ref else []).split("/", 1)[1]
        if isinstance(output, dict):
            return {k: self._output_refs(v) for k, v in output.items() if isinstance(v, pd.DataFrame)}
        if isinstance(output, (list, tuple)):
            return [self._output_refs(v) for v in output if isinstance(v, pd.DataFrame)]
        return None

    def __enter__(self) -> "Run":
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _current.reset(self._token)
        if self.manifest["status"] == "running":
            self.finish(error=exc)


class Workspace:
    """A folder of runs and artifacts, plus the pipelines registered in this session.

    Args:
        path: where runs are stored; created if missing. On Colab, a mounted Drive path
            keeps artifacts across sessions.
    """

    def __init__(self, path: str | Path = "workspace"):
        self.path = Path(path).expanduser()
        (self.path / "runs").mkdir(parents=True, exist_ok=True)
        self.pipelines: dict[str, Pipeline] = {}

    def __repr__(self) -> str:
        return f"Workspace({str(self.path)!r}, pipelines={list(self.pipelines)}, runs={len(self._manifests())})"

    # ----------------------------------------------------------------- pipelines

    def pipeline(self, fn: Callable | None = None, *, name: str | None = None, description: str | None = None):
        """Register a function as a pipeline. Use as ``@ws.pipeline`` or ``@ws.pipeline(description=...)``.

        The function takes DataFrames (and any other parameters) and runs ``df.lm_*`` operators.
        The description says what the pipeline is for, to whoever reads the workspace later.
        """

        def register(f: Callable) -> Pipeline:
            if getattr(f, "_tracepress_pipeline", True) is False:
                raise TypeError(
                    f"{f.__name__} reads files and is not a workspace pipeline. "
                    "Call it yourself, then pass the loaded data into a pipeline."
                )
            p = Pipeline(f, name or f.__name__, description or inspect.getdoc(f) or "", self)
            self.pipelines[p.name] = p
            return p

        return register(fn) if fn is not None else register

    def run(self, pipeline: str | Pipeline, *args: Any, **kwargs: Any) -> Run:
        """Run a registered pipeline. DataFrame arguments are saved as input artifacts first."""
        p = pipeline if isinstance(pipeline, Pipeline) else self.pipelines[pipeline]
        bound = inspect.signature(p.fn).bind(*args, **kwargs)
        params = {k: (artifact_of(v) or f"<DataFrame {v.shape}>") if isinstance(v, pd.DataFrame) else repr(v)
                  for k, v in bound.arguments.items()}
        with Run(self, p.name, params) as run:
            for key, value in bound.arguments.items():
                if isinstance(value, pd.DataFrame):
                    parent = artifact_of(value)
                    value = value.copy()
                    run.save(value, name=key, op="input", parents=[parent] if parent else [])
                    bound.arguments[key] = value
            try:
                output = p.fn(*bound.args, **bound.kwargs)
            except BaseException as exc:
                run.finish(error=exc)
                raise
            run.finish(output)
        return run

    def save(self, name: str, df: pd.DataFrame, description: str | None = None) -> pd.DataFrame:
        """Add a DataFrame to the workspace as a standalone artifact (e.g. raw data)."""
        with Run(self, f"save-{name}") as run:
            parent = artifact_of(df)
            run.save(df, name=name, op="input", parents=[parent] if parent else [], description=description)
            run.finish(df)
        return df

    # ----------------------------------------------------------------- artifacts

    def _manifests(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted((self.path / "runs").glob("*/manifest.json")):
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def runs(self) -> pd.DataFrame:
        """One row per run, oldest first."""
        rows = [
            {k: m.get(k) for k in ("run_id", "pipeline", "status", "started", "finished", "output", "error")}
            | {"n_artifacts": len(m["artifacts"])}
            for m in self._manifests()
        ]
        return pd.DataFrame(rows, columns=["run_id", "pipeline", "status", "started", "finished", "output", "error", "n_artifacts"])

    def artifact_records(self) -> list[dict[str, Any]]:
        """Every artifact's manifest entry, with ``run_id``, ``pipeline`` and ``ref`` added. Oldest first."""
        out = []
        for m in self._manifests():
            for a in m["artifacts"]:
                out.append({**a, "run_id": m["run_id"], "pipeline": m["pipeline"], "ref": f"{m['run_id']}/{a['name']}"})
        return out

    def artifacts(self, latest: bool = False) -> pd.DataFrame:
        """A table of artifacts. ``latest=True`` keeps only the newest per name."""
        records = self.artifact_records()
        if latest:
            records = list({r["name"]: r for r in records}.values())
        cols = ["ref", "name", "pipeline", "op", "n_rows", "columns", "parents", "instruction", "model", "created"]
        return pd.DataFrame([{c: r.get(c) for c in cols} for r in records], columns=cols)

    def resolve(self, name: str) -> dict[str, Any]:
        """The manifest entry for ``name`` (latest with that name) or ``run_id/name``."""
        records = self.artifact_records()
        matches = [r for r in records if r["ref"] == name] if "/" in name else [r for r in records if r["name"] == name]
        if not matches:
            known = sorted({r["name"] for r in records})
            raise KeyError(f"no artifact {name!r}; known names: {known}")
        return matches[-1]

    def load(self, name: str) -> pd.DataFrame:
        """Load an artifact by name (the latest one) or by ``run_id/name``."""
        rec = self.resolve(name)
        df = pd.read_parquet(self.path / "runs" / rec["run_id"] / f"{rec['name']}.parquet")
        df.attrs[ATTR] = rec["ref"]
        return df

    __getitem__ = load

    def lineage(self, name: str) -> list[dict[str, Any]]:
        """The artifact and all its ancestors, the artifact first."""
        by_ref = {r["ref"]: r for r in self.artifact_records()}
        out, queue, seen = [], [self.resolve(name)["ref"]], set()
        while queue:
            ref = queue.pop(0)
            if ref in seen or ref not in by_ref:
                continue
            seen.add(ref)
            rec = by_ref[ref]
            out.append({k: rec.get(k) for k in ("ref", "op", "instruction", "model", "n_rows", "parents")})
            queue.extend(rec["parents"])
        return out
