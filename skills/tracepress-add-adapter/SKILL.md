---
name: tracepress-add-adapter
description: Add support for a new agent-trace format or source to TracePress by writing an adapter (e.g. DSPy, ChatGPT exports, Inspect AI logs, another coding agent's sessions, or a custom log format). Use when asked to load traces TracePress can't read yet, implement one of its placeholder readers, or turn some trace data into TracePress tables.
---

# Writing a TracePress adapter

An adapter turns a trace source into named tables of rows, and says when each group of rows is
complete. The engine knows nothing about specific formats. Hierarchy, tables and closure are all
the adapter's choice. Existing adapters to copy from:
- `src/tracepress/adapters/atif.py`: ATIF trajectories (simple, two levels)
- `src/tracepress/adapters/harbor.py`: Harbor jobs (four levels, several tables)
- `src/tracepress/adapters/sessions.py` + `claude_code.py` / `codex.py`: one `.jsonl` per
  session, a shared base class, steps in the ATIF step columns
- `src/tracepress/adapters/otel.py`: a variable-depth span tree, kept flat (`spans`) and read
  as steps (`steps`); `otlp.py` is its live version (`bounded = False`)
- `src/tracepress/adapters/base.py`: the `Adapter` base class and `RowsAdapter`

## When to use

- A user has traces in a format with no reader. `tp.read.dspy` and `tp.read.chatgpt` are
  placeholders that raise `NotImplementedError`
  (see `adapters/placeholders.py` and "Planned adapters" in `docs/SPEC.md`).
- Turning a one-off log format into TracePress tables

## Steps

1. **Look at real files first.** Find the natural hierarchy (e.g. session > message,
   trace > span), the timestamp fields, where outcomes live, and how big the files are. Look for
   images or other large blobs; drop them (see `clean_content` in `atif.py`).

2. **Choose levels and tables.** Levels are names, outermost first. Each table's grain is a prefix
   of those levels, plus at most one table-specific level (as `tool_calls` adds `call`). Prefer:
   - a top-level table with one row per run, holding outcome and metadata
   - a step or message table, with small run-level fields (agent, model) copied onto each row so
     they can be grouped on directly
   - columns named after the optional conventions (`source`, `message`, `tool_calls`, `error`,
     `timestamp`, `reward`), so existing pipelines carry over. Never name a column after a level.
   - the context that explains a run, wherever the format has it: the task or user prompt (as a
     `source="user"` step), the outcome (reward, pass or fail, error), and any ground truth such as
     test or grader results (as its own table, like Harbor's `tests`). Also keep the agent's
     reasoning (thinking or chain-of-thought blocks) in a `reasoning` column on its step:
     `STEP_INPUTS` reads it. Summaries are only as specific as their inputs, so don't drop these
     even when they're awkward to parse.

   For trees of variable depth (OpenTelemetry spans, branching chats), mapping depth onto keys is an
   open design question in `docs/SPEC.md`. A flat table with `parent_id` plus a depth column, and a
   derived table in the step convention, is what the OpenTelemetry adapter does; copy it.

3. **Implement the adapter.**

   ```python
   from collections.abc import AsyncIterator
   from tracepress.adapters.base import Adapter
   from tracepress.rows import Closed, Event, Row, RowAdded
   from tracepress.schema import Schema

   class MyAdapter(Adapter):
       levels = ("session", "message")
       bounded = True                      # False for a live source

       def __init__(self, sessions):       # parsed or indexed input
           self.sessions = sessions

       def tables(self) -> dict[str, Schema]:
           return {
               "sessions": Schema(("session",), {"agent": str, "succeeded": bool}),
               "messages": Schema(("session", "message"), {"agent": str, "role": str, "content": str, "timestamp": str}),
           }

       async def events(self, table: str) -> AsyncIterator[Event]:
           for s in self.sessions:
               key = (("session", s["id"]),)
               if table == "sessions":
                   yield RowAdded(Row(key, {"agent": s["agent"], "succeeded": s["ok"]}))
               else:
                   for i, m in enumerate(s["messages"]):
                       yield RowAdded(Row((*key, ("message", i)),
                                          {"agent": s["agent"], "role": m["role"],
                                           "content": m["text"], "timestamp": m["ts"]}))
               yield Closed(key)           # this session is complete
   ```

   **Closure rules.** The engine depends on these; break them and reduces silently go wrong.
   - `Closed(prefix)` means no more rows will ever arrive under `prefix`. Emit it **after** the
     last row under that prefix, never before.
   - Close every group you can: each run, and each parent level once all its children are done
     (deepest first). Groups never closed only finish at the end of the stream.
   - Keys must be unique within a table, and ids within a level must sort consistently
     (all ints or all strings).
   - Read big files lazily, per run, with `await asyncio.to_thread(...)`, not all up front.
   - Give rows a `timestamp` (ISO string), `finished_at` or `started_at` column, so `tp.replay`
     can order them realistically.

4. **Add a reader function and register it.** Put the reader (e.g. `read_myformat(path, ...) ->
   MyAdapter`, raising `FileNotFoundError` when nothing is found) in
   `src/tracepress/adapters/<name>.py`. Export it from `adapters/__init__.py`, and expose it in
   `src/tracepress/read.py` (e.g. `from .adapters import read_myformat as myformat`). If it replaces
   a placeholder, delete that entry from `placeholders.py` and update its test in `tests/test_harbor.py`.

5. **Test it.** Copy `tests/conftest.py::harbor_dir` and `tests/test_harbor.py`:
   - build a tiny fixture of synthetic files in `tmp_path` (never commit real user traces), covering
     a normal run, a crashed or empty run, and an image or blob
   - collect each table and assert keys, values and row counts
   - check closure by replay-vs-batch equivalence, which catches closes emitted too early:
     ```python
     src = read_myformat(fixture)
     batch = [(r.key, r.values) for r in src.table("messages").group_by("session").reduce(lambda rs: {"n": len(rs)}).collect()]
     for seed in range(3):
         live = tp.replay(src, seed=seed, shuffle_within=4, duplicates=0.3)
         assert [(r.key, r.values) for r in live.table("messages").group_by("session").reduce(lambda rs: {"n": len(rs)}).collect()] == batch
     ```
   - run `uv run pytest`

6. **Document it.** Add the adapter to the "Planned adapters" table in `docs/SPEC.md` (status v1), and
   add a line to the `README.md` if users will call it directly.
