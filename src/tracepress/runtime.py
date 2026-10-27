"""Runs a plan DAG: one asyncio task per operator, connected by queues.

Batch and streaming share this runtime. A bounded source simply ends.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from . import telemetry
from .lm import Executor, RunStats
from .operators import END, Operator, supersedes
from .rows import Event, Row, RowAdded, RowUpdated, sort_key

if TYPE_CHECKING:
    from .plan import Node

last_run: RunStats | None = None


class SourceOp(Operator):
    def __init__(self, adapter, table: str):
        super().__init__()
        self.adapter, self.table = adapter, table

    async def produce(self) -> None:
        async for event in self.adapter.events(self.table):
            self.emit(event)
        self.emit(END)


class CollectSink:
    """Keeps the newest final version of each row."""

    def __init__(self) -> None:
        self.rows: dict = {}

    async def consume(self, queue: asyncio.Queue) -> None:
        while (event := await queue.get()) is not END:
            if isinstance(event, (RowAdded, RowUpdated)) and event.row.final:
                if supersedes(event.row, self.rows.get(event.row.key)):
                    self.rows[event.row.key] = event.row

    def result(self) -> list[Row]:
        return sorted(self.rows.values(), key=lambda r: sort_key(r.key))


class QueueSink:
    """Passes every event through, for `Table.events()`."""

    def __init__(self) -> None:
        self.out: asyncio.Queue = asyncio.Queue()

    async def consume(self, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            self.out.put_nowait(event)
            if event is END:
                return

    def result(self) -> None:
        return None


async def run_async(nodes: list[Node], max_concurrency: int | None = None, sinks: list | None = None) -> list:
    global last_run
    executor = Executor(max_concurrency)
    ops: dict[int, Operator] = {}
    coros = []

    def build(node: Node) -> tuple[Operator, bool]:
        """Returns the operator and whether any source upstream of it is live."""
        if id(node) in ops:
            op = ops[id(node)]
            return op, op.live
        op = node.make()
        op.executor = executor
        op.label = node.label
        ops[id(node)] = op
        if node.parent is None:
            op.live = not op.adapter.bounded
            coros.append(op.produce())
        else:
            queue: asyncio.Queue = asyncio.Queue()
            parent, live = build(node.parent)
            parent.outputs.append(queue)
            op.live = live
            coros.append(op.consume(queue))
        return op, op.live

    sinks = sinks or [CollectSink() for _ in nodes]
    for node, sink in zip(nodes, sinks):
        queue: asyncio.Queue = asyncio.Queue()
        build(node)[0].outputs.append(queue)
        coros.append(sink.consume(queue))

    tasks = [asyncio.ensure_future(c) for c in coros]
    # Operators fail this to stop the run at once, e.g. on an error every call will hit.
    executor.stop = stop = asyncio.get_running_loop().create_future()
    work = asyncio.gather(*tasks)  # fails with the first task that fails
    work.add_done_callback(lambda f: f.cancelled() or f.exception())  # read, so a stopped run doesn't warn
    error: BaseException | None = None
    try:
        done, _ = await asyncio.wait([work, stop], return_when=asyncio.FIRST_COMPLETED)
        failures = [f.exception() for f in done if f.exception()]
        if failures:
            raise failures[0]
    except BaseException as e:
        error = e
        raise
    finally:
        for task in [*tasks, stop]:
            task.cancel()
        executor.stats.finished = time.monotonic()
        last_run = executor.stats
        telemetry.report(executor.stats, nodes, ops, error)
    return [sink.result() for sink in sinks]


async def stream_events(node: Node, max_concurrency: int | None = None) -> AsyncIterator[Event]:
    sink = QueueSink()
    run = asyncio.ensure_future(run_async([node], max_concurrency, [sink]))
    try:
        while (event := await sink.out.get()) is not END:
            yield event
        await run
    finally:
        run.cancel()


def run(nodes: list[Node], max_concurrency: int | None = None) -> list[list[Row]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_async(nodes, max_concurrency))
    raise RuntimeError("collect() can't run inside an event loop (e.g. Jupyter); use `await table.acollect()`")
