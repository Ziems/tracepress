"""Runs a plan DAG: one asyncio task per operator, connected by queues.

Batch and streaming share this runtime. A bounded source simply ends.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from .lm import Executor, RunStats
from .operators import END, Operator
from .rows import Row, RowAdded, RowUpdated, sort_key

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
    """Keeps the latest final version of each row."""

    def __init__(self) -> None:
        self.rows: dict = {}

    async def consume(self, queue: asyncio.Queue) -> None:
        while (event := await queue.get()) is not END:
            if isinstance(event, (RowAdded, RowUpdated)) and event.row.final:
                self.rows[event.row.key] = event.row

    def result(self) -> list[Row]:
        return sorted(self.rows.values(), key=lambda r: sort_key(r.key))


async def run_async(nodes: list[Node], max_concurrency: int | None = None) -> list[list[Row]]:
    global last_run
    executor = Executor(max_concurrency)
    ops: dict[int, Operator] = {}
    coros = []

    def build(node: Node) -> Operator:
        if id(node) in ops:
            return ops[id(node)]
        op = node.make()
        op.executor = executor
        ops[id(node)] = op
        if node.parent is None:
            coros.append(op.produce())
        else:
            queue: asyncio.Queue = asyncio.Queue()
            build(node.parent).outputs.append(queue)
            coros.append(op.consume(queue))
        return op

    sinks = []
    for node in nodes:
        sink, queue = CollectSink(), asyncio.Queue()
        build(node).outputs.append(queue)
        coros.append(sink.consume(queue))
        sinks.append(sink)

    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            task.result()  # re-raise the first failure
    finally:
        for task in tasks:
            task.cancel()
        executor.stats.finished = time.monotonic()
        last_run = executor.stats
    return [sink.result() for sink in sinks]


def run(nodes: list[Node], max_concurrency: int | None = None) -> list[list[Row]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_async(nodes, max_concurrency))
    raise RuntimeError("collect() can't run inside an event loop (e.g. Jupyter); use `await table.acollect()`")
