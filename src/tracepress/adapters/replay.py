"""Replay a bounded source as a live stream, for testing streaming without a live system."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from datetime import datetime

from ..rows import Closed, Event, Key, Row, RowAdded
from ..schema import Schema
from .base import Adapter

# Columns tried, in order, for when a row "happened". A trial row is emitted when the
# trial finishes, since that's when its reward is known.
TIME_COLUMNS = ("timestamp", "finished_at", "started_at")


def _time(row: Row, columns: tuple[str, ...]) -> float | None:
    for column in columns:
        value = row.values.get(column)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return None


class ReplayAdapter(Adapter):
    bounded = False

    def __init__(self, source: Adapter, speed: float | None, seed: int, drop_closes: float,
                 shuffle_within: int, duplicates: float, time_columns: tuple[str, ...]):
        self.source = source
        self.levels = source.levels
        self.speed = speed
        self.seed = seed
        self.drop_closes = drop_closes
        self.shuffle_within = shuffle_within
        self.duplicates = duplicates
        self.time_columns = time_columns

    def tables(self) -> dict[str, Schema]:
        return self.source.tables()

    async def events(self, table: str) -> AsyncIterator[Event]:
        timeline = await self.timeline(table)
        rng = random.Random(self.seed)
        timeline = self._shuffle(timeline, rng)
        start = timeline[0][0] if timeline else 0.0
        clock = asyncio.get_running_loop().time()
        for t, event in timeline:
            if self.speed:
                delay = clock + (t - start) / self.speed - asyncio.get_running_loop().time()
                await asyncio.sleep(max(delay, 0))
            else:
                await asyncio.sleep(0)  # let downstream operators run between events
            if isinstance(event, Closed) and rng.random() < self.drop_closes:
                continue
            yield event
            if isinstance(event, RowAdded) and rng.random() < self.duplicates:
                yield event

    async def timeline(self, table: str) -> list[tuple[float, Event]]:
        """Every event with the time it should happen, in replay order.

        Rows are placed at their own time. A close is placed right after the last row
        under it (deeper closes first), so a close never overtakes its rows. A close with
        no rows under it (e.g. a crashed trial with no steps) keeps its position relative
        to the source's preceding row.
        """
        rows: list[tuple[float, int, Row]] = []
        closes: list[tuple[Key, float]] = []
        last = float("-inf")
        async for event in self.source.events(table):
            if isinstance(event, Closed):
                closes.append((event.prefix, last))
                continue
            t = _time(event.row, self.time_columns)
            if t is None:
                t = last if last != float("-inf") else 0.0
            last = max(last, t)
            rows.append((t, len(rows), event.row))

        latest: dict[Key, float] = {}
        for t, _, row in rows:
            for i in range(1, len(row.key) + 1):
                prefix = row.key[:i]
                latest[prefix] = max(latest.get(prefix, t), t)

        ordered: list[tuple[tuple, Event]] = [((t, 0, 0, seq), RowAdded(row)) for t, seq, row in rows]
        for seq, (prefix, fallback) in enumerate(closes):
            t = latest.get(prefix, fallback)
            ordered.append(((t, 1, -len(prefix), seq), Closed(prefix)))
        ordered.sort(key=lambda item: item[0])
        return [(sort_key[0], event) for sort_key, event in ordered]

    def _shuffle(self, timeline: list[tuple[float, Event]], rng: random.Random) -> list[tuple[float, Event]]:
        """Shuffle rows within windows of `shuffle_within`, never across a close.

        Rows stay before every close that covers them, so the ordering invariant holds;
        only the order of rows relative to each other changes.
        """
        if self.shuffle_within <= 1:
            return timeline
        out: list[tuple[float, Event]] = []
        run: list[tuple[float, Event]] = []

        def flush():
            for i in range(0, len(run), self.shuffle_within):
                window = run[i : i + self.shuffle_within]
                times = [t for t, _ in window]
                rng.shuffle(window)
                out.extend(zip(times, (e for _, e in window)))  # keep the clock monotonic
            run.clear()

        for item in timeline:
            if isinstance(item[1], Closed):
                flush()
                out.append(item)
            else:
                run.append(item)
        flush()
        return out


def replay(source: Adapter, speed: float | None = None, seed: int = 0, drop_closes: float = 0.0,
           shuffle_within: int = 0, duplicates: float = 0.0,
           time_columns: tuple[str, ...] = TIME_COLUMNS) -> ReplayAdapter:
    """Replay a bounded source as a stream, in timestamp order.

    `speed=100` plays 100x faster than real time; `None` plays as fast as possible but
    keeps the interleaving. Fault injection, all seeded:

    - `drop_closes`: fraction of closes never sent (traces that never finish). Their
      groups close at the end of the stream instead.
    - `shuffle_within`: shuffle rows within windows of this many events.
    - `duplicates`: fraction of rows sent twice.

    With a warm LM cache, a pipeline over `replay(src)` must give the same final rows as
    over `src` itself.
    """
    if not source.bounded:
        raise ValueError("replay needs a bounded source")
    for name, p in (("drop_closes", drop_closes), ("duplicates", duplicates)):
        if not 0 <= p <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    return ReplayAdapter(source, speed, seed, drop_closes, shuffle_within, duplicates, tuple(time_columns))
