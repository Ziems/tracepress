"""The server and its clients: register pipelines from one client, listen from others."""

import asyncio

import pytest
from fakes import FuncLM

import tracepress as tp
from tracepress.cli import main
from tracepress.rows import Closed, Row
from tracepress.schema import Schema
from tracepress.server.app import serve_in_thread

STEPS = Schema(("trace", "step"), {"message": str})


def echo(**outputs):
    return FuncLM(lambda fields: outputs)


def step(trace, n, message="hi"):
    return Row((("trace", trace), ("step", n)), {"message": message})


@pytest.fixture
def server():
    srv = serve_in_thread()
    yield srv
    srv.stop()


async def take(client, name, n, from_start=False, timeout=10.0):
    """The first `n` row events a pipeline sends."""
    events = []

    async def go():
        async for e in client.listen(name, from_start=from_start):
            if e.row is not None:
                events.append(e)
                if len(events) == n:
                    return

    await asyncio.wait_for(go(), timeout)
    return events


async def after_listening(push, *listeners):
    """Start the listeners, push once they're connected, and wait for them all."""
    tasks = [asyncio.ensure_future(listener) for listener in listeners]
    await asyncio.sleep(0.5)
    await asyncio.get_running_loop().run_in_executor(None, push)
    return await asyncio.gather(*tasks)


def test_register_from_one_client_listen_from_others(server):
    registrar = tp.connect(server.address)
    registrar.declare({"steps": STEPS})
    live = registrar.source
    summaries = live.steps.lm_map("message -> summary", lm=echo(summary="s"))
    per_trace = summaries.group_by("trace").lm_reduce("summary -> overall", lm=echo(overall="o"))
    registrar.register(name="summaries", pipeline=summaries)
    tp.connect(server.address).register(name="per_trace", pipeline=per_trace)  # any client can register

    def push():
        tp.connect(server.address).push([step("a", 1), step("a", 2), step("a", 3), Closed((("trace", "a"),))],
                                        table="steps")

    bot, runner, oncall = (tp.connect(server.address) for _ in range(3))
    seen_bot, seen_runner, totals = asyncio.run(after_listening(
        push, take(bot, "summaries", 3), take(runner, "summaries", 3), take(oncall, "per_trace", 1)))
    assert [e.row.summary for e in seen_bot] == ["s"] * 3 == [e.row.summary for e in seen_runner]
    assert totals[0].row.overall == "o" and totals[0].row.trace == "a"

    # A new step reruns only what depends on it: one new summary, the old ones aren't sent again.
    more = asyncio.run(after_listening(lambda: bot.push([step("b", 1)], table="steps"), take(bot, "summaries", 1)))
    assert [e.row.trace for e in more[0]] == ["b"]
    info = {p["name"]: p for p in registrar.pipelines()}
    assert info["summaries"]["rows"] == 4 and info["summaries"]["status"] == "running"


def test_a_late_pipeline_catches_up(server):
    client = tp.connect(server.address)
    client.declare({"steps": STEPS})
    client.push([step("a", 1), step("a", 2)], table="steps")
    client.register(name="late", pipeline=client.source.steps.lm_map("message -> summary", lm=echo(summary="s")))
    rows = asyncio.run(take(client, "late", 2, from_start=True))
    assert sorted(e.row.step for e in rows) == [1, 2]


def test_unregister_ends_listeners(server):
    client = tp.connect(server.address)
    client.declare({"steps": STEPS})
    client.register(name="p", pipeline=client.source.steps.lm_map("message -> summary", lm=echo(summary="s")))

    async def listen_then_stop():
        got = []

        async def listen():
            async for e in client.listen("p"):
                got.append(e)

        task = asyncio.ensure_future(listen())
        await asyncio.sleep(0.5)
        await asyncio.get_running_loop().run_in_executor(None, client.unregister, "p")
        await asyncio.wait_for(task, 5)  # the stream ends instead of hanging
        return got

    assert asyncio.run(listen_then_stop()) == []
    assert client.pipelines() == []


SEEN: list = []


def record_step(row):
    SEEN.append(row.step)
    return {}


def test_registering_a_name_twice_at_once_leaves_one_pipeline_running(server):
    """Each registration replaces the last, even when they overlap: no replaced pipeline keeps running."""
    import threading
    import time

    client = tp.connect(server.address)
    client.declare({"steps": STEPS})
    pipeline = client.source.steps.map(record_step, columns={})
    client.register("p", pipeline)
    racers = [threading.Thread(target=client.register, args=("p", pipeline)) for _ in range(2)]
    for t in racers:
        t.start()
    for t in racers:
        t.join()
    SEEN.clear()
    client.push([step("a", 1)], table="steps")
    time.sleep(0.5)
    assert SEEN == [1]


def test_source_only_runs_on_the_server(server):
    client = tp.connect(server.address)
    client.declare({"steps": STEPS})
    with pytest.raises(RuntimeError, match="runs on the server"):
        client.source.steps.lm_map("message -> summary", lm=echo(summary="s")).collect()


def test_tokens(server):
    with pytest.raises(ValueError, match="token"):
        serve_in_thread(host="0.0.0.0")
    with pytest.raises(SystemExit):
        main(["serve", "--host", "0.0.0.0"])
    guarded = serve_in_thread(token="secret")
    try:
        with pytest.raises(RuntimeError, match="401"):
            tp.connect(guarded.address).pipelines()
        assert tp.connect(guarded.address, token="secret").pipelines() == []
    finally:
        guarded.stop()


class Marker:
    """Unpickling this records that the server ran it."""

    ran = False

    def __reduce__(self):
        return (setattr, (Marker, "ran", True))


def test_web_pages_cant_reach_the_server(server):
    """A page in the user's browser can POST to localhost; the server must not unpickle what it sends."""
    import pickle
    import urllib.error
    import urllib.request

    import aiohttp

    def post(headers):
        req = urllib.request.Request(f"http://{server.address}/register", data=pickle.dumps(Marker()),
                                     method="POST", headers=headers)
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            return e.code

    assert post({"Content-Type": "text/plain", "Origin": "https://example.com"}) == 403
    assert post({"Content-Type": "text/plain"}) == 415
    assert not Marker.ran

    client = tp.connect(server.address)
    client.declare({"steps": STEPS})
    client.register("p", client.source.steps)

    async def listen_from_a_page():
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError, match="403"):
                await session.ws_connect(f"ws://{server.address}/listen/p", origin="https://example.com")

    asyncio.run(listen_from_a_page())


def test_stopping_a_server_twice_returns_at_once():
    import time

    srv = serve_in_thread()
    srv.stop()
    start = time.monotonic()
    srv.stop()
    assert time.monotonic() - start < 2  # the stopped loop never ran the second cleanup: 10 s, then TimeoutError


def test_stream_an_adapter(server, harbor_dir):
    client = tp.connect(server.address)
    src = tp.read.harbor(harbor_dir)
    sent = client.stream(src, tables=["steps"]).wait(10)
    assert sent > 0
    assert sorted(client.source.tables()) == ["steps"]
