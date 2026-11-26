"""A fake LM whose answers are computed from the prompt's input fields."""

import asyncio
import re

import dspy
from dspy.lm15 import Message, Response, TextPart, Usage

FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n(?:\[\[ ## |Respond with)|\Z)", re.DOTALL)


def _last_user_text(request) -> str:
    return "".join(p.text for p in request.messages[-1].parts if isinstance(p, TextPart))


class _Engine:
    """DSPy 3.5 engine: `complete(Request) -> Response`, answering through the owning FuncLM."""

    def __init__(self, owner):
        self.owner = owner

    def complete(self, request):
        return self.owner._answer(request)

    def close(self):
        pass


class _AsyncEngine:
    def __init__(self, owner):
        self.owner = owner

    async def complete(self, request):
        return await self.owner._aanswer(request)

    async def aclose(self):
        pass


class FuncLM(dspy.BaseLM):
    """`fn(fields) -> outputs`, where `fields` maps each input field name to its rendered text.

    Speaks DSPy's chat format, so use it with `tp.configure(adapter=dspy.ChatAdapter())`
    (the test suite does this for every test).
    """

    def __init__(self, fn):
        super().__init__(model="func", engine=_Engine(self), async_engine=_AsyncEngine(self))
        self.fn = fn
        self.calls = 0
        self.seen: list[dict] = []
        self._cache_responses = False  # answer every call, like a live model

    def _answer(self, request):
        self.calls += 1
        out = self.fn(dict(FIELD.findall(_last_user_text(request))))
        content = "".join(f"[[ ## {k} ## ]]\n{v}\n\n" for k, v in out.items()) + "[[ ## completed ## ]]"
        return Response(
            id=None,
            model="func",
            message=Message.assistant([TextPart(content)]),
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    async def _aanswer(self, request):
        return self._answer(request)


class HangingLM(FuncLM):
    """Like FuncLM, but the first `hangs` requests never answer, like a stuck connection."""

    def __init__(self, fn, hangs: int):
        super().__init__(fn)
        self.hangs = hangs
        self.attempts = 0

    async def _aanswer(self, request):
        self.attempts += 1
        if self.attempts <= self.hangs:
            await asyncio.sleep(3600)
        return self._answer(request)
