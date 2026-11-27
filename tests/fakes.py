"""Fake LMs: answers computed from the prompt's input fields, or fixed JSON."""

import asyncio
import json
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


class _JSONEngine:
    """Answers every request with fixed JSON, declaring JSON outputs like OpenAI's engine."""

    supported_params = ("response_format", "temperature", "max_tokens", "n")

    def __init__(self, answer: dict, schemas: bool):
        self.supports_response_schema = schemas
        self.answer = json.dumps(answer)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return Response(
            id=None,
            model="json",
            message=Message.assistant([TextPart(self.answer)]),
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    def close(self):
        pass


class _AsyncJSONEngine:
    def __init__(self, engine):
        self.engine = engine

    async def complete(self, request):
        return self.engine.complete(request)

    async def aclose(self):
        pass


class JSONLM(dspy.LM):
    engine_requests: list  # what reached its engine


def structured_lm(answer: dict, schemas: bool = True) -> JSONLM:
    """A dspy.LM that answers with `answer` as JSON, taking DSPy's structured-output path
    (or, with `schemas=False`, its plain JSON mode).

    Its engine's `requests` lists what reached it, so a cache hit shows as no new request.
    """
    engine = _JSONEngine(answer, schemas)
    lm = JSONLM("openai/json", engine=engine, async_engine=_AsyncJSONEngine(engine))
    lm.engine_requests = engine.requests
    return lm
