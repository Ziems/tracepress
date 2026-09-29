from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from typing import Any, Callable, Sequence

# Opt out before the package is imported, and keep the install id off this machine.
os.environ["TRACEPRESS_TELEMETRY"] = "off"
os.environ["TRACEPRESS_HOME"] = tempfile.mkdtemp(prefix="tracepress-test-home-")

import pytest
from dspy.lm15 import Message, Request, Response, TextPart, ToolCallPart, Usage

import tracepress

# Fakes are lm15 engines: DSPy sends them the canonical Request its JSON adapter built, and
# parses the Response they return. Nothing reaches a network.


def prompt_of(request: Request) -> str:
    """The text of the last user message (DSPy puts the inputs there)."""
    return "\n".join(p.text for p in request.messages[-1].parts if isinstance(p, TextPart))


def field_value(request: Request, name: str) -> str:
    """One input field's value, from DSPy's ``[[ ## name ## ]]`` blocks."""
    m = re.search(rf"\[\[ ## {name} ## \]\]\n(.*?)(?=\n\n\[\[ ## |\n\nRespond with|\Z)", prompt_of(request), re.S)
    return m.group(1) if m else ""


def output_fields(request: Request) -> list[str]:
    """The output fields DSPy asked for, in order."""
    tail = prompt_of(request).rsplit("Respond with", 1)[-1]
    return re.findall(r"`(\w+)`", tail)


def response(text: str) -> Response:
    return Response(
        id="fake",
        model="fake",
        message=Message.assistant(TextPart(text)),
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def tool_call(tool: str, **arguments: Any) -> Response:
    return Response(
        id="fake",
        model="fake",
        message=Message.assistant([ToolCallPart(id=f"call_{tool}", name=tool, input=arguments)]),
        finish_reason="tool_call",
        usage=Usage(input_tokens=100, output_tokens=20),
    )


class RuleEngine:
    """Answers each request with ``fn(request)``: a JSON string, or a dict to encode. Thread-safe."""

    def __init__(self, fn: Callable[[Request], Any]):
        self.fn = fn
        self.requests: list[Request] = []
        self._lock = threading.Lock()

    def complete(self, request: Request) -> Response:
        with self._lock:
            self.requests.append(request)
        out = self.fn(request)
        return out if isinstance(out, Response) else response(out if isinstance(out, str) else json.dumps(out))


class ScriptedEngine:
    """Answers requests in order from a script of Responses or strings (for the agent loop)."""

    def __init__(self, script: Sequence[Response | str]):
        self.script = list(script)
        self.requests: list[Request] = []

    def complete(self, request: Request) -> Response:
        self.requests.append(request)
        assert self.script, f"ScriptedEngine ran out after {len(self.requests) - 1} requests"
        item = self.script.pop(0)
        return item if isinstance(item, Response) else response(item)


def numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"-?\d+", text)]


@pytest.fixture
def make_lm():
    def make(fn: Callable[[Request], Any], **kwargs) -> tuple[tracepress.Model, RuleEngine]:
        engine = RuleEngine(fn)
        kwargs.setdefault("cache", False)
        kwargs.setdefault("progress", False)
        return tracepress.Model("fake/model", engine=engine, **kwargs), engine

    return make
