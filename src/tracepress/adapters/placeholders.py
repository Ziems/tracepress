"""Adapters we intend to support but haven't built yet.

They exist so the API (and our design) doesn't drift toward any single trace format.
"""

from __future__ import annotations


def _planned(name: str, hierarchy: str):
    def reader(*args, **kwargs):
        raise NotImplementedError(
            f"The {name} adapter isn't implemented yet (planned hierarchy: {hierarchy}). "
            "See SPEC.md, 'Planned adapters'."
        )

    reader.__name__ = name
    reader.__doc__ = f"Planned: {name} traces ({hierarchy})."
    return reader


read_dspy = _planned("DSPy", "run > module call (nested) > LM call")
read_otel = _planned("OpenTelemetry / OpenInference", "trace > span tree via parent_id")
read_claude_code = _planned("Claude Code / Agent SDK session", "session > message (+ subagent sidechains)")
read_chatgpt = _planned("ChatGPT conversation export", "conversation > message (branching tree)")
