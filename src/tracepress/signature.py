"""Signatures: which columns an operator reads, and which columns it writes. These are DSPy signatures.

    df.lm_map("product, review -> has_defect: bool, part", "Does the review report a defect?")

A string signature is ``inputs -> outputs``. Outputs are ``str`` unless typed (``bool``,
``int``, ``float``, ``list[str]``, ``Literal['a', 'b']``, ...). Operators that only read
rows (filter, topk, dedup) take just the inputs: ``"product, review"``.

When fields need descriptions, subclass ``tracepress.Signature`` (which is ``dspy.Signature``);
its docstring is the instruction:

    class FindIssue(tracepress.Signature):
        \"\"\"Does the review report a defect, and in which part?\"\"\"
        product: str = tracepress.In()
        review: str = tracepress.In()
        part: str = tracepress.Out(desc="the part at fault; empty if none")
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import dspy
from pydantic import BaseModel, create_model

from .prompt import as_text

Signature = dspy.Signature
In = dspy.InputField
Out = dspy.OutputField

def to_signature(spec: Any, instruction: str | None = None, *, outputs: bool | None = None) -> type[dspy.Signature]:
    """A DSPy signature from a string or class, with ``instruction`` attached.

    ``outputs=True`` requires output fields, ``False`` forbids them, None allows either.
    An explicit ``instruction`` overrides a class signature's docstring.
    """
    if isinstance(spec, str):
        if "->" not in spec:
            if outputs is True:
                raise ValueError(f"signature {spec!r} needs outputs: 'inputs -> outputs'")
            sig = dspy.Signature(f"{spec} -> tracepress_placeholder_").delete("tracepress_placeholder_")
        else:
            sig = dspy.Signature(spec)
    elif isinstance(spec, type) and issubclass(spec, dspy.Signature):
        sig = spec
    else:
        raise TypeError(f"a signature is a string like 'a, b -> c' or a tracepress.Signature class, not {type(spec).__name__}")

    if not sig.input_fields:
        raise ValueError(f"signature {spec!r} names no input columns")
    if outputs is True and not sig.output_fields:
        raise ValueError(f"signature {spec!r} needs outputs: 'inputs -> outputs'")
    if outputs is False and sig.output_fields:
        raise ValueError(f"this operator only reads rows; give just the inputs, not {spec!r}")
    if instruction is not None:
        sig = sig.with_instructions(instruction)
    return sig


def has_instruction(sig: type[dspy.Signature]) -> bool:
    """Whether the signature carries a real instruction, not DSPy's generated default."""
    default = dspy.Signature(sig.signature).instructions if "->" in sig.signature else None
    return bool(sig.instructions) and sig.instructions != default


def check(sig: type[dspy.Signature], columns: Iterable[Any]) -> None:
    """Raise if an input is not a column, before any model call is made."""
    cols = set(map(str, columns))
    missing = [n for n in sig.input_fields if n not in cols]
    if missing:
        raise KeyError(f"signature inputs {missing} are not columns; the columns are {sorted(cols)}")


def inputs(sig: type[dspy.Signature], row: Mapping[str, Any]) -> dict[str, Any]:
    """The row's values for the signature's inputs (missing values become None)."""
    out = {}
    for name in sig.input_fields:
        value = row[name]
        try:
            if value != value:  # NaN
                value = None
        except (TypeError, ValueError):
            pass
        out[name] = value
    return out


def render(sig: type[dspy.Signature], row: Mapping[str, Any], limit: int | None = None) -> str:
    """The row's inputs as ``name: value`` lines, for ops that show many rows in one call."""
    return "\n".join(f"{name}: {as_text(row[name], limit)}" for name in sig.input_fields)


def _type_name(annotation: Any) -> str:
    return annotation.__name__ if isinstance(annotation, type) else str(annotation).replace("typing.", "")


def describe(sig: type[dspy.Signature]) -> str:
    """The signature as a string, with output types: ``"a, b -> c: bool, d"``."""
    ins = ", ".join(sig.input_fields)
    outs = ", ".join(
        name if f.annotation is str else f"{name}: {_type_name(f.annotation)}" for name, f in sig.output_fields.items()
    )
    return f"{ins} -> {outs}" if outs else ins


def output_schema(sig: type[dspy.Signature]) -> dict[str, Any]:
    """The JSON schema of the outputs, recorded in manifests so the agent sees allowed values."""
    fields = {name: (f.annotation, ...) for name, f in sig.output_fields.items()}
    return strict_json_schema(create_model("Outputs", **fields))


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A pydantic model's JSON schema without titles, defaults, or open objects."""

    def fix(node: Any) -> Any:
        if isinstance(node, list):
            return [fix(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for key, value in node.items():
            if key in ("title", "default"):
                continue
            out[key] = {n: fix(v) for n, v in value.items()} if key in ("properties", "$defs") else fix(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return fix(model.model_json_schema())


__all__ = ["Signature", "In", "Out", "to_signature", "check", "inputs", "render", "describe", "output_schema"]
