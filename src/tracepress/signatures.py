"""Turning user signatures into the DSPy signatures each operator actually runs."""

from __future__ import annotations

from typing import Any

import dspy

SigLike = str | type[dspy.Signature]

# Reserved reduce input that receives whole child records.
ROWS = "rows"


def parse(sig: SigLike, instructions: str | None = None) -> type[dspy.Signature]:
    parsed = dspy.Signature(sig) if isinstance(sig, str) else sig
    if instructions:
        parsed = parsed.with_instructions(instructions)
    return parsed


def inputs(sig: type[dspy.Signature]) -> list[str]:
    return list(sig.input_fields)


def outputs(sig: type[dspy.Signature]) -> dict[str, Any]:
    return {name: f.annotation for name, f in sig.output_fields.items()}


def bool_output(sig: type[dspy.Signature]) -> str:
    """The single bool output of an lm_filter signature."""
    outs = outputs(sig)
    if len(outs) != 1 or next(iter(outs.values())) is not bool:
        raise ValueError(f"lm_filter needs exactly one bool output, got {outs}")
    return next(iter(outs))


def list_output(sig: type[dspy.Signature]) -> str:
    """The single list output of an lm_extract signature."""
    outs = outputs(sig)
    if len(outs) != 1 or getattr(next(iter(outs.values())), "__origin__", None) is not list:
        raise ValueError(f"lm_extract needs exactly one list[...] output, got {outs}")
    return next(iter(outs))


def _is_default_instructions(sig: type[dspy.Signature]) -> bool:
    return sig.instructions.startswith("Given the fields")


def reduce_signature(
    sig: type[dspy.Signature], key_inputs: dict[str, Any], child_columns: list[str], children_name: str
) -> type[dspy.Signature]:
    """Rewrite a reduce signature so children arrive as one list of aligned records.

    `"task, summary, succeeded -> notes"` grouped by task becomes
    `"task, trials: list[dict] -> notes"`, where each dict holds one trial's summary and
    succeeded. Keeping each child's fields together lets the model tell which summary
    goes with which outcome. `key_inputs` maps each group-key input to its type.
    """
    fields = {name: (type_, dspy.InputField()) for name, type_ in key_inputs.items()}
    desc = f"one record per {children_name.rstrip('s')}, with fields: {', '.join(child_columns)}"
    fields[children_name] = (list[dict], dspy.InputField(desc=desc))
    for name, f in sig.output_fields.items():
        fields[name] = (f.annotation, f)
    if _is_default_instructions(sig):
        given = ", ".join(f"`{n}`" for n in [*key_inputs, children_name])
        produced = ", ".join(f"`{n}`" for n in sig.output_fields)
        instructions = f"Given {given}, produce {produced}, considering all of the {children_name} together."
    else:
        instructions = sig.instructions
    return dspy.Signature(fields, instructions)


def merge_signature(
    reduce_sig: type[dspy.Signature], key_inputs: dict[str, Any], children_name: str
) -> type[dspy.Signature]:
    """Combines partial reduce outputs, each computed over a subset of the children."""
    fields = {name: (type_, dspy.InputField()) for name, type_ in key_inputs.items()}
    fields["partials"] = (
        list[dict],
        dspy.InputField(desc=f"partial results, each computed over a different subset of the {children_name}"),
    )
    for name, f in reduce_sig.output_fields.items():
        fields[name] = (f.annotation, f)
    instructions = (
        f"Each partial result answered the task below for a subset of the {children_name}. "
        f"Combine them into a single result covering all of the {children_name}.\n\n"
        f"Task: {reduce_sig.instructions}"
    )
    return dspy.Signature(fields, instructions)
