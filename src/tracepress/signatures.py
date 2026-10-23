"""Turning user signatures into the DSPy signatures each operator actually runs."""

from __future__ import annotations

from dataclasses import dataclass
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


def running_reduce_signature(
    reduce_sig: type[dspy.Signature], key_inputs: dict[str, Any], children_name: str
) -> type[dspy.Signature]:
    """Updates a provisional reduce result with newly arrived children (for emit="update")."""
    fields = {name: (type_, dspy.InputField()) for name, type_ in key_inputs.items()}
    fields["current"] = (dict | None, dspy.InputField(desc="the result so far; null if nothing has arrived yet"))
    fields[f"new_{children_name}"] = (
        list[dict],
        dspy.InputField(desc=f"{children_name} that arrived since the current result"),
    )
    for name, f in reduce_sig.output_fields.items():
        fields[name] = (f.annotation, f)
    instructions = (
        f"{children_name.capitalize()} arrive over time. Update the current result to account for the new "
        f"{children_name}, keeping what still holds from the current result.\n\nTask: {reduce_sig.instructions}"
    )
    return dspy.Signature(fields, instructions)


@dataclass
class FoldSpec:
    """A parsed fold signature. State fields appear on both sides of the arrow."""

    inputs: list[str]  # in the order written, state fields included
    outputs: dict[str, Any]  # name -> type
    output_fields: dict[str, Any]  # name -> dspy FieldInfo
    state: list[str]
    instructions: str | None

    def previous(self, name: str) -> str:
        return f"previous_{name}"


def parse_fold(sig: str, instructions: str | None = None) -> FoldSpec:
    if not isinstance(sig, str) or "->" not in sig:
        raise TypeError('lm_fold takes a string signature such as "progress, step -> progress, stuck: bool"')
    lhs, rhs = (side.strip() for side in sig.split("->", 1))
    ins = list(dspy.Signature(f"{lhs} -> fold_output__").input_fields)
    outs_sig = dspy.Signature(f"fold_input__ -> {rhs}")
    state = [name for name in outs_sig.output_fields if name in ins]
    if not state:
        raise ValueError(
            f"lm_fold needs at least one state field on both sides of the arrow, e.g. 'progress, step -> progress'; got {sig!r}"
        )
    return FoldSpec(
        inputs=ins,
        outputs={n: f.annotation for n, f in outs_sig.output_fields.items()},
        output_fields=dict(outs_sig.output_fields),
        state=state,
        instructions=instructions,
    )


def fold_signature(
    spec: FoldSpec, key_inputs: dict[str, Any], child_inputs: dict[str, Any], child_name: str
) -> type[dspy.Signature]:
    fields = {}
    for name in spec.inputs:
        if name in spec.state:
            desc = f"`{name}` before this {child_name}; the initial value if this is the first"
            fields[spec.previous(name)] = (spec.outputs[name], dspy.InputField(desc=desc))
        elif name in key_inputs:
            fields[name] = (key_inputs[name], dspy.InputField())
        else:
            fields[name] = (child_inputs[name], dspy.InputField(desc=f"from the newest {child_name}"))
    for name, f in spec.output_fields.items():
        fields[name] = (spec.outputs[name], f)
    if spec.instructions:
        instructions = spec.instructions
    else:
        state = ", ".join(f"`{n}`" for n in spec.state)
        instructions = (
            f"{child_name.capitalize()}s arrive one at a time, in order. Update {state} to account for the "
            f"newest {child_name}, and produce the remaining outputs."
        )
    return dspy.Signature(fields, instructions)


def default_value(type_: Any) -> Any:
    """A neutral initial value for a fold state field."""
    origin = getattr(type_, "__origin__", type_)
    return {str: "", list: [], dict: {}, bool: False, int: 0, float: 0.0}.get(origin)


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
