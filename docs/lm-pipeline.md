# Amendment: LM-Backed Stages

An amendment to [`pipeline-spec.md`](pipeline-spec.md). Read that first. This document adds one capability: a stage
may be a declarative language model module instead of a Python function.

Section numbers below refer to the base spec.

## A1. What this adds and what it does not

A stage name in config may now resolve to an LM module rather than a plain callable. The config
does not change. `- map: label` is written identically whether `label` is a Python function or
an LM module.

Everything semantic about the stage lives in the stage. The signature, the instructions, and
any few-shot demonstrations are all properties of the module in `./stages`. The provider and
model id are execution policy and belong in config, because changing from a cheap model to a
stronger model should not require editing stage code. The MVP has a built-in provider and model
default so a first LM pipeline can run without extra YAML.

This preserves base spec principle 2, config owns the topology. A prompt is not topology, and a
model id is not topology either. What the config gains is the ability to say "run this declared
stage with this provider and model," and nothing more in the MVP.

**Explicitly deferred.** Context window management, folding, and hierarchical reduce are out of
scope for this amendment. Assume any reduce input fits in one call. The base spec's variable
group sizes still hold, and a group of 30 is passed to the LM in a single invocation. Folding is
a later amendment and nothing here should foreclose it.

## A1.1. LM MVP boundary

The LM MVP should prove that declarative modules can participate in the existing stage contract
without adding a second runtime.

In scope for the LM MVP:

- DSPy-backed LM stages, detected through an inspectable `signature` with `input_fields` and
  `output_fields`.
- Duck typing for the signature surface where practical, so tests and small custom modules can
  use the same contract without invoking DSPy.
- LM stages in `map`, `filter`, `expand`, and `reduce`.
- A minimal `lm` config surface for provider and model selection: built-in OpenRouter/DeepSeek
  defaults, top-level overrides, and per-step model override.
- Field mapping, passthrough, and clear missing-field errors.
- Load-time signature validation for `filter`, `expand`, and `reduce`.
- Reduce field resolution from gathered children plus parent fields, including synthetic parent
  fields for `by` groups.
- Mixed Python and LM stages in one pipeline.
- Invalid LM outputs raise from the adapter and flow through the base `on_child_error` policy.
- Bounded concurrent LM invocation with `lm.max_concurrent`.

Post-MVP:

- `lm` config policy beyond provider/model selection, including retries and repair.
- Record-level caching and module-state hashing.
- Structured LM invocation tracing.
- Optimization/compilation workflows. These need their own later amendment if they still feel
  worth doing.

## A2. Stage form

```python
# ./stages.py
import dspy

class Label(dspy.Signature):
    """Label a session with its dominant failure mode."""
    session: str = dspy.InputField()
    label: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

label = dspy.Predict(Label)
```

```yaml
- map: label
```

DSPy is the supported authoring dependency. Detect LM modules by duck typing on the presence of
an introspectable `signature` attribute exposing `input_fields` and `output_fields`, so a small
custom module class can still work in tests or local experiments without modification.

DSPy modules are executed by DSPy (`dspy.Predict` via stock `dspy.LM`). The provider and model
still come from config: `lm.provider`, `lm.default`, and per-step `lm.model`.

## A3. Amends section 3, stage contract

Section 3 currently defines resolution, arity binding, and the four callable shapes. Add a fifth
resolution outcome ahead of the arity check.

At load time, after resolving a stage name to an object:

1. If the object has an introspectable signature, it is an **LM stage**. Skip signature binding
   from section 3 entirely, since LM stages are not called positionally. Apply the field mapping
   in A4.
2. Otherwise it is a Python stage and section 3 applies unchanged.

LM stages do not receive `ctx`. This is deliberate rather than a limitation. A module that takes
only fields and returns only fields is a pure function, which is what makes it optimizable in
A10. Giving it access to group state would break that.

A stage that needs `ctx` is written as a Python stage that wraps the module. Do not wrap
`Predict` unless you need `ctx`; the module name in YAML is enough.

```python
_label = dspy.Predict(Label)

def label(record, ctx):
    pred = _label(session=record["session"])
    if pred.confidence < 0.5:
        ctx.side("low_conf", record)
    return {**record, "label": pred.label}
```

This is the single escape hatch. One mechanism, two entry points, and pure stages stay pure. A
wrapped module resolves as a Python stage, since the wrapper is a plain function.

## A4. Field mapping

Records are dicts. Signature fields name record keys.

**Inputs.** Build kwargs as `{f: record[f] for f in signature.input_fields}`.

**Outputs.** Merge the prediction back with `{**record, **outputs}`.

**Passthrough is required.** Fields the signature does not mention survive untouched. Without
this, the first LM stage in a chain silently destroys ids, upstream annotations, and anything a
later stage depends on, and the resulting failure is far from its cause.

**Missing input field.** Cannot be checked at load time, since it depends on the data. Fail on
the first record with an error naming the missing field, the stage, and the keys the record
actually had. The default failure here is a `KeyError` raised from inside a framework call
stack, which is miserable to trace, so catch and re-raise.

## A5. Amends section 2, step contracts

The base spec enforces emit counts at runtime for `map` and `filter`. For LM stages these are
checkable at **load time** by inspecting type annotations, which is strictly better and fits
principle 5.

| Step | Required signature shape | Emits |
|---|---|---|
| `map` | any outputs | 1, merged |
| `filter` | exactly one `bool` output field | 1 if true, 0 if false |
| `expand` | exactly one `list[T]` output field | one child per element |
| `reduce` | exactly one `list[T]` input field | 1 |

Load-time validation:

- A `filter` whose module has zero or more than one bool output is a load error.
- An `expand` whose module has zero or more than one list-typed output is a load error.
- A `reduce` whose module has zero or more than one list-typed input is a load error.

Report these the same way as an unknown stage name, before any record is read.

**Modules without inspectable signatures** are not LM stages for this amendment. If they are
callable, they resolve as ordinary Python stages and the base spec applies. If they are not
callable, they are a load error. Do not add a second LM detection path until there is a concrete
module type that needs it.

## A6. Reduce field resolution

Reduce is the only position where a field name is ambiguous, because both a child list and a
parent record are in scope. Resolve by annotation rather than by new syntax.

- The single `list[T]`-typed input field receives the gathered children.
- Every other input field resolves against `ctx.parent`.

```python
class Merge(dspy.Signature):
    """Summarize labeled sessions from one trace."""
    trace: str = dspy.InputField()          # from the parent record
    labels: list[str] = dspy.InputField()   # gathered from children
    summary: str = dspy.OutputField()

merge = dspy.Predict(Merge)
```

Gathering follows base spec section 9. When `ordered: true`, children are sorted by emit path
before the field is assembled, so the list order is deterministic.

For a `by` group, there is no natural parent record. If `by` names a record field, provide a
synthetic parent mapping `{field_name: key}` to reduce field resolution. If `by` names a stage,
there is no parent field unless a later config form gives that key a name.

## A7. Config additions

The only new MVP config surface is provider and model selection. If `lm` is omitted, Trace Miner
uses:

```yaml
lm:
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
  max_concurrent: 64
  cache: false
```

The default model id is the OpenRouter slug for DeepSeek V4.1 Flash.

```yaml
lm:
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
  max_concurrent: 64

pipelines:
  - in: traces.jsonl
    out: overviews.jsonl
    pipeline:
      - filter: keep
      - map:
          stage: label
          lm: {model: openai/gpt-4o}
```

The expanded step form, a mapping with `stage:` plus options, must also work for Python stages.
Do not create two step grammars.

A per-step `lm.model` overrides the top-level `lm.default` or built-in default for that stage.

If any LM stage is used, a model is resolved from the per-step `lm.model`, top-level
`lm.default`, or built-in default, in that order. A per-step `lm` block on a Python stage is a
load error, because model selection only has meaning for LM stages. Providers other than
OpenRouter are post-MVP and fail at load time.

`lm.max_concurrent` bounds simultaneous LM provider calls. It defaults to 64 and must be a
positive integer.

`lm.cache` enables DSPy's LM request cache. It defaults to `false`. Set it to `true` to reuse
identical provider calls. This is not the post-MVP record-level pipeline cache.

The resolved model id is passed to the LM adapter/provider. It is not a signature input field
and is not visible to the module unless the adapter's backend chooses to bind it there.

Post-MVP, this `lm` block may grow policy fields such as:

```yaml
lm:
  provider: openrouter
  default: deepseek/deepseek-v4.1-flash
  retries: 3
  on_invalid: repair        # repair | fail
```

`max_concurrent` is a sub-limit within the eventual base spec's `max_in_flight` from section 7.
Until global backpressure exists, it only limits simultaneous LM provider calls. Reduce slots
are unaffected, and the deadlock analysis in section 7 holds unchanged, since an LM reduce still
draws from `reduce_slots` once that pool exists.

## A8. Failure handling

LM stages fail differently from Python stages. They produce malformed output rather than raising.

Handle this **inside the stage adapter**, not as a new failure path. In the LM MVP, invalid
output raises immediately from the adapter, at which point base spec section 8 applies unchanged.
Post-MVP, validation and repair may happen in the adapter before it raises.

- `on_invalid: repair` retries up to `retries` times, feeding the validation error back to the
  model. Then raises.
- `on_invalid: fail` raises on the first invalid output.

Either way the eventual exception flows through `on_child_error`. Do not add an LM-specific
failure policy. One failure path.

Retries are invisible to the barrier. A retrying stage has not emitted, so no count changes.

## A9. Caching (post-MVP)

The base spec's resumability in section 13 is per pipeline block, which is too coarse when a
rerun means paying for every call again. LM stages need record-level caching.

**The cache key must include the module's compiled state, not just the stage name.** Hash the
serialized module: signature fields, instructions, attached demonstrations, model, and
parameters, together with the resolved inputs.

This is the one item on this list that will actively waste your time if you get it wrong. If you
optimize a stage and the key does not move, you get outputs from the old prompt and a run that
looks like the optimization did nothing. Compute the module hash once at load time and log it.

Caching also makes base spec test 7, determinism, meaningful for LM pipelines. A fully cached
rerun must be byte identical even though the model is not deterministic.

## A10. Tracing and optimization (post-MVP)

Because stages are declarative modules rather than opaque functions, the pipeline can eventually
be treated as a program that can be traced or compiled. This is a reason for keeping signatures
declarative, but it should not be part of the LM MVP.

**Tracing.** Log every LM invocation with its inputs, outputs, stage name, module hash, emit
path, group id, and parent identity. The base spec's `ctx.path` and `ctx.group_id` already carry
what is needed. Structured this way, a run is a dataset rather than a log file.

**Optimization.** Deferred. Do not add an `optimize` verb in this amendment. If optimization
still feels useful after basic LM execution, tracing, and caching exist, write a separate
amendment for it.

## A11. Amends section 15, non-goals

Add:

- **Prompts in config.** Once prompt text is in YAML you will want templating, then conditionals,
  and you will have built a worse programming language inside a config file. Signatures and
  instructions live in the module.
- **Model-chosen routing.** Do not let an LM stage select its downstream destination. This is
  the `emit_to` problem from section 15 in a more appealing costume, and it breaks the property
  that config describes the flow.
- **A separate LM failure policy.** Covered by A8.
- **Optimization in the LM MVP.** Covered by A10.

## A12. Amends section 16, build order

Insert after the base MVP is correct. The barrier, ordering, nesting, and all three group
constructors should already be tested before LM stages enter the picture.

### LM MVP

1. LM module detection and field mapping, A2 and A4.
2. Minimal model config: built-in OpenRouter/DeepSeek defaults, top-level `lm.provider` and
   `lm.default`, and per-step `lm.model`.
3. Expanded step form, shared by Python and LM stages.
4. LM `map`, including passthrough, output merge, and missing-field errors.
5. Load-time signature validation for LM `filter`, `expand`, and `reduce`, A5.
6. LM `filter` and `expand`, adapted onto the existing emit-count behavior.
7. LM `reduce` field resolution, including synthetic parent fields for `by` groups, A6.
8. Bounded concurrent LM invocation with `lm.max_concurrent`.
9. Mixed Python and LM stages, with adapter-raised errors flowing through `on_child_error`.

### Post-MVP

1. Structured tracing, A10.
2. Caching with module-state hashing, A9.
3. `lm` config policy beyond provider/model selection, including retries and repair, A7 and A8.
4. Global concurrency/backpressure after async execution is stable.
5. Optimization only as a separate later amendment.

## A13. Tests

Additive to base spec section 17. Numbering continues from 27.

### LM MVP tests

**Field mapping**

28. **Passthrough.** A record with fields the signature does not mention survives an LM map with
    those fields intact.
29. **Merge, not replace.** Output fields are added to the record rather than replacing it.
30. **Missing input field.** Fails on the first record naming the field, the stage, and the
    record's actual keys, not with a raw `KeyError`.

**Load-time validation**

31. **Filter without bool output** is a load error.
32. **Filter with two bool outputs** is a load error.
33. **Expand without a list output** is a load error.
34. **Reduce without a list input** is a load error.
35. **Uninspectable non-callable module** is a load error rather than an LM stage.
36. **Built-in LM defaults.** A config that uses an LM stage without an `lm` block resolves to
    `provider: openrouter` and `default: deepseek/deepseek-v4.1-flash`.
37. **Unsupported provider.** A provider other than `openrouter` is a load error.
38. **Python stage with LM options.** A per-step `lm` block on a Python stage is a load error.

**Semantics**

39. **LM expand with variable output.** One trace yields 10 sessions, another yields 30. Both
    groups reduce correctly. This is base spec test 1 driven by a model rather than by code.
40. **LM reduce field resolution.** A parent field and a gathered child field both populate
    correctly.
41. **Gather order.** With `ordered: true`, the gathered list follows emit path order.
42. **Mixed pipeline.** Python and LM stages interleave in one pipeline and both work.
43. **Per-step model override.** The adapter receives the per-step model instead of the top-level
    default for that stage.
44. **OpenRouter adapter call.** A non-callable declarative LM stage invokes OpenRouter with the
    default model and JSON-object response format.
45. **`by` synthetic parent.** An LM reduce in a `by` group can receive the named grouping key as
    a parent-derived input field.

**Policy**

46. **Invalid output fails through the adapter.** A module missing a required output field raises
    with the stage name and missing field.
47. **Adapter error policy.** Invalid output flows through `on_child_error: partial`, reducing
    over survivors.

### Post-MVP tests

**Tracing**

48. **Invocation trace.** Every LM call records inputs, outputs, stage name, module hash, emit
    path, group id, and parent identity.

**Caching**

47. **Cache hit.** A rerun with an unchanged module makes zero API calls and produces byte
    identical output.
48. **Module state invalidation.** Changing only the instructions, or only the attached
    demonstrations, changes the cache key and forces a re-call. This is the test that prevents
    the A9 failure.
49. **Determinism.** Base spec test 7 passes for an LM pipeline under a warm cache.

**Policy**

50. **Repair.** A module producing invalid output twice then valid output succeeds, with no
    change to the group's barrier count.
51. **Repair exhaustion.** Retries exhausted raises and flows through `on_child_error: partial`,
    reducing over survivors.
52. **`max_concurrent`.** Never exceeded, and never exceeds the global `max_in_flight`.
