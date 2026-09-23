"""DSPy modules that answer typed outputs with Jev on OpenRouter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

import dspy
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, SystemOneResponse, TypeSafeClient, Usage

from trace_miner.errors import PipelineExecutionError
from trace_miner.model import LmUsage


OPENROUTER_JEV_BASE_URL = "https://openrouter.ai/api"
DEFAULT_JEV_MODEL = "jev-1.13"
DEFAULT_JEV_TIMEOUT = 60.0
DEFAULT_JEV_RETRIES = 1
DEFAULT_JEV_RETRY_BUDGET = 130.0
JEV_INPUT_USD_PER_MILLION = 0.042
NOUL_TRUE_THRESHOLD = 0.5
_MIN_CHOICES = 2
_MAX_CHOICES = 255
_MIN_SCORE_LEVELS = 2
_MAX_SCORE_LEVELS = 10

QuestionKind = Literal["noul", "choice", "score"]


class JevUsage(Usage):
    cost: float | None = None


class JevSystemOneResponse(SystemOneResponse):
    usage: JevUsage


@dataclass(frozen=True)
class _OutputPlan:
    name: str
    kind: QuestionKind
    choice_values: dict[str, Any]
    annotation: Any


class JevPredict(dspy.Predict):
    """Answer bool, Literal, and score outputs with one Jev request.

    Freeform strings and lists stay on ``dspy.Predict``. Jev is called through
    OpenRouter's System One endpoint with ``OPENROUTER_API_KEY``.
    """

    uses_jev = True

    def __init__(
        self,
        signature: type[dspy.Signature] | str,
        *,
        model: str = DEFAULT_JEV_MODEL,
        score_fields: dict[str, list[str]] | None = None,
        noul_true_threshold: float = NOUL_TRUE_THRESHOLD,
        client: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Jev model must be a non-empty string.")
        super().__init__(signature)
        self.model = model
        self.noul_true_threshold = noul_true_threshold
        self._client = client
        self._plans = _plan_outputs(self.signature, score_fields or {})
        self._questions = _questions(self.signature, self._plans, score_fields or {})

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        inputs = {name: kwargs[name] for name in self.signature.input_fields if name in kwargs}
        state = {
            "signature": self.signature.__name__,
            "instructions": self.signature.instructions,
            "inputs": _jsonable(inputs),
        }
        response = self._system_one(state)
        outputs = _read_outputs(response, self._plans, self.noul_true_threshold)
        prediction = dspy.Prediction(**outputs)
        prediction._jev_usage = _usage_from_response(response)
        return prediction

    def _system_one(self, state: dict[str, Any]) -> Any:
        if self._client is not None:
            return self._client.system_one(
                state,
                self._questions,
                model=self.model,
                response_model=JevSystemOneResponse,
            )
        with _openrouter_client(self.model) as client:
            return client.system_one(
                state,
                self._questions,
                model=self.model,
                response_model=JevSystemOneResponse,
            )


def _openrouter_client(model: str) -> TypeSafeClient:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise PipelineExecutionError("OPENROUTER_API_KEY is required for Jev stages.")
    return TypeSafeClient(
        api_key=api_key,
        base_url=OPENROUTER_JEV_BASE_URL,
        model=model,
        timeout=DEFAULT_JEV_TIMEOUT,
        retry=RetryPolicy(max_retries=DEFAULT_JEV_RETRIES, timeout=DEFAULT_JEV_RETRY_BUDGET),
    )


def _plan_outputs(signature: type[dspy.Signature], score_fields: dict[str, list[str]]) -> tuple[_OutputPlan, ...]:
    output_names = set(signature.output_fields)
    unknown = sorted(set(score_fields) - output_names)
    if unknown:
        raise ValueError(f"Jev score field `{unknown[0]}` is not an output of `{signature.__name__}`.")
    plans: list[_OutputPlan] = []
    for name, field in signature.output_fields.items():
        plans.append(_plan_output(signature.__name__, name, field, score_fields.get(name)))
    return tuple(plans)


def _plan_output(signature_name: str, name: str, field: Any, rubric: list[str] | None) -> _OutputPlan:
    annotation = field.annotation
    if rubric is not None:
        if annotation not in (int, float):
            raise ValueError(f"Jev score field `{name}` on `{signature_name}` must be an int or float.")
        _validate_rubric(name, rubric)
        return _OutputPlan(name, "score", {}, annotation)
    if annotation is bool:
        return _OutputPlan(name, "noul", {}, annotation)
    choice_values = _literal_choice_values(annotation)
    if choice_values is not None:
        count = len(choice_values)
        if count < _MIN_CHOICES or count > _MAX_CHOICES:
            raise ValueError(
                f"Jev output `{name}` on `{signature_name}` must have between {_MIN_CHOICES} and {_MAX_CHOICES} options."
            )
        return _OutputPlan(name, "choice", choice_values, annotation)
    raise ValueError(
        f"Jev output `{name}` on `{signature_name}` must be a bool, a Literal, or a score field."
    )


def _validate_rubric(name: str, rubric: list[str]) -> None:
    if (
        not isinstance(rubric, list)
        or not _MIN_SCORE_LEVELS <= len(rubric) <= _MAX_SCORE_LEVELS
        or not all(isinstance(level, str) and level for level in rubric)
    ):
        raise ValueError(f"Jev score field `{name}` needs {_MIN_SCORE_LEVELS} to {_MAX_SCORE_LEVELS} level descriptions.")


def _literal_choice_values(annotation: Any) -> dict[str, Any] | None:
    if get_origin(annotation) is not Literal:
        return None
    values = get_args(annotation)
    if not values or any(not isinstance(value, str | int | float) for value in values):
        return None
    choices: dict[str, Any] = {}
    for value in values:
        key = value if isinstance(value, str) else str(value)
        if key in choices:
            raise ValueError(f"Jev choice option `{key}` is ambiguous.")
        choices[key] = value
    return choices


def _questions(
    signature: type[dspy.Signature],
    plans: tuple[_OutputPlan, ...],
    score_fields: dict[str, list[str]],
) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for plan in plans:
        field = signature.output_fields[plan.name]
        instructions = _instructions(signature, plan.name, field)
        if plan.kind == "noul":
            questions[plan.name] = Noul(instructions=instructions)
        elif plan.kind == "choice":
            description = _field_desc(field)
            criteria = (
                {key: description for key in plan.choice_values}
                if description
                else dict.fromkeys(plan.choice_values)
            )
            questions[plan.name] = Choice(instructions=instructions, criteria=criteria)
        else:
            questions[plan.name] = Score(instructions=instructions, criteria=list(score_fields[plan.name]))
    return questions


def _instructions(signature: type[dspy.Signature], name: str, field: Any) -> dict[str, str]:
    instructions = {"task": signature.instructions, "field": name}
    description = _field_desc(field)
    if description:
        instructions["description"] = description
    return instructions


def _field_desc(field: Any) -> str | None:
    extra = getattr(field, "json_schema_extra", None) or {}
    description = extra.get("desc") if isinstance(extra, dict) else None
    if not isinstance(description, str) or not description or description.startswith("${"):
        return None
    return description


def _read_outputs(response: Any, plans: tuple[_OutputPlan, ...], threshold: float) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for plan in plans:
        if plan.kind == "noul":
            answer = _answer(response, "nouls", plan.name)
            outputs[plan.name] = float(answer.noul) >= threshold
        elif plan.kind == "choice":
            answer = _answer(response, "choices", plan.name)
            key = answer.choice
            if key not in plan.choice_values:
                raise PipelineExecutionError(f"Jev choice `{plan.name}` returned unknown option `{key}`.")
            outputs[plan.name] = plan.choice_values[key]
        else:
            answer = _answer(response, "scores", plan.name)
            score = float(answer.score)
            outputs[plan.name] = int(round(score)) if plan.annotation is int else score
    return outputs


def _answer(response: Any, group: str, name: str) -> Any:
    answers = getattr(response, group, None) or {}
    answer = answers.get(name) if isinstance(answers, dict) else None
    if answer is None:
        raise PipelineExecutionError(f"Jev response missing `{name}`.")
    return answer


def _usage_from_response(response: Any) -> LmUsage:
    usage = getattr(response, "usage", None)
    prompt = _as_int(getattr(usage, "input_tokens", None))
    completion = _as_int(getattr(usage, "output_tokens", None))
    cost = getattr(usage, "cost", None)
    if cost is None and isinstance(usage, dict):
        cost = usage.get("cost")
        prompt = prompt or _as_int(usage.get("input_tokens"))
        completion = completion or _as_int(usage.get("output_tokens"))
    if cost is None:
        cost = prompt * JEV_INPUT_USD_PER_MILLION / 1_000_000
    return LmUsage(prompt_tokens=prompt, completion_tokens=completion, cost_usd=float(cost))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return str(value)


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)
