# Amendment: Jev Modules

An amendment to [`lm-pipeline.md`](lm-pipeline.md). A stage may be a `JevPredict` module. YAML still names the module. The runtime calls that module directly and does not wrap it in `dspy.LM`.

Jev answers a bool, a choice, or a score in one request. It does not write summaries or quotes. Those outputs stay on `dspy.Predict`.

## Authoring

```python
from typing import Literal

import dspy

from trace_miner.jev import JevPredict


class CategorizeTrajectory(dspy.Signature):
    """Categorize one trajectory."""

    task: str = dspy.InputField()
    reported_passed: bool = dspy.InputField()
    domain: Literal["bioinformatics", "systems", "data_processing", "other"] = dspy.OutputField()
    failed: bool = dspy.OutputField()


categorize_trajectory = JevPredict(CategorizeTrajectory)
```

```yaml
- map: categorize_trajectory
```

Output fields are planned when the module is constructed:

- `bool` is a yes/no probability, true at `0.5` or above.
- `Literal[...]` of strings, ints, floats, or bools is a choice. The field `desc` is the option text when it is set.
- An `int` or `float` listed in `score_fields` is a score on that ordered rubric. The rubric is 2 to 10 level descriptions, low end first.

```python
score = JevPredict(Triage, score_fields={"severity": ["can wait", "today", "right now"]})
```

Any other output, including a freeform `str` or a `list`, is an error naming the field.

`JevPredict` sends every typed question in one `system_one` call. The state is the signature instructions plus the input values. Declared outputs merge into the record. Probabilities are not extra fields.

## OpenRouter

Jev is called through OpenRouter's System One API, [`typesafe/jev-1.13`](https://openrouter.ai/typesafe/jev-1.13). The client uses `OPENROUTER_API_KEY` and `https://openrouter.ai/api`, which posts to `/v1/systemone`. The default model is `jev-1.13`. Pass `model="jev-latest"` only if you want OpenRouter's moving alias. This is not a chat completion.

A missing `OPENROUTER_API_KEY` fails at run start when any stage is a Jev module. The run profile counts the call, the input tokens, and the cost. OpenRouter's `usage.cost` is used when the response includes it. Otherwise input tokens are priced at $0.042 per million, and output tokens are unpriced.

A Jev stage does not take `lm.provider` or `lm.model`. A per-step `lm:` block on one is a load error. `map`, `filter`, `alert`, and `reduce` use the same signature rules as other LM stages. `expand` is a load error, because Jev cannot emit a list.

`lm.max_concurrent` also bounds simultaneous Jev calls.
