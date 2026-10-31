# Models

TracePress has no default model. Every LM operator needs one, and you pick it in one of
three ways:

```python
tp.configure(lm="openrouter:openai/gpt-6-luna")         # for everything
notes = steps.lm_map("step_text -> summary", lm="...")   # for one operator
```

or by setting the `TRACEPRESS_LM` environment variable. A run without a model fails before it
makes any calls.

`openrouter:openai/gpt-6-luna` is the suggested model. It's cheap (about $0.10 per million input
tokens and $0.50 per million output tokens) and works well for trace analysis.

## Model strings

Model strings have the form `provider:model`.

| String | Goes to | Needs |
|---|---|---|
| `openrouter:openai/gpt-6-luna` | OpenRouter | `OPENROUTER_API_KEY` |
| `portkey:openai-prod/gpt-5` | Portkey's gateway, model `@openai-prod/gpt-5` from your Model Catalog | `PORTKEY_API_KEY` |
| `typesafe:jev-latest` | TypeSafe's Jev, recommended for classification-only stages (see [Jev](#jev)) | `TYPESAFE_API_KEY` |
| `anthropic/claude-opus-5` (no `provider:`) | litellm, as is | that provider's usual key |

- **Portkey:** the part before the `/` is your Model Catalog provider slug. Writing
  `portkey:@openai-prod/gpt-5` means the same thing. Set `PORTKEY_BASE_URL` to use a
  self-hosted gateway.
- **Colons in model names:** a colon after a `/` is part of the model name, so strings like
  `ollama_chat/llama3:8b` pass straight through to litellm.
- **Other setups:** pass a `dspy.LM` object directly.

Keys are read from the environment, or from a `.env` file in the working directory.

## Jev

**Jev is the recommended model for any stage where classification alone is enough:** flagging
steps, picking an agent or an error category, scoring severity, filtering rows. Use a generative
model only for stages that must write text.

Jev, TypeSafe's System One model, doesn't generate text. It answers decision outputs (a `Noul`
yes/no, a `Choice` among fixed options, or a `Score`) with probabilities. That makes it much
cheaper (about $0.042 per million input tokens) and faster than a generative model, with no
output to parse. On the Who&When Pro failure-attribution benchmark, one Jev call per trace
matched a GPT-6-luna judge's accuracy for about a third of the cost ($0.34 vs. $1.06 per 1,000
traces) in about 7 seconds per 1,000 traces instead of 3.5 minutes.

Install the extra and use it in any operator whose signature's outputs are all decisions:

```bash
pip install "tracepress[jev]"
```

```python
from typing import Annotated, Literal

import dspy
from dspy.experimental import Choice, Noul

class Triage(dspy.Signature):
    """Judge one step of an agent's trace."""
    step_text: str = dspy.InputField()
    failed: Noul = dspy.OutputField(desc="Did this step fail?")
    kind: Annotated[Literal["tool", "reasoning", "planning"], Choice] = dspy.OutputField(
        desc="What kind of mistake it is, if any")

judged = steps.lm_map(Triage, lm="typesafe:jev-latest")
```

Every decision output needs a `desc`; Jev reads it as the question. Text outputs (`str`) need a
generative model, so pipelines that need both mix them per operator: a generative `lm_map` that
writes notes, then a Jev stage that decides from them, or the other way around. If a stage asks
for a free-text `reason` only to justify its label, consider dropping it and running the stage
on Jev; Jev's probabilities say how sure it is.

## Local models

Any locally hosted model works, so your traces never leave your machine.

- **Ollama:** use its litellm string, e.g. `tp.configure(lm="ollama_chat/llama3.1")`. It talks to
  Ollama on `localhost:11434` by default.
- **vLLM, llama.cpp, LM Studio, or any other OpenAI-compatible server:** point a `dspy.LM` at it:

  ```python
  import dspy

  tp.configure(lm=dspy.LM("openai/my-model", api_base="http://localhost:8000/v1", api_key="local"))
  ```

Local models have no price, so their calls show up as `unpriced_calls` in `tp.last_run()`. Add a
price with `tp.configure(prices=...)` if you want a cost estimate.

## Cost and caching

Every call goes through DSPy's disk cache, so rerunning a pipeline only pays for prompts that
changed. After a run, `tp.last_run()` shows what it did:

```python
>>> tp.last_run()
RunStats(calls=186, cached=7, errors=0, fallbacks=0, timeouts=0, prompt_tokens=132982,
         completion_tokens=39700, cost=$0.033148, seconds=37.3)
```

`by_model` and `by_op` break the totals down by model and by operator. `latency()` gives
per-call percentiles.

Cost is estimated from token counts. Prices come from litellm's price table, plus a built-in
entry for the suggested model. Add your own prices, in USD per million input and output tokens,
keyed by the litellm model name:

```python
tp.configure(prices={"openai/my-model": (0.50, 2.00)})
```

Calls to models with no known price are counted as `unpriced_calls` instead of being guessed.
