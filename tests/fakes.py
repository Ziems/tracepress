"""A fake LM whose answers are computed from the prompt's input fields."""

import re

import dspy
import litellm

FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n(?:\[\[ ## |Respond with)|\Z)", re.S)


class FuncLM(dspy.BaseLM):
    """`fn(fields) -> outputs`, where `fields` maps each input field name to its rendered text.

    Speaks DSPy's chat format, so use it with `tp.configure(adapter=dspy.ChatAdapter())`
    (the test suite does this for every test).
    """

    def __init__(self, fn):
        super().__init__(model="func")
        self.fn = fn
        self.calls = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls += 1
        fields = dict(FIELD.findall(messages[-1]["content"]))
        out = self.fn(fields)
        content = "".join(f"[[ ## {k} ## ]]\n{v}\n\n" for k, v in out.items()) + "[[ ## completed ## ]]"
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": content}}],
            model="func",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)
