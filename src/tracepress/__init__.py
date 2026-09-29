"""TracePress: LLM operators on DataFrames, registered pipelines, and an agent over their artifacts.

    import tracepress as tp
    tp.configure(lm=tp.Model("openrouter:openai/gpt-6-luna", temperature=0))

    df.lm_map("text -> summary", "Summarize in one sentence")
    df.lm_filter("summary", "Describes a timeout")
    df.lm_agg("summary -> common_cause", "What do these have in common?", by="task")

    ws = tp.Workspace("ws")
    @ws.pipeline(description="...")
    def my_pipeline(df): ...
    ws.ask("...")
"""

from . import ops  # registers the df.lm_* accessors
from .model import Budget, BudgetExceeded, Model, Usage, budget
from .signature import In, Out, Signature
from .settings import configure, using
from .workspace import Pipeline, Run, Workspace

__all__ = [
    "Model",
    "Usage",
    "Budget",
    "BudgetExceeded",
    "budget",
    "configure",
    "using",
    "Workspace",
    "Pipeline",
    "Run",
    "Signature",
    "In",
    "Out",
]
