"""Load Harbor trajectories and classify them with the Terminal-Bench taxonomies.

    loaded = tracepress.harbor.load("jobs/my-job")
    labels = tracepress.harbor.terminal_agent_taxonomy(loaded.steps)
    commands = tracepress.harbor.command_taxonomy(loaded.steps)
"""

from .load import HarborLoad, load
from .taxonomy import (
    COMMAND_CLASS,
    COMMAND_SUBCATEGORIES,
    TAT_CLASS,
    TAT_MODES,
    command_taxonomy,
    register,
    terminal_agent_taxonomy,
)

__all__ = [
    "HarborLoad",
    "load",
    "terminal_agent_taxonomy",
    "command_taxonomy",
    "register",
    "TAT_MODES",
    "TAT_CLASS",
    "COMMAND_SUBCATEGORIES",
    "COMMAND_CLASS",
]
