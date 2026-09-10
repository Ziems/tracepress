class MineTraceError(Exception):
    """Base exception for Mine Trace."""


class ConfigError(MineTraceError):
    """Raised when a config cannot be validated before execution."""


class PipelineExecutionError(MineTraceError):
    """Raised when a pipeline fails while processing records."""

