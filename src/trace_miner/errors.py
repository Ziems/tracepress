class TraceMinerError(Exception):
    """Base exception for Trace Miner."""


class ConfigError(TraceMinerError):
    """Raised when a config cannot be validated before execution."""


class PipelineExecutionError(TraceMinerError):
    """Raised when a pipeline fails while processing records."""

