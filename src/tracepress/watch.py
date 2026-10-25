"""Live sources: `tp.watch.otlp(port=4318)`. For stored runs played back live, see `tp.replay`."""

from .adapters import otlp

__all__ = ["otlp"]
