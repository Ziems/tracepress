"""Providers beyond the ones lm15 ships.

Portkey is an OpenAI-compatible gateway. Models are named ``@provider-slug/model``, where the
slug is a provider in your Portkey workspace (Model Catalog):

    tp.Model("portkey:@openai/gpt-5-nano", metadata={"engagement_name": "demo"})

Set ``PORTKEY_API_KEY``. ``PORTKEY_BASE_URL`` points at a self-hosted gateway; ``PORTKEY_METADATA``
(a JSON object) and ``PORTKEY_CONFIG`` set defaults for every Portkey model.

DSPy's native engine takes headers only from a registered provider (a per-call ``extra_headers``
would force the LiteLLM engine), so each distinct header set is registered as its own provider.
That id is part of DSPy's response-cache key: keep ``metadata`` stable within a run (no per-row run
ids), or identical prompts miss the cache and are billed again.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

from dspy.lm15 import AccessPolicy, OpenAIChatCompat, ProviderDefinition, register_provider

PORTKEY_BASE_URL = "https://api.portkey.ai/v1"


def _env_metadata() -> dict[str, Any] | None:
    raw = os.environ.get("PORTKEY_METADATA", "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"PORTKEY_METADATA is not valid JSON: {e}") from e
    if not isinstance(value, dict):
        raise ValueError(f"PORTKEY_METADATA must be a JSON object, got {type(value).__name__}")
    return value


def portkey_headers(
    *,
    metadata: Mapping[str, Any] | None = None,
    config: str | Mapping[str, Any] | None = None,
    virtual_key: str | None = None,
) -> dict[str, str]:
    """The ``x-portkey-*`` headers for these settings.

    ``None`` metadata and config fall back to ``PORTKEY_METADATA`` and ``PORTKEY_CONFIG``. Anything
    else, including ``{}`` or ``""``, is taken as given, so it can switch a default off.
    """
    metadata = metadata if metadata is not None else _env_metadata()
    if metadata and not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")
    metadata = {str(k): str(v) for k, v in (metadata or {}).items()}  # Portkey metadata values are strings
    config = config if config is not None else os.environ.get("PORTKEY_CONFIG")
    out: dict[str, str] = {}
    if metadata:
        out["x-portkey-metadata"] = json.dumps(dict(metadata), sort_keys=True)
    if config:
        out["x-portkey-config"] = config if isinstance(config, str) else json.dumps(dict(config), sort_keys=True)
    if virtual_key:
        out["x-portkey-virtual-key"] = virtual_key
    return out


def register_portkey(headers: Mapping[str, str] | None = None) -> str:
    """Declare a Portkey provider with these static headers and return its provider id.

    The id is ``portkey`` for the default gateway with no headers, else ``portkey-<hash>`` of the
    headers and base URL. Calling again with the same settings is a no-op.
    """
    headers = dict(headers or {})
    base_url = os.environ.get("PORTKEY_BASE_URL", "").strip() or PORTKEY_BASE_URL
    if headers or base_url != PORTKEY_BASE_URL:
        key = json.dumps([base_url, sorted(headers.items())])
        pid = f"portkey-{hashlib.blake2b(key.encode(), digest_size=8).hexdigest()}"
    else:
        pid = "portkey"
    access = AccessPolicy(
        provider=pid,
        env_keys=("PORTKEY_API_KEY",),
        base_url=base_url,
        headers=tuple(sorted(headers.items())),
    )
    register_provider(
        ProviderDefinition.chat(access, compat=OpenAIChatCompat(), console_url="https://app.portkey.ai/api-keys",
                                note="Portkey AI gateway"),
    )
    return pid


def portkey_model(
    model_id: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    config: str | Mapping[str, Any] | None = None,
    virtual_key: str | None = None,
) -> str:
    """The DSPy model string (``<provider id>/<model_id>``) for a Portkey ``@slug/model`` id."""
    pid = register_portkey(portkey_headers(metadata=metadata, config=config, virtual_key=virtual_key))
    return f"{pid}/{model_id}"
