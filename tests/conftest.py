from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_trace_miner_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path_factory.mktemp("trace-miner-homes") / ".trace-miner"
    monkeypatch.setenv("TRACE_MINER_HOME", str(home))
    monkeypatch.delenv("DSPY_CACHEDIR", raising=False)
    return home
