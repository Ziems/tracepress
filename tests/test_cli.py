from __future__ import annotations

import pytest

from mine_trace.cli import main


def test_stream_rejects_conflicting_start_modes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["stream", "pipeline.yaml", "--from-start", "--from-end"])

    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
