from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_miner.cli import main
from trace_miner.config import load_config


def read_jsonl(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_stream_rejects_conflicting_start_modes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["stream", "pipeline.yaml", "--from-start", "--from-end"])

    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_root_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "init" in out
    assert "run" in out
    assert "stream" in out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "init" in capsys.readouterr().out


def test_init_creates_hello_world_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "demo"
    assert main(["init", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "Created a hello-world project" in out
    assert "trace-miner summarize.yaml" in out
    assert "pii.yaml" in out
    assert "AGENTS.md" in out
    assert (dest / "summarize.yaml").is_file()
    assert (dest / "pii.yaml").is_file()
    assert (dest / "frustration.yaml").is_file()
    assert (dest / "category.yaml").is_file()
    assert (dest / "stages.py").is_file()
    assert (dest / "AGENTS.md").is_file()
    assert (dest / "input" / "trajectories.jsonl").is_file()
    agents = (dest / "AGENTS.md").read_text(encoding="utf-8")
    assert "trace-miner summarize.yaml" in agents
    assert "stages.py" in agents

    traces = (dest / "input" / "trajectories.jsonl").read_text(encoding="utf-8")
    assert "123-45-6789" in traces
    assert "(555) 010-1234" in traces
    assert "This is taking forever" in traces
    records = read_jsonl(dest / "input" / "trajectories.jsonl")
    assert [row["eval"]["task_name"] for row in records] == [
        "terminal-bench/atrx-vep-crispr",
        "terminal-bench/photonic-waveguide-routing",
        "terminal-bench/freecad-spring-clip",
    ]

    for name in ("summarize.yaml", "pii.yaml", "frustration.yaml", "category.yaml"):
        config = load_config(dest / name)
        assert config.pipelines[0].output.name.endswith(".jsonl")


def test_init_refuses_to_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "demo"
    assert main(["init", str(dest)]) == 0
    capsys.readouterr()
    assert main(["init", str(dest)]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err


def test_init_force_overwrites(tmp_path: Path) -> None:
    dest = tmp_path / "demo"
    assert main(["init", str(dest)]) == 0
    (dest / "summarize.yaml").write_text("broken\n", encoding="utf-8")
    assert main(["init", str(dest), "--force"]) == 0
    assert "from:" in (dest / "summarize.yaml").read_text(encoding="utf-8")


def test_init_prompts_for_project_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "demo")

    assert main(["init"]) == 0
    out = capsys.readouterr().out
    assert "Created a hello-world project in demo/" in out
    assert (tmp_path / "demo" / "summarize.yaml").is_file()
    assert (tmp_path / "demo" / "stages.py").is_file()


def test_init_uses_hello_world_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert main(["init"]) == 0
    out = capsys.readouterr().out
    assert "Created a hello-world project in hello-world/" in out
    assert (tmp_path / "hello-world" / "summarize.yaml").is_file()


def test_init_reprompts_until_name_is_valid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter([".", "demo"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert main(["init"]) == 0
    assert "must be a new directory" in capsys.readouterr().err
    assert (tmp_path / "demo" / "summarize.yaml").is_file()


def test_init_uses_hello_world_when_stdin_is_not_a_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    assert main(["init"]) == 0
    assert (tmp_path / "hello-world" / "summarize.yaml").is_file()


def test_batch_prints_output_path(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "stages.py").write_text("def keep(record):\n    return record\n", encoding="utf-8")
    (tmp_path / "input.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
    (tmp_path / "pipeline.yaml").write_text(
        """
from: ./stages
pipelines:
  - in: input.jsonl
    out: output/findings.jsonl
    pipeline:
      - map: keep
""",
        encoding="utf-8",
    )

    assert main(["pipeline.yaml"]) == 0
    out = capsys.readouterr().out
    assert "Wrote output/findings.jsonl" in out
    assert "LM calls" not in out
    assert any(line.endswith("s") and "Wrote" not in line for line in out.splitlines())


def test_batch_does_not_print_output_on_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "pipeline.yaml"
    config.write_text("pipelines: []\n", encoding="utf-8")
    assert main([str(config)]) == 1
    captured = capsys.readouterr()
    assert "Wrote " not in captured.out
    assert "trace-miner:" in captured.err
