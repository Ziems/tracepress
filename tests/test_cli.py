from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_miner.cli import main
from trace_miner.config import load_config
from trace_miner.home import load_stats


def read_jsonl(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_serve_rejects_conflicting_start_modes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "pipeline.yaml", "--from-start", "--from-end"])

    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def _command_help(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_root_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "init" in out
    assert "run" in out
    assert "serve" in out
    assert "stats" in out
    assert "stream" not in out
    assert "replay" not in out
    assert "trace-miner run pipeline.yaml" in out


def test_command_help_documents_usage(capsys: pytest.CaptureFixture[str]) -> None:
    init = _command_help(["init", "--help"], capsys)
    assert "hello-world" in init
    assert "--force" in init

    run = _command_help(["run", "--help"], capsys)
    assert "Pipeline YAML path" in run
    assert "--progress" in run

    serve = _command_help(["serve", "--help"], capsys)
    assert "Pipeline YAML path" in serve
    assert "--host" in serve
    assert "--port" in serve
    assert "--reset-checkpoint" in serve
    assert "--from-end" in serve

    stats = _command_help(["stats", "--help"], capsys)
    assert "~/.trace-miner" in stats


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "init" in capsys.readouterr().out


def test_help_does_not_create_home(isolate_trace_miner_home: Path) -> None:
    assert main(["--help"]) == 0
    assert not isolate_trace_miner_home.exists()


def test_init_creates_home_and_records_command(
    tmp_path: Path, isolate_trace_miner_home: Path
) -> None:
    dest = tmp_path / "demo"
    assert main(["init", str(dest)]) == 0
    assert isolate_trace_miner_home.is_dir()
    assert (isolate_trace_miner_home / "cache").is_dir()
    stats = load_stats()
    assert stats["commands"]["init"] == 1
    assert stats["runs"] == 0


def test_init_creates_hello_world_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "demo"
    assert main(["init", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "Created a hello-world project" in out
    assert "trace-miner run summarize.yaml" in out
    assert "pii.yaml" in out
    assert "alerts.yaml" in out
    assert "trace-miner serve alerts.yaml" in out
    assert "python client_demo.py" in out
    assert "AGENTS.md" in out
    assert "stream.yaml" not in out
    assert "trace-miner replay" not in out
    assert (dest / "summarize.yaml").is_file()
    assert (dest / "pii.yaml").is_file()
    assert (dest / "frustration.yaml").is_file()
    assert (dest / "category.yaml").is_file()
    assert (dest / "alerts.yaml").is_file()
    assert not (dest / "stream.yaml").exists()
    assert (dest / "client_demo.py").is_file()
    assert (dest / "stages.py").is_file()
    assert (dest / "AGENTS.md").is_file()
    assert (dest / "input" / "trajectories.jsonl").is_file()
    assert not (dest / "input" / "live.jsonl").exists()
    assert (dest / "input" / "alerts.jsonl").read_text(encoding="utf-8") == ""
    assert not (dest / "simulate_stream.py").exists()
    agents = (dest / "AGENTS.md").read_text(encoding="utf-8")
    assert "trace-miner run summarize.yaml" in agents
    assert "trace-miner serve alerts.yaml" in agents
    assert "trace-miner replay" not in agents
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

    for name in ("summarize.yaml", "pii.yaml", "frustration.yaml", "category.yaml", "alerts.yaml"):
        config = load_config(dest / name)
        assert config.pipelines[0].output.name.endswith(".jsonl")
    alerts = load_config(dest / "alerts.yaml")
    assert alerts.pipelines[0].name == "alerts"


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


def test_bare_config_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    isolate_trace_miner_home: Path,
) -> None:
    config = tmp_path / "summarize.yaml"
    config.write_text("pipelines: []\n", encoding="utf-8")

    assert main([str(config)]) == 2
    captured = capsys.readouterr()
    assert "unknown command" in captured.err
    assert "trace-miner run" in captured.err
    assert not isolate_trace_miner_home.exists()
    assert not (tmp_path / "output").exists()


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

    assert main(["run", "pipeline.yaml"]) == 0
    out = capsys.readouterr().out
    assert "Wrote output/findings.jsonl" in out
    assert "LM calls" not in out
    assert any(line.endswith("s") and "Wrote" not in line for line in out.splitlines())


def test_batch_does_not_print_output_on_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "pipeline.yaml"
    config.write_text("pipelines: []\n", encoding="utf-8")
    assert main(["run", str(config)]) == 1
    captured = capsys.readouterr()
    assert "Wrote " not in captured.out
    assert "trace-miner:" in captured.err

