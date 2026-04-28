import subprocess
from pathlib import Path

from agent.core import external_cli


def test_builds_codex_command(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_INTERN_CODEX_COMMAND", "codex")
    cmd = external_cli.build_external_cli_command(
        external_cli.CODEX_MODEL_ID,
        "fix the tests",
        str(tmp_path),
        tmp_path / "last.txt",
    )

    assert cmd[:5] == ["codex", "--ask-for-approval", "never", "--sandbox", "workspace-write"]
    assert "exec" in cmd
    assert "-C" in cmd
    assert str(tmp_path) in cmd
    assert cmd[-1] == "fix the tests"


def test_builds_copilot_command(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_INTERN_COPILOT_COMMAND", "copilot")
    cmd = external_cli.build_external_cli_command(
        external_cli.COPILOT_MODEL_ID,
        "fix the tests",
        str(tmp_path),
        tmp_path / "unused.txt",
    )

    assert cmd[:3] == ["copilot", "-p", "fix the tests"]
    assert "--allow-all" in cmd
    assert "--add-dir" in cmd
    assert str(tmp_path) in cmd


def test_external_cli_reads_codex_last_message(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text("done", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="progress", stderr="")

    monkeypatch.setattr(external_cli.subprocess, "run", fake_run)

    result = external_cli._run_external_cli_sync(
        external_cli.CODEX_MODEL_ID,
        "hello",
        str(tmp_path),
    )

    assert result.success is True
    assert result.content == "done"
    assert result.command_invoked[-1] == "<prompt>"


def test_external_cli_failure_reports_stderr(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="failed")

    monkeypatch.setattr(external_cli.subprocess, "run", fake_run)

    result = external_cli._run_external_cli_sync(
        external_cli.COPILOT_MODEL_ID,
        "hello",
        str(tmp_path),
    )

    assert result.success is False
    assert result.stderr == "failed"
    assert result.exit_code == 2
