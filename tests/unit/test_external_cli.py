import asyncio
import json
import os
import subprocess
import sys
import time

import pytest

from agent.core import external_cli


def test_builds_codex_command(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_INTERN_CODEX_COMMAND", "codex")
    cmd = external_cli.build_external_cli_command(
        external_cli.CODEX_MODEL_ID,
        "fix the tests",
        str(tmp_path),
        tmp_path / "last.txt",
    )

    assert cmd[0] == "codex"
    assert (
        f'mcp_servers.{external_cli.HF_MCP_SERVER_NAME}.url="{external_cli.HF_MCP_SERVER_URL}"'
        in cmd
    )
    assert (
        f'mcp_servers.{external_cli.GITHUB_MCP_SERVER_NAME}.url="{external_cli.GITHUB_MCP_SERVER_URL}"'
        in cmd
    )
    assert "--ask-for-approval" in cmd
    assert "never" in cmd
    assert "--sandbox" in cmd
    assert "workspace-write" in cmd
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
    assert "--enable-all-github-mcp-tools" in cmd
    assert "--add-dir" in cmd
    assert str(tmp_path) in cmd
    config = json.loads(cmd[cmd.index("--additional-mcp-config") + 1])
    assert config["mcpServers"][external_cli.HF_MCP_SERVER_NAME] == {
        "type": "http",
        "url": external_cli.HF_MCP_SERVER_URL,
        "tools": ["*"],
    }


def test_external_cli_prompt_includes_backend_instructions():
    prompt = external_cli.render_external_cli_prompt(
        [{"role": "user", "content": "launch the experiment"}],
        external_cli.COPILOT_MODEL_ID,
    )

    assert "scripts/run_gcp_experiment.py" in prompt
    assert "Do not run raw gcloud commands directly." in prompt
    assert "built-in GitHub MCP server is enabled" in prompt
    assert external_cli.HF_MCP_SERVER_NAME in prompt


def test_external_cli_reads_codex_last_message(monkeypatch, tmp_path):
    def fake_command(model_name, prompt, cwd, output_path):
        code = (
            "from pathlib import Path; "
            "import sys; "
            "Path(sys.argv[1]).write_text('done', encoding='utf-8'); "
            "print('progress')"
        )
        return [sys.executable, "-c", code, str(output_path)]

    monkeypatch.setattr(external_cli, "build_external_cli_command", fake_command)

    result = asyncio.run(
        external_cli._run_external_cli(
            external_cli.CODEX_MODEL_ID,
            "hello",
            str(tmp_path),
        )
    )

    assert result.success is True
    assert result.content == "done"
    assert result.stdout.strip() == "progress"


def test_external_cli_failure_reports_stderr(monkeypatch, tmp_path):
    def fake_command(model_name, prompt, cwd, output_path):
        code = "import sys; print('failed', file=sys.stderr); raise SystemExit(2)"
        return [sys.executable, "-c", code]

    monkeypatch.setattr(external_cli, "build_external_cli_command", fake_command)

    result = asyncio.run(
        external_cli._run_external_cli(
            external_cli.COPILOT_MODEL_ID,
            "hello",
            str(tmp_path),
        )
    )

    assert result.success is False
    assert result.stderr.strip() == "failed"
    assert result.exit_code == 2


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            check=False,
            capture_output=True,
            text=True,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_external_cli_cancellation_kills_subprocess(monkeypatch, tmp_path):
    pid_path = tmp_path / "external-cli.pid"

    def fake_command(model_name, prompt, cwd, output_path):
        code = (
            "from pathlib import Path; "
            "import os, sys, time; "
            "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8'); "
            "time.sleep(60)"
        )
        return [sys.executable, "-c", code, str(pid_path)]

    async def body():
        monkeypatch.setattr(external_cli, "build_external_cli_command", fake_command)
        task = asyncio.create_task(
            external_cli._run_external_cli(
                external_cli.CODEX_MODEL_ID,
                "hello",
                str(tmp_path),
            )
        )
        for _ in range(100):
            if pid_path.exists():
                break
            await asyncio.sleep(0.02)
        assert pid_path.exists()
        pid = int(pid_path.read_text(encoding="utf-8"))

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        for _ in range(100):
            if not _pid_exists(pid):
                return
            time.sleep(0.02)
        pytest.fail(f"external CLI subprocess {pid} was still running after cancellation")

    asyncio.run(body())
