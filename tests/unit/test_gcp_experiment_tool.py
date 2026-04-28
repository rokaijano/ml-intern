import asyncio
import json
import subprocess

import pytest

from agent.tools.gcp_experiment_tool import run_gcp_experiment_handler


@pytest.fixture(autouse=True)
def _clear_gcp_env(monkeypatch):
    monkeypatch.delenv("ML_INTERN_GCP_MAX_MINUTES", raising=False)
    monkeypatch.delenv("ML_INTERN_GCP_ALLOW_ANY_BRANCH", raising=False)
    monkeypatch.delenv("ML_INTERN_GCP_SINGLE_FLIGHT", raising=False)
    monkeypatch.delenv("ML_INTERN_GCP_LOCK_PATH", raising=False)


def _run_tool(args):
    output, ok = asyncio.run(run_gcp_experiment_handler(args))
    return json.loads(output), ok


def _make_repo(tmp_path, with_script=True):
    repo = tmp_path / "research"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    if with_script:
        (scripts / "run_gcp_experiment.py").write_text("print('wrapper')\n")
    return repo


def _args(repo):
    return {
        "repo_path": str(repo),
        "branch": "exp/test-run",
        "minutes": 10,
        "command": "python train.py --steps 2",
    }


def test_rejects_too_many_minutes(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("ML_INTERN_GCP_MAX_MINUTES", "5")

    args = _args(repo)
    args["minutes"] = 6
    payload, ok = _run_tool(args)

    assert ok is False
    assert payload["success"] is False
    assert "ML_INTERN_GCP_MAX_MINUTES (5)" in payload["stderr"]


def test_rejects_non_exp_branch_by_default(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.delenv("ML_INTERN_GCP_ALLOW_ANY_BRANCH", raising=False)

    args = _args(repo)
    args["branch"] = "main"
    payload, ok = _run_tool(args)

    assert ok is False
    assert payload["success"] is False
    assert "branch must start with 'exp/'" in payload["stderr"]


def test_rejects_missing_script(tmp_path):
    repo = _make_repo(tmp_path, with_script=False)

    payload, ok = _run_tool(_args(repo))

    assert ok is False
    assert payload["success"] is False
    assert "scripts/run_gcp_experiment.py" in payload["stderr"]


def test_invokes_subprocess_without_shell_true(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("agent.tools.gcp_experiment_tool.subprocess.run", fake_run)

    payload, ok = _run_tool(_args(repo))

    assert ok is True
    assert payload["success"] is True
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == str(repo)
    assert kwargs["timeout"] == 10 * 60 + 600
    assert cmd == [
        "python",
        "scripts/run_gcp_experiment.py",
        "--branch",
        "exp/test-run",
        "--minutes",
        "10",
        "--command",
        "python train.py --steps 2",
    ]
    assert payload["command_invoked"] == cmd
    assert payload["stdout"] == "ok"
    assert payload["stderr"] == ""
    assert payload["exit_code"] == 0


def test_invokes_optional_mode(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("agent.tools.gcp_experiment_tool.subprocess.run", fake_run)

    args = _args(repo)
    args["mode"] = "persistent"
    payload, ok = _run_tool(args)

    assert ok is True
    assert payload["command_invoked"][-2:] == ["--mode", "persistent"]


def test_returns_structured_failure_output(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            7,
            stdout="partial logs",
            stderr="experiment failed",
        )

    monkeypatch.setattr("agent.tools.gcp_experiment_tool.subprocess.run", fake_run)

    payload, ok = _run_tool(_args(repo))

    assert ok is False
    assert payload == {
        "success": False,
        "command_invoked": [
            "python",
            "scripts/run_gcp_experiment.py",
            "--branch",
            "exp/test-run",
            "--minutes",
            "10",
            "--command",
            "python train.py --steps 2",
        ],
        "stdout": "partial logs",
        "stderr": "experiment failed",
        "exit_code": 7,
    }


def test_single_flight_rejects_existing_lock(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    lock = tmp_path / "experiment.lock"
    lock.write_text("pid=123\n")
    monkeypatch.setenv("ML_INTERN_GCP_LOCK_PATH", str(lock))

    payload, ok = _run_tool(_args(repo))

    assert ok is False
    assert payload["success"] is False
    assert "already running" in payload["stderr"]


def test_tool_registered_only_when_enabled():
    from agent.core.tools import create_builtin_tools

    disabled_names = {tool.name for tool in create_builtin_tools()}
    enabled_names = {
        tool.name for tool in create_builtin_tools(enable_gcp_experiment=True)
    }

    assert "run_gcp_experiment" not in disabled_names
    assert "run_gcp_experiment" in enabled_names
