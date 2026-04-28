"""
Controlled GCP experiment tool.

This module intentionally does not expose gcloud or arbitrary shell execution
to the agent. It validates a narrow set of inputs, then invokes one fixed
script inside the research repo:

    python scripts/run_gcp_experiment.py ...

The research repo script owns git state, VM lifecycle, log collection, and VM
cleanup. ml-intern only enforces policy and reports structured subprocess
results.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent.core.redact import scrub, scrub_string

_SCRIPT_RELATIVE_PATH = Path("scripts") / "run_gcp_experiment.py"
_DEFAULT_MAX_MINUTES = 60
_TIMEOUT_GRACE_SECONDS = 600


class GCPExperimentConfigError(ValueError):
    """Raised when local GCP experiment policy env vars are invalid."""


def _max_minutes() -> int:
    raw = os.environ.get("ML_INTERN_GCP_MAX_MINUTES")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MAX_MINUTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise GCPExperimentConfigError(
            "ML_INTERN_GCP_MAX_MINUTES must be a positive integer."
        ) from exc
    if value <= 0:
        raise GCPExperimentConfigError(
            "ML_INTERN_GCP_MAX_MINUTES must be a positive integer."
        )
    return value


def _allow_any_branch() -> bool:
    return os.environ.get("ML_INTERN_GCP_ALLOW_ANY_BRANCH", "").lower() == "true"


def _single_flight_enabled() -> bool:
    return os.environ.get("ML_INTERN_GCP_SINGLE_FLIGHT", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _string_arg(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{name}' must be a non-empty string.")
    return value.strip()


def _minutes_arg(arguments: dict[str, Any]) -> int:
    value = arguments.get("minutes")
    if isinstance(value, bool):
        raise ValueError("'minutes' must be a positive integer.")
    try:
        minutes = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'minutes' must be a positive integer.") from exc
    if minutes <= 0:
        raise ValueError("'minutes' must be a positive integer.")
    return minutes


def _mode_arg(arguments: dict[str, Any]) -> str | None:
    value = arguments.get("mode")
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("'mode' must be one of auto, local, persistent, or disposable.")
    mode = value.strip().lower()
    if mode not in {"auto", "local", "persistent", "disposable"}:
        raise ValueError("'mode' must be one of auto, local, persistent, or disposable.")
    return mode


def _result(
    *,
    success: bool,
    command_invoked: list[str],
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = None,
) -> tuple[str, bool]:
    payload = {
        "success": success,
        "command_invoked": scrub(command_invoked),
        "stdout": scrub_string(_to_text(stdout)),
        "stderr": scrub_string(_to_text(stderr)),
        "exit_code": exit_code,
    }
    return json.dumps(payload, indent=2), success


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _validate_and_build_command(arguments: dict[str, Any]) -> tuple[Path, list[str], int]:
    repo_path = Path(_string_arg(arguments, "repo_path")).expanduser()
    branch = _string_arg(arguments, "branch")
    command = _string_arg(arguments, "command")
    minutes = _minutes_arg(arguments)
    mode = _mode_arg(arguments)

    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"repo_path does not exist or is not a directory: {repo_path}")

    script_path = repo_path / _SCRIPT_RELATIVE_PATH
    if not script_path.exists() or not script_path.is_file():
        raise ValueError(
            f"required script not found: {script_path}. "
            "Expected scripts/run_gcp_experiment.py inside repo_path."
        )

    max_minutes = _max_minutes()
    if minutes > max_minutes:
        raise ValueError(
            f"minutes must be <= ML_INTERN_GCP_MAX_MINUTES ({max_minutes})."
        )

    if not branch.startswith("exp/") and not _allow_any_branch():
        raise ValueError(
            "branch must start with 'exp/' unless "
            "ML_INTERN_GCP_ALLOW_ANY_BRANCH=true."
        )

    cmd = [
        "python",
        str(_SCRIPT_RELATIVE_PATH),
        "--branch",
        branch,
        "--minutes",
        str(minutes),
        "--command",
        command,
    ]
    if mode is not None:
        cmd.extend(["--mode", mode])
    return repo_path, cmd, minutes


def _run_command(repo_path: Path, cmd: list[str], minutes: int) -> subprocess.CompletedProcess:
    timeout = minutes * 60 + _TIMEOUT_GRACE_SECONDS
    return subprocess.run(
        cmd,
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        timeout=timeout,
    )


class ExperimentLock:
    """Cross-process single-flight guard for expensive remote experiments."""

    def __init__(self) -> None:
        self.path = Path(
            os.environ.get(
                "ML_INTERN_GCP_LOCK_PATH",
                str(Path(tempfile.gettempdir()) / "ml-intern-gcp-experiment.lock"),
            )
        )
        self.fd: int | None = None

    def __enter__(self) -> "ExperimentLock":
        if not _single_flight_enabled():
            return self
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, f"pid={os.getpid()}\n".encode("utf-8"))
        except FileExistsError as exc:
            details = self.path.read_text(encoding="utf-8", errors="replace")
            raise ValueError(
                "another ml-intern GCP experiment is already running: "
                f"{details.strip()}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.path.unlink(missing_ok=True)


GCP_EXPERIMENT_TOOL_SPEC = {
    "name": "run_gcp_experiment",
    "description": (
        "Run a bounded research experiment locally, on a persistent Google Cloud "
        "VM, or on a disposable Google Cloud VM via "
        "the research repo's scripts/run_gcp_experiment.py wrapper. Safety model: "
        "ml-intern never calls gcloud directly and never runs shell=True. It only "
        "invokes that fixed script with an argv list after validating repo_path, "
        "bounded runtime from ML_INTERN_GCP_MAX_MINUTES, and an exp/* branch by "
        "default. The external script owns git handoff, mode selection, VM "
        "lifecycle, experiment execution, log/result collection, manifests, and "
        "teardown."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Local path to the research project repo.",
            },
            "branch": {
                "type": "string",
                "description": (
                    "Git branch to test. Must start with exp/ unless "
                    "ML_INTERN_GCP_ALLOW_ANY_BRANCH=true."
                ),
            },
            "minutes": {
                "type": "integer",
                "description": (
                    "Requested bounded runtime in minutes. Must be positive and "
                    "<= ML_INTERN_GCP_MAX_MINUTES, default 60."
                ),
            },
            "command": {
                "type": "string",
                "description": (
                    "Experiment command for the external script to run on the VM. "
                    "Passed as one argv value; ml-intern does not interpret it."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["auto", "local", "persistent", "disposable"],
                "description": (
                    "Optional execution mode. Defaults to auto, which lets the "
                    "research repo choose persistent, disposable, or local from env."
                ),
            },
        },
        "required": ["repo_path", "branch", "minutes", "command"],
    },
}


async def run_gcp_experiment_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Validate policy, invoke the repo wrapper script, and return JSON output."""
    command_invoked: list[str] = []
    try:
        repo_path, command_invoked, minutes = _validate_and_build_command(arguments)
        with ExperimentLock():
            completed = await asyncio.to_thread(
                _run_command, repo_path, command_invoked, minutes
            )
    except subprocess.TimeoutExpired as exc:
        return _result(
            success=False,
            command_invoked=command_invoked,
            stdout=exc.stdout or "",
            stderr=f"Timed out after {exc.timeout} seconds.",
            exit_code=None,
        )
    except (GCPExperimentConfigError, ValueError) as exc:
        return _result(
            success=False,
            command_invoked=command_invoked,
            stderr=str(exc),
            exit_code=None,
        )
    except Exception as exc:
        return _result(
            success=False,
            command_invoked=command_invoked,
            stderr=f"Error running GCP experiment wrapper: {exc}",
            exit_code=None,
        )

    return _result(
        success=completed.returncode == 0,
        command_invoked=command_invoked,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        exit_code=completed.returncode,
    )
