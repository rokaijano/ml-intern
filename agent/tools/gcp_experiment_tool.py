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
from pathlib import Path
from typing import Any

from agent.core.redact import scrub, scrub_string

_SCRIPT_RELATIVE_PATH = Path("scripts") / "run_gcp_experiment.py"
_DEFAULT_MAX_MINUTES = 60
_TIMEOUT_GRACE_SECONDS = 600


class GCPExperimentConfigError(ValueError):
    """Raised when local GCP experiment policy env vars are invalid."""


def _allowed_templates() -> set[str]:
    raw = os.environ.get("ML_INTERN_GCP_ALLOWED_TEMPLATES", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


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
    template = _string_arg(arguments, "template")
    zone = _string_arg(arguments, "zone")
    command = _string_arg(arguments, "command")
    minutes = _minutes_arg(arguments)

    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"repo_path does not exist or is not a directory: {repo_path}")

    script_path = repo_path / _SCRIPT_RELATIVE_PATH
    if not script_path.exists() or not script_path.is_file():
        raise ValueError(
            f"required script not found: {script_path}. "
            "Expected scripts/run_gcp_experiment.py inside repo_path."
        )

    allowed_templates = _allowed_templates()
    if template not in allowed_templates:
        if allowed_templates:
            allowed = ", ".join(sorted(allowed_templates))
            raise ValueError(
                f"template '{template}' is not allowed. "
                f"Allowed templates: {allowed}."
            )
        raise ValueError(
            "no GCP templates are allowed. Set "
            "ML_INTERN_GCP_ALLOWED_TEMPLATES to a comma-separated allowlist."
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
        "--template",
        template,
        "--zone",
        zone,
        "--minutes",
        str(minutes),
        "--command",
        command,
    ]
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


GCP_EXPERIMENT_TOOL_SPEC = {
    "name": "run_gcp_experiment",
    "description": (
        "Run a bounded research experiment on a disposable Google Cloud VM via "
        "the research repo's scripts/run_gcp_experiment.py wrapper. Safety model: "
        "ml-intern never calls gcloud directly and never runs shell=True. It only "
        "invokes that fixed script with an argv list after validating repo_path, "
        "an approved instance template from ML_INTERN_GCP_ALLOWED_TEMPLATES, a "
        "bounded runtime from ML_INTERN_GCP_MAX_MINUTES, and an exp/* branch by "
        "default. The external script is responsible for VM creation from the "
        "approved template, experiment execution, log/result collection, manifests, "
        "and teardown."
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
            "template": {
                "type": "string",
                "description": (
                    "GCP instance template name. Must be present in "
                    "ML_INTERN_GCP_ALLOWED_TEMPLATES."
                ),
            },
            "zone": {
                "type": "string",
                "description": "GCP zone where the disposable VM should be created.",
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
        },
        "required": ["repo_path", "branch", "template", "zone", "minutes", "command"],
    },
}


async def run_gcp_experiment_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Validate policy, invoke the repo wrapper script, and return JSON output."""
    command_invoked: list[str] = []
    try:
        repo_path, command_invoked, minutes = _validate_and_build_command(arguments)
        completed = await asyncio.to_thread(_run_command, repo_path, command_invoked, minutes)
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
