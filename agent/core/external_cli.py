"""Whole-turn external CLI backends for ml-intern.

Codex and GitHub Copilot CLIs are agents, not chat-completion providers. This
module treats them as whole-turn backends: ml-intern renders the current
conversation into one prompt, invokes the selected CLI once, and uses the final
CLI answer as the assistant message. Tool calls are handled inside that CLI
process rather than emitted back through ml-intern's ToolRouter.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.core.redact import scrub, scrub_string


CODEX_MODEL_ID = "codex-cli/default"
COPILOT_MODEL_ID = "copilot-cli/default"
EXTERNAL_CLI_MODELS = {CODEX_MODEL_ID, COPILOT_MODEL_ID}
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_PROMPT_MAX_CHARS = 60000


@dataclass
class ExternalCliResult:
    content: str
    success: bool
    stdout: str
    stderr: str
    command_invoked: list[str]
    latency_ms: int
    exit_code: int | None


def is_external_cli_model(model_name: str | None) -> bool:
    return bool(model_name) and model_name in EXTERNAL_CLI_MODELS


def _timeout_seconds() -> int:
    raw = os.environ.get("ML_INTERN_EXTERNAL_CLI_TIMEOUT_SECONDS", "")
    if not raw.strip():
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return max(1, value)


def _prompt_max_chars() -> int:
    raw = os.environ.get("ML_INTERN_EXTERNAL_CLI_PROMPT_MAX_CHARS", "")
    if not raw.strip():
        return DEFAULT_PROMPT_MAX_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_PROMPT_MAX_CHARS
    return max(1000, value)


def _message_to_text(message: Any) -> str:
    role = getattr(message, "role", None) or (
        message.get("role") if isinstance(message, dict) else "unknown"
    )
    content = getattr(message, "content", None) or (
        message.get("content") if isinstance(message, dict) else ""
    )
    tool_calls = getattr(message, "tool_calls", None) or (
        message.get("tool_calls") if isinstance(message, dict) else None
    )
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    text = f"{role}: {content or ''}".rstrip()
    if tool_calls:
        text += f"\n{role}_tool_calls: {tool_calls}"
    return text


def render_external_cli_prompt(messages: list[Any]) -> str:
    parts = [
        "You are acting as ml-intern's selected coding backend for this turn.",
        "Work in the current repository, make necessary edits directly, run focused checks, "
        "and finish with a concise summary for the user.",
        "Conversation so far:",
    ]
    parts.extend(_message_to_text(message) for message in messages)
    prompt = "\n\n".join(parts)
    max_chars = _prompt_max_chars()
    if len(prompt) <= max_chars:
        return prompt
    head = max_chars // 4
    tail = max_chars - head
    return (
        prompt[:head]
        + "\n\n[... earlier conversation truncated for external CLI backend ...]\n\n"
        + prompt[-tail:]
    )


def _codex_command(prompt: str, cwd: str, output_path: Path) -> list[str]:
    executable = os.environ.get("ML_INTERN_CODEX_COMMAND", "codex")
    sandbox = os.environ.get("ML_INTERN_CODEX_SANDBOX", "workspace-write")
    return [
        executable,
        "--ask-for-approval",
        "never",
        "--sandbox",
        sandbox,
        "exec",
        "-C",
        cwd,
        "--color",
        "never",
        "-o",
        str(output_path),
        prompt,
    ]


def _copilot_command(prompt: str, cwd: str) -> list[str]:
    executable = os.environ.get("ML_INTERN_COPILOT_COMMAND", "copilot")
    return [
        executable,
        "-p",
        prompt,
        "--allow-all",
        "--add-dir",
        cwd,
        "--stream",
        "off",
        "--output-format",
        "text",
    ]


def build_external_cli_command(
    model_name: str,
    prompt: str,
    cwd: str,
    output_path: Path,
) -> list[str]:
    if model_name == CODEX_MODEL_ID:
        return _codex_command(prompt, cwd, output_path)
    if model_name == COPILOT_MODEL_ID:
        return _copilot_command(prompt, cwd)
    raise ValueError(f"unsupported external CLI model: {model_name}")


def _run_external_cli_sync(model_name: str, prompt: str, cwd: str) -> ExternalCliResult:
    timeout = _timeout_seconds()
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ml-intern-cli-") as tmp:
        output_path = Path(tmp) / "last-message.txt"
        cmd = build_external_cli_command(model_name, prompt, cwd, output_path)
        try:
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                shell=False,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            content = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            if not content.strip():
                content = completed.stdout or completed.stderr
            return ExternalCliResult(
                content=scrub_string(content or ""),
                success=completed.returncode == 0,
                stdout=scrub_string(completed.stdout or ""),
                stderr=scrub_string(completed.stderr or ""),
                command_invoked=scrub(_redacted_command(cmd)),
                latency_ms=int((time.monotonic() - start) * 1000),
                exit_code=completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ExternalCliResult(
                content="",
                success=False,
                stdout=scrub_string(exc.stdout or ""),
                stderr=f"External CLI timed out after {exc.timeout} seconds.",
                command_invoked=scrub(_redacted_command(cmd)),
                latency_ms=int((time.monotonic() - start) * 1000),
                exit_code=None,
            )


def _redacted_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for flag in ("-p", "--prompt"):
        if flag in redacted:
            idx = redacted.index(flag) + 1
            if idx < len(redacted):
                redacted[idx] = "<prompt>"
    if redacted and redacted[0].endswith("codex") and redacted[-1]:
        redacted[-1] = "<prompt>"
    return redacted


async def call_external_cli(model_name: str, messages: list[Any], cwd: str) -> ExternalCliResult:
    import asyncio

    prompt = render_external_cli_prompt(messages)
    return await asyncio.to_thread(_run_external_cli_sync, model_name, prompt, cwd)
