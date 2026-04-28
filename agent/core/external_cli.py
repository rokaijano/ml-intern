"""Whole-turn external CLI backends for ml-intern.

Codex and GitHub Copilot CLIs are agents, not chat-completion providers. This
module treats them as whole-turn backends: ml-intern renders the current
conversation into one prompt, invokes the selected CLI once, and uses the final
CLI answer as the assistant message. Tool calls are handled inside that CLI
process rather than emitted back through ml-intern's ToolRouter.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
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
HF_MCP_SERVER_NAME = "hf-mcp-server"
HF_MCP_SERVER_URL = "https://huggingface.co/mcp?login"
GITHUB_MCP_SERVER_NAME = "github"
GITHUB_MCP_SERVER_URL = "https://api.githubcopilot.com/mcp/"


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


def _external_cli_backend_instructions(model_name: str | None) -> list[str]:
    common = [
        "Use the backend's built-in local file and shell coding tools for repository work.",
        "For GCP experiments, use the repository wrapper scripts/run_gcp_experiment.py "
        "(or the ml-intern run_gcp_experiment tool if it is available). Do not run raw "
        "gcloud commands directly.",
    ]
    if model_name == CODEX_MODEL_ID:
        return [
            *common,
            f"Hugging Face MCP is configured as {HF_MCP_SERVER_NAME}.",
            f"GitHub MCP is configured as {GITHUB_MCP_SERVER_NAME}.",
        ]
    if model_name == COPILOT_MODEL_ID:
        return [
            *common,
            "Copilot CLI's built-in GitHub MCP server is enabled.",
            f"Hugging Face MCP is configured as {HF_MCP_SERVER_NAME}.",
        ]
    return common


def render_external_cli_prompt(messages: list[Any], model_name: str | None = None) -> str:
    parts = [
        "You are acting as ml-intern's selected coding backend for this turn.",
        "Work in the current repository, make necessary edits directly, run focused checks, "
        "and finish with a concise summary for the user.",
        "Backend instructions:",
        *[f"- {instruction}" for instruction in _external_cli_backend_instructions(model_name)],
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
        "-c",
        f'mcp_servers.{HF_MCP_SERVER_NAME}.url="{HF_MCP_SERVER_URL}"',
        "-c",
        f'mcp_servers.{GITHUB_MCP_SERVER_NAME}.url="{GITHUB_MCP_SERVER_URL}"',
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
    hf_mcp_config = json.dumps(
        {
            "mcpServers": {
                HF_MCP_SERVER_NAME: {
                    "type": "http",
                    "url": HF_MCP_SERVER_URL,
                    "tools": ["*"],
                }
            }
        },
        separators=(",", ":"),
    )
    return [
        executable,
        "-p",
        prompt,
        "--allow-all",
        "--add-dir",
        cwd,
        "--enable-all-github-mcp-tools",
        "--additional-mcp-config",
        hf_mcp_config,
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


def _process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _terminate_external_cli_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    if os.name == "nt":
        try:
            taskkill = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await taskkill.wait()
        except OSError:
            process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        await process.wait()


def _decode_output(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", errors="replace")


async def _run_external_cli(model_name: str, prompt: str, cwd: str) -> ExternalCliResult:
    timeout = _timeout_seconds()
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ml-intern-cli-") as tmp:
        output_path = Path(tmp) / "last-message.txt"
        cmd = build_external_cli_command(model_name, prompt, cwd, output_path)
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_process_group_kwargs(),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            stdout = _decode_output(stdout_bytes)
            stderr = _decode_output(stderr_bytes)
            content = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            if not content.strip():
                content = stdout or stderr
            return ExternalCliResult(
                content=scrub_string(content or ""),
                success=process.returncode == 0,
                stdout=scrub_string(stdout),
                stderr=scrub_string(stderr),
                command_invoked=scrub(_redacted_command(cmd)),
                latency_ms=int((time.monotonic() - start) * 1000),
                exit_code=process.returncode,
            )
        except asyncio.TimeoutError:
            if process is not None:
                await _terminate_external_cli_process(process)
            return ExternalCliResult(
                content="",
                success=False,
                stdout="",
                stderr=f"External CLI timed out after {timeout} seconds.",
                command_invoked=scrub(_redacted_command(cmd)),
                latency_ms=int((time.monotonic() - start) * 1000),
                exit_code=None,
            )
        except asyncio.CancelledError:
            if process is not None:
                await _terminate_external_cli_process(process)
            raise


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
    prompt = render_external_cli_prompt(messages, model_name)
    return await _run_external_cli(model_name, prompt, cwd)
