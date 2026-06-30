"""
cortex/cli_invoker.py — CLI agent orchestrator.

Discovers installed CLI agents on the system and wraps them as
Odysseus tools. Each CLI agent gets a tool that runs the CLI with
a prompt and captures output (stdout, stderr, exit code).

Inspired by CORTEX invoker/ and integration with Claude Code / Codex.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent manifest
# ---------------------------------------------------------------------------

@dataclass
class CliAgent:
    id: str
    name: str
    binary: str
    path: Optional[str] = None
    version: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    cost_tier: float = 0.5  # 0 = free, 1 = expensive
    invoke_template: str = "{binary} {prompt}"
    timeout_seconds: int = 180


# Discovery patterns: binary → agent config
_AGENT_DISCOVERY: List[CliAgent] = [
    CliAgent(id="claude", name="Claude Code", binary="claude",
             capabilities=["code_edit", "code_review", "architecture_design",
                          "doc_generation", "adversarial_review", "test_generation"],
             cost_tier=0.8, invoke_template="{binary} {prompt}"),
    CliAgent(id="codex", name="Codex CLI", binary="codex",
             capabilities=["code_edit", "multi_file_diff", "code_review",
                          "test_generation"],
             cost_tier=0.7, invoke_template="{binary} {prompt}"),
    CliAgent(id="agy", name="Antigravity CLI", binary="agy",
             capabilities=["web_search", "code_review", "doc_generation",
                          "test_execution"],
             cost_tier=0.3, invoke_template="{binary} {prompt}"),
    CliAgent(id="graphify", name="Graphify", binary="graphify",
             capabilities=["code_review", "architecture_design"],
             cost_tier=0.2, invoke_template="{binary} {prompt}"),
    CliAgent(id="ollama", name="Ollama", binary="ollama",
             capabilities=["doc_generation", "code_review"],
             cost_tier=0.1, invoke_template="{binary} run {model} {prompt}",
             timeout_seconds=300),
]


def discover_agents() -> List[CliAgent]:
    """Scan the system for installed CLI agents.

    Returns:
        List of discovered CliAgent with resolved paths and versions.
    """
    agents: List[CliAgent] = []
    for agent in _AGENT_DISCOVERY:
        path = shutil.which(agent.binary)
        if not path:
            # Try codex.js as fallback for codex
            if agent.id == "codex":
                path = shutil.which("codex.js")
            if not path:
                continue

        agent.path = path
        agent.version = _get_version(agent.binary, path)
        agents.append(agent)

    return agents


def _get_version(binary: str, path: str) -> Optional[str]:
    """Attempt to get the version of a CLI agent."""
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:60]
    except Exception:
        pass
    return None


def invoke(
    agent: CliAgent,
    prompt: str,
    workdir: Optional[str] = None,
    timeout: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict:
    """Invoke a CLI agent with the given prompt.

    Args:
        agent: The CliAgent to invoke
        prompt: The prompt to send
        workdir: Working directory for the subprocess
        timeout: Timeout in seconds (default: agent.timeout_seconds)
        extra_args: Additional CLI arguments

    Returns:
        Dict with keys: status, stdout, stderr, exit_code, duration_seconds
    """
    if not agent.path:
        return {
            "status": "error",
            "error": f"agent '{agent.id}' not found in PATH",
            "exit_code": 10,
            "duration_seconds": 0,
        }

    # Build command from template
    template = agent.invoke_template
    cmd_str = template.replace("{binary}", agent.path).replace("{prompt}", prompt)
    if extra_args:
        cmd_str += " " + " ".join(extra_args)

    cmd = cmd_str.split()

    timeout = timeout or agent.timeout_seconds

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir or os.getcwd(),
        )
        duration = time.time() - start

        status = "success" if result.returncode == 0 else "error"
        return {
            "status": status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "duration_seconds": round(duration, 2),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "stdout": "",
            "stderr": f"timed out after {timeout}s",
            "exit_code": -1,
            "duration_seconds": float(timeout),
        }
    except Exception as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "duration_seconds": round(time.time() - start, 2),
        }
