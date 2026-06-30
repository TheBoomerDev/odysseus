"""
cortex/hermes_bridge.py — Bridge between Odysseus and Hermes Agent.

Two integration modes:
  1. CHAT API — calls `hermes chat --query --quiet` for a given profile
  2. PROXY API — OpenAI-compatible HTTP endpoint (when hermes proxy is running)

Each Hermes profile is a separate "model endpoint" with its own:
  - System prompt (SOUL.md)
  - Configured model/provider
  - Skills, toolsets, and plugins
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HERMES_BIN = shutil.which("hermes") or os.path.expanduser(
    "~/.hermes/hermes-agent/venv/bin/hermes"
)
HERMES_PROFILES_DIR = os.path.expanduser("~/.hermes/profiles")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class HermesProfile:
    """A Hermes Agent profile with its configuration."""

    name: str
    model: str = ""
    provider: str = ""
    gateway_status: str = "stopped"
    alias: str = ""
    distribution: str = ""
    skills: List[str] = field(default_factory=list)
    toolsets: List[str] = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class HermesResponse:
    """Response from a Hermes chat call."""

    content: str
    session_id: str = ""
    duration_ms: float = 0.0
    error: Optional[str] = None
    exit_code: int = 0


# ---------------------------------------------------------------------------
# Profile discovery
# ---------------------------------------------------------------------------


def discover_profiles() -> List[HermesProfile]:
    """Discover all available Hermes profiles via `hermes profile list`."""
    profiles: List[HermesProfile] = []
    try:
        result = subprocess.run(
            [HERMES_BIN, "profile", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.warning("hermes profile list failed: %s", result.stderr)
            return _discover_from_fs()

        raw = json.loads(result.stdout)
        for entry in raw:
            profiles.append(
                HermesProfile(
                    name=entry.get("name", entry.get("profile", "")),
                    model=entry.get("model", ""),
                    provider=entry.get("provider", ""),
                    gateway_status=entry.get("gateway", "stopped"),
                    alias=entry.get("alias", ""),
                    distribution=entry.get("distribution", ""),
                )
            )
    except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Failed to list profiles via CLI, falling back to FS scan")
        return _discover_from_fs()

    return profiles


def _discover_from_fs() -> List[HermesProfile]:
    """Fallback: scan ~/.hermes/profiles/ directly."""
    profiles: List[HermesProfile] = []
    profiles_dir = Path(HERMES_PROFILES_DIR)
    if not profiles_dir.exists():
        return profiles

    for entry in sorted(profiles_dir.iterdir()):
        if entry.is_dir() and (entry / "config.yaml").exists():
            profile = HermesProfile(name=entry.name)
            # Try to extract model from config.yaml
            config_path = entry / "config.yaml"
            try:
                import yaml  # type: ignore[reportMissingImports]

                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg and "model" in cfg:
                    profile.model = cfg["model"].get("default", "")
                    profile.provider = cfg["model"].get("provider", "")
                # Read SOUL.md for system prompt
                soul_path = entry / "SOUL.md"
                if soul_path.exists():
                    profile.system_prompt = soul_path.read_text(encoding="utf-8")[:500]
            except Exception:
                pass
            profiles.append(profile)

    return profiles


def get_profile(name: str) -> Optional[HermesProfile]:
    """Get a single profile by name."""
    for p in discover_profiles():
        if p.name == name:
            return p
    return None


def get_profile_system_prompt(name: str) -> str:
    """Read a profile's SOUL.md content."""
    soul_path = Path(HERMES_PROFILES_DIR) / name / "SOUL.md"
    if soul_path.exists():
        return soul_path.read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Chat API — call Hermes via CLI
# ---------------------------------------------------------------------------


def chat(
    prompt: str,
    profile: str = "",
    model: str = "",
    skills: Optional[List[str]] = None,
    timeout: int = 120,
) -> HermesResponse:
    """Send a prompt to Hermes Agent and get a response.

    Uses `hermes chat --query --quiet` under the hood.
    If profile is specified, runs under that profile's config.

    Args:
        prompt: The prompt to send
        profile: Hermes profile name (default: current/active profile)
        model: Override model (e.g., "anthropic/claude-sonnet-4")
        skills: List of skill names to preload
        timeout: Max seconds to wait for response

    Returns:
        HermesResponse with the agent's reply
    """
    start = time.time()
    try:
        cmd = [HERMES_BIN, "chat", "--query", prompt, "--quiet"]

        if profile:
            cmd.extend(["--profile", profile])
        if model:
            cmd.extend(["-m", model])
        if skills:
            cmd.extend(["-s", ",".join(skills)])

        env = os.environ.copy()
        if profile:
            # Point to the profile's config
            env["HERMES_PROFILE"] = profile

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        duration = (time.time() - start) * 1000

        if result.returncode != 0:
            return HermesResponse(
                content="",
                error=result.stderr.strip() or f"Exit code {result.returncode}",
                exit_code=result.returncode,
                duration_ms=duration,
            )

        # Parse output: session_id line + response
        output = result.stdout.strip()
        session_id = ""
        content = output

        # Extract session_id from first line if present
        if output.startswith("session_id:"):
            parts = output.split("\n", 1)
            session_id = parts[0].replace("session_id:", "").strip()
            content = parts[1].strip() if len(parts) > 1 else ""

        return HermesResponse(
            content=content,
            session_id=session_id,
            duration_ms=duration,
            exit_code=0,
        )

    except subprocess.TimeoutExpired:
        return HermesResponse(
            content="",
            error=f"Timeout after {timeout}s",
            exit_code=124,
            duration_ms=(time.time() - start) * 1000,
        )
    except FileNotFoundError:
        return HermesResponse(
            content="",
            error="Hermes binary not found. Is Hermes Agent installed?",
            exit_code=127,
        )
    except Exception as e:
        return HermesResponse(
            content="",
            error=str(e),
            exit_code=1,
            duration_ms=(time.time() - start) * 1000,
        )


async def chat_async(
    prompt: str,
    profile: str = "",
    model: str = "",
    skills: Optional[List[str]] = None,
    timeout: int = 120,
) -> HermesResponse:
    """Async wrapper around chat()."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: chat(
            prompt=prompt,
            profile=profile,
            model=model,
            skills=skills,
            timeout=timeout,
        ),
    )


# ---------------------------------------------------------------------------
# Proxy API — OpenAI-compatible HTTP adapter
# ---------------------------------------------------------------------------


def get_proxy_base_url() -> Optional[str]:
    """Get the proxy URL if Hermes proxy is running."""
    try:
        result = subprocess.run(
            [HERMES_BIN, "proxy", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "ready" in result.stdout.lower():
            return "http://localhost:8080/v1"  # default proxy port
    except Exception:
        pass
    return None


def proxy_chat(
    prompt: str,
    model: str = "default",
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> HermesResponse:
    """Send a chat completion via the Hermes proxy (OpenAI-compatible)."""
    import httpx

    base_url = get_proxy_base_url()
    if not base_url:
        return HermesResponse(
            content="",
            error="Hermes proxy is not running. Start with: hermes proxy start",
            exit_code=1,
        )

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            if resp.status_code != 200:
                return HermesResponse(
                    content="",
                    error=f"Proxy returned {resp.status_code}: {resp.text}",
                    exit_code=resp.status_code,
                )

            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return HermesResponse(content=content, exit_code=0)

    except Exception as e:
        return HermesResponse(content="", error=str(e), exit_code=1)
