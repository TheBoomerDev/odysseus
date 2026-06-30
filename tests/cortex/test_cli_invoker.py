"""Tests for cortex/cli_invoker.py — CLI agent orchestrator.

Tests:
  - CliAgent dataclass construction and defaults
  - discover_agents() with mocked shutil.which + subprocess
  - _get_version() success/failure/exception paths
  - invoke() success, timeout, exception, non-zero, missing-path
"""

import subprocess
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from cortex.cli_invoker import (
    CliAgent,
    _AGENT_DISCOVERY,
    _get_version,
    discover_agents,
    invoke,
)


# ---------------------------------------------------------------------------
# CliAgent dataclass
# ---------------------------------------------------------------------------


class TestCliAgent:
    def test_minimal_construction(self):
        """CliAgent can be constructed with only required fields."""
        agent = CliAgent(id="test", name="Test", binary="test-bin")
        assert agent.id == "test"
        assert agent.name == "Test"
        assert agent.binary == "test-bin"
        assert agent.path is None
        assert agent.version is None
        assert agent.capabilities == []
        assert agent.cost_tier == 0.5
        assert agent.invoke_template == "{binary} {prompt}"
        assert agent.timeout_seconds == 180

    def test_full_construction(self):
        """CliAgent can be constructed with all fields."""
        agent = CliAgent(
            id="full",
            name="Full Agent",
            binary="full-bin",
            path="/usr/bin/full-bin",
            version="1.0.0",
            capabilities=["code_edit"],
            cost_tier=0.9,
            invoke_template="{binary} run {prompt}",
            timeout_seconds=300,
        )
        assert agent.id == "full"
        assert agent.path == "/usr/bin/full-bin"
        assert agent.version == "1.0.0"
        assert agent.capabilities == ["code_edit"]
        assert agent.cost_tier == 0.9

    def test_agent_discovery_list_has_agents(self):
        """_AGENT_DISCOVERY contains known agent definitions."""
        ids = [a.id for a in _AGENT_DISCOVERY]
        assert "claude" in ids
        assert "codex" in ids
        assert "agy" in ids
        assert "graphify" in ids
        assert "ollama" in ids

    def test_ollama_invoke_template_different(self):
        """Ollama uses a different invoke template with {model}."""
        ollama = [a for a in _AGENT_DISCOVERY if a.id == "ollama"][0]
        assert ollama.invoke_template == "{binary} run {model} {prompt}"
        assert ollama.timeout_seconds == 300


# ---------------------------------------------------------------------------
# discover_agents
# ---------------------------------------------------------------------------


class TestDiscoverAgents:
    @patch("cortex.cli_invoker.shutil.which")
    @patch("cortex.cli_invoker._get_version")
    def test_discovers_claude(self, mock_get_version, mock_which):
        """discover_agents() returns claude when claude binary is in PATH."""
        # claude found, codex not found, agy not found, graphify not found, ollama not found
        def which_side(binary):
            mapping = {
                "claude": "/usr/local/bin/claude",
                "codex": None,
                "codex.js": None,
                "agy": None,
                "graphify": None,
                "ollama": None,
            }
            return mapping.get(binary)

        mock_which.side_effect = which_side
        mock_get_version.return_value = "1.0.0"

        agents = discover_agents()
        assert len(agents) == 1
        assert agents[0].id == "claude"
        assert agents[0].path == "/usr/local/bin/claude"
        assert agents[0].version == "1.0.0"

    @patch("cortex.cli_invoker.shutil.which")
    @patch("cortex.cli_invoker._get_version")
    def test_codex_fallback_to_codex_js(self, mock_get_version, mock_which):
        """discover_agents() tries codex.js fallback when codex binary not found."""
        def which_side(binary):
            mapping = {
                "claude": None,
                "codex": None,
                "codex.js": "/usr/local/bin/codex.js",
                "agy": None,
                "graphify": None,
                "ollama": None,
            }
            return mapping.get(binary)

        mock_which.side_effect = which_side
        mock_get_version.return_value = "0.5.0"

        agents = discover_agents()
        assert len(agents) == 1
        assert agents[0].id == "codex"
        assert agents[0].path == "/usr/local/bin/codex.js"
        assert agents[0].version == "0.5.0"
        # Verify that _get_version was called with "codex" and the path
        mock_get_version.assert_called_once_with("codex", "/usr/local/bin/codex.js")

    @patch("cortex.cli_invoker.shutil.which")
    @patch("cortex.cli_invoker._get_version")
    def test_discovers_all_agents(self, mock_get_version, mock_which):
        """discover_agents() returns all agents when all binaries are in PATH."""
        def which_side(binary):
            return f"/usr/bin/{binary}"

        mock_which.side_effect = which_side
        mock_get_version.return_value = "1.0"

        agents = discover_agents()
        assert len(agents) == 5  # all 5 agents from _AGENT_DISCOVERY
        ids = [a.id for a in agents]
        assert ids == ["claude", "codex", "agy", "graphify", "ollama"]

    @patch("cortex.cli_invoker.shutil.which")
    @patch("cortex.cli_invoker._get_version")
    def test_no_agents_found(self, mock_get_version, mock_which):
        """discover_agents() returns empty list when no binaries found."""
        mock_which.return_value = None

        agents = discover_agents()
        assert agents == []

    @patch("cortex.cli_invoker.shutil.which")
    @patch("cortex.cli_invoker._get_version")
    def test_version_fetch_failure_does_not_block(self, mock_get_version, mock_which):
        """Version fetch failure doesn't prevent agent from being discovered."""
        mock_which.return_value = "/usr/bin/claude"
        mock_get_version.return_value = None

        agents = discover_agents()
        assert len(agents) >= 1
        assert agents[0].version is None


# ---------------------------------------------------------------------------
# _get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    @patch("cortex.cli_invoker.subprocess.run")
    def test_version_success(self, mock_run):
        """_get_version returns stdout when --version succeeds."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Claude CLI v1.5.0\nBuild 2024\n"
        mock_run.return_value = mock_result

        version = _get_version("claude", "/usr/bin/claude")
        assert version == "Claude CLI v1.5.0\nBuild 2024"
        mock_run.assert_called_once_with(
            ["/usr/bin/claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

    @patch("cortex.cli_invoker.subprocess.run")
    def test_version_non_zero(self, mock_run):
        """_get_version returns None when --version returns non-zero."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        version = _get_version("codex", "/usr/bin/codex")
        assert version is None

    @patch("cortex.cli_invoker.subprocess.run")
    def test_version_timeout(self, mock_run):
        """_get_version returns None when subprocess times out."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["/usr/bin/claude", "--version"],
            timeout=10,
        )

        version = _get_version("claude", "/usr/bin/claude")
        assert version is None

    @patch("cortex.cli_invoker.subprocess.run")
    def test_version_exception(self, mock_run):
        """_get_version returns None when subprocess raises an exception."""
        mock_run.side_effect = FileNotFoundError("binary not found")

        version = _get_version("claude", "/usr/bin/claude")
        assert version is None

    @patch("cortex.cli_invoker.subprocess.run")
    def test_version_truncated_to_60_chars(self, mock_run):
        """_get_version truncates version string to 60 characters."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        long_version = "v" * 100
        mock_result.stdout = long_version
        mock_run.return_value = mock_result

        version = _get_version("claude", "/usr/bin/claude")
        assert len(version) == 60
        assert version == "v" * 60


# ---------------------------------------------------------------------------
# invoke
# ---------------------------------------------------------------------------


class TestInvoke:
    def test_error_when_no_path(self):
        """invoke() returns error when agent has no path."""
        agent = CliAgent(id="test", name="Test", binary="test-bin", path=None)
        result = invoke(agent, "hello")
        assert result["status"] == "error"
        assert "not found in PATH" in result["error"]
        assert result["exit_code"] == 10
        assert result["duration_seconds"] == 0

    @patch("cortex.cli_invoker.subprocess.run")
    def test_success(self, mock_run):
        """invoke() returns success with stdout/stderr when exit code is 0."""
        agent = CliAgent(
            id="claude",
            name="Claude Code",
            binary="claude",
            path="/usr/bin/claude",
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello, I'm Claude"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = invoke(agent, "write a poem")
        assert result["status"] == "success"
        assert result["stdout"] == "Hello, I'm Claude"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0
        assert isinstance(result["duration_seconds"], float)
        assert result["duration_seconds"] >= 0

        # Verify command was built correctly
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["/usr/bin/claude", "write", "a", "poem"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 180  # default from CliAgent
        assert kwargs["cwd"]  # should be os.getcwd()

    @patch("cortex.cli_invoker.subprocess.run")
    def test_non_zero_exit(self, mock_run):
        """invoke() returns error status when exit code is non-zero."""
        agent = CliAgent(
            id="codex",
            name="Codex CLI",
            binary="codex",
            path="/usr/bin/codex",
        )
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "something went wrong"
        mock_run.return_value = mock_result

        result = invoke(agent, "do something")
        assert result["status"] == "error"
        assert result["exit_code"] == 1
        assert result["stderr"] == "something went wrong"

    @patch("cortex.cli_invoker.subprocess.run")
    def test_with_workdir_and_timeout(self, mock_run):
        """invoke() passes workdir and custom timeout to subprocess."""
        agent = CliAgent(
            id="agy",
            name="Antigravity CLI",
            binary="agy",
            path="/usr/bin/agy",
            timeout_seconds=120,
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "result"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = invoke(
            agent,
            "search web",
            workdir="/tmp/project",
            timeout=60,
        )
        assert result["status"] == "success"
        args, kwargs = mock_run.call_args
        assert kwargs["cwd"] == "/tmp/project"
        assert kwargs["timeout"] == 60

    @patch("cortex.cli_invoker.subprocess.run")
    def test_with_extra_args(self, mock_run):
        """invoke() appends extra_args to the command."""
        agent = CliAgent(
            id="claude",
            name="Claude Code",
            binary="claude",
            path="/usr/bin/claude",
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "done"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = invoke(
            agent,
            "review code",
            extra_args=["--verbose", "--model", "opus"],
        )
        assert result["status"] == "success"
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd == ["/usr/bin/claude", "review", "code", "--verbose", "--model", "opus"]

    @patch("cortex.cli_invoker.subprocess.run")
    def test_timeout_expired(self, mock_run):
        """invoke() returns timeout status when subprocess times out."""
        agent = CliAgent(
            id="ollama",
            name="Ollama",
            binary="ollama",
            path="/usr/bin/ollama",
            timeout_seconds=300,
        )
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["/usr/bin/ollama", "run", "model", "hello"],
            timeout=300,
        )

        result = invoke(agent, "hello", timeout=300)
        assert result["status"] == "timeout"
        assert result["stdout"] == ""
        assert "timed out after 300s" in result["stderr"]
        assert result["exit_code"] == -1
        assert result["duration_seconds"] == 300.0

    @patch("cortex.cli_invoker.subprocess.run")
    def test_generic_exception(self, mock_run):
        """invoke() catches generic exceptions and returns error."""
        agent = CliAgent(
            id="claude",
            name="Claude Code",
            binary="claude",
            path="/usr/bin/claude",
        )
        mock_run.side_effect = PermissionError("access denied")

        result = invoke(agent, "do something")
        assert result["status"] == "error"
        assert result["stdout"] == ""
        assert "access denied" in result["stderr"]
        assert result["exit_code"] == -1
        assert isinstance(result["duration_seconds"], float)

    @patch("cortex.cli_invoker.subprocess.run")
    def test_invoke_with_ollama_template(self, mock_run):
        """invoke() uses agent's custom invoke_template (ollama)."""
        agent = CliAgent(
            id="ollama",
            name="Ollama",
            binary="ollama",
            path="/usr/bin/ollama",
            invoke_template="{binary} run {model} {prompt}",
            timeout_seconds=300,
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "response from model"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # The template has {model} which doesn't get replaced, so it stays as literal "{model}"
        result = invoke(agent, "tell me a story")
        assert result["status"] == "success"
        args, kwargs = mock_run.call_args
        cmd = args[0]
        # {model} stays as literal in the template since we only replace {binary} and {prompt}
        assert cmd == ["/usr/bin/ollama", "run", "{model}", "tell", "me", "a", "story"]

    @patch("cortex.cli_invoker.subprocess.run")
    def test_duration_rounded(self, mock_run):
        """invoke() rounds duration to 2 decimal places."""
        agent = CliAgent(
            id="claude",
            name="Claude Code",
            binary="claude",
            path="/usr/bin/claude",
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # We can't easily control time.time() without additional mocking,
        # but we can verify the value is a float with 2 decimal places
        result = invoke(agent, "hello")
        assert result["duration_seconds"] == round(result["duration_seconds"], 2)
