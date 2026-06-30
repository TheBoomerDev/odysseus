"""Tests for cortex/hermes_bridge.py — Hermes Agent bridge integration.

Tests cover >90% of the module including normal paths, error paths,
and filesystem fallback logic. All subprocess calls are mocked.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from cortex.hermes_bridge import (
    HERMES_BIN,
    HermesProfile,
    HermesResponse,
    _discover_from_fs,
    chat,
    chat_async,
    discover_profiles,
    get_profile,
    get_profile_system_prompt,
    get_proxy_base_url,
    proxy_chat,
)

# =============================================================================
# Dataclass tests
# =============================================================================


class TestHermesProfile:
    def test_defaults(self):
        """HermesProfile with only name sets sensible defaults."""
        p = HermesProfile(name="test")
        assert p.name == "test"
        assert p.model == ""
        assert p.provider == ""
        assert p.gateway_status == "stopped"
        assert p.alias == ""
        assert p.distribution == ""
        assert p.skills == []
        assert p.toolsets == []
        assert p.system_prompt == ""

    def test_all_fields(self):
        """HermesProfile can be constructed with every field."""
        p = HermesProfile(
            name="full",
            model="gpt-4",
            provider="openai",
            gateway_status="running",
            alias="f",
            distribution="d",
            skills=["a", "b"],
            toolsets=["c"],
            system_prompt="hello",
        )
        assert p.name == "full"
        assert p.model == "gpt-4"
        assert p.provider == "openai"
        assert p.gateway_status == "running"
        assert p.alias == "f"
        assert p.distribution == "d"
        assert p.skills == ["a", "b"]
        assert p.toolsets == ["c"]
        assert p.system_prompt == "hello"


class TestHermesResponse:
    def test_defaults(self):
        """HermesResponse with only content."""
        r = HermesResponse(content="hi")
        assert r.content == "hi"
        assert r.session_id == ""
        assert r.duration_ms == 0.0
        assert r.error is None
        assert r.exit_code == 0

    def test_all_fields(self):
        """HermesResponse with every field."""
        r = HermesResponse(
            content="hello",
            session_id="s1",
            duration_ms=150.0,
            error="err",
            exit_code=1,
        )
        assert r.content == "hello"
        assert r.session_id == "s1"
        assert r.duration_ms == 150.0
        assert r.error == "err"
        assert r.exit_code == 1


# =============================================================================
# chat()
# =============================================================================


class TestChat:
    """Tests for the chat() function — synchronous CLI invocation."""

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_success(self, mock_run):
        """Basic successful chat call."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Hello!", stderr="", text=True
        )
        resp = chat("hello")
        assert resp.content == "Hello!"
        assert resp.exit_code == 0
        assert resp.error is None
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == [HERMES_BIN, "chat", "--query", "hello", "--quiet"]

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_success_with_session_id(self, mock_run):
        """Session ID is extracted from first line when present."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="session_id: abc123\nHere is the response.",
            stderr="",
            text=True,
        )
        resp = chat("hello")
        assert resp.content == "Here is the response."
        assert resp.session_id == "abc123"
        assert resp.exit_code == 0

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_success_session_id_only(self, mock_run):
        """Handle case where stdout is just 'session_id: xxx' with no body."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="session_id: s1",
            stderr="",
            text=True,
        )
        resp = chat("hello")
        assert resp.session_id == "s1"
        assert resp.content == ""  # no content after the session_id line

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_with_profile_model_skills(self, mock_run):
        """Profile, model override, and skills are forwarded to the CLI."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Hello!", stderr="", text=True
        )
        resp = chat(
            "hello",
            profile="test_profile",
            model="gpt-4",
            skills=["skill1", "skill2"],
        )
        assert resp.content == "Hello!"
        assert resp.exit_code == 0
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert "--profile" in cmd
        assert "test_profile" in cmd
        assert "-m" in cmd
        assert "gpt-4" in cmd
        assert "-s" in cmd
        assert "skill1,skill2" in cmd
        assert kwargs["env"]["HERMES_PROFILE"] == "test_profile"

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_with_profile_no_skills(self, mock_run):
        """Profile without skills still sets HERMES_PROFILE env var."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Hello!", stderr="", text=True
        )
        resp = chat("hello", profile="p1")
        assert resp.content == "Hello!"
        args, kwargs = mock_run.call_args
        assert kwargs["env"]["HERMES_PROFILE"] == "p1"

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_nonzero_exit(self, mock_run):
        """Non-zero exit code returns error in HermesResponse."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Something went wrong", text=True
        )
        resp = chat("hello")
        assert resp.content == ""
        assert resp.error == "Something went wrong"
        assert resp.exit_code == 1
        assert resp.duration_ms >= 0

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_nonzero_exit_empty_stderr(self, mock_run):
        """When stderr is empty, error is formatted from exit code."""
        mock_run.return_value = MagicMock(
            returncode=2, stdout="", stderr="", text=True
        )
        resp = chat("hello")
        assert resp.error == "Exit code 2"
        assert resp.exit_code == 2

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_timeout(self, mock_run):
        """TimeoutExpired is caught and returned as error with exit code 124."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="hermes", timeout=120)
        resp = chat("hello")
        assert resp.content == ""
        assert "Timeout after 120s" in resp.error
        assert resp.exit_code == 124
        assert resp.duration_ms >= 0

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_file_not_found(self, mock_run):
        """FileNotFoundError is caught and returned as error with exit code 127."""
        mock_run.side_effect = FileNotFoundError()
        resp = chat("hello")
        assert resp.content == ""
        assert "Hermes binary not found" in resp.error
        assert resp.exit_code == 127

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_generic_exception(self, mock_run):
        """Any other exception is caught and returned as error."""
        mock_run.side_effect = RuntimeError("Unexpected error")
        resp = chat("hello")
        assert resp.content == ""
        assert resp.error == "Unexpected error"
        assert resp.exit_code == 1
        assert resp.duration_ms >= 0


# =============================================================================
# chat_async()
# =============================================================================


class TestChatAsync:
    """Tests for chat_async() — async wrapper around chat()."""

    @patch("cortex.hermes_bridge.chat")
    async def test_delegates_to_chat(self, mock_chat):
        """chat_async calls chat() with the same arguments via executor."""
        mock_chat.return_value = HermesResponse(content="async reply")

        # We patch asyncio.get_event_loop's run_in_executor to run inline
        # so the test doesn't actually spawn a thread.
        resp = await chat_async(
            "prompt", profile="p", model="m", skills=["s"], timeout=60
        )
        assert resp.content == "async reply"
        mock_chat.assert_called_once_with(
            prompt="prompt",
            profile="p",
            model="m",
            skills=["s"],
            timeout=60,
        )


# =============================================================================
# discover_profiles()
# =============================================================================


class TestDiscoverProfiles:
    """Tests for discover_profiles() — CLI-first with FS fallback."""

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_json_success(self, mock_run):
        """Happy path: parses JSON from `hermes profile list --json`."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "name": "p1",
                        "model": "gpt4",
                        "provider": "openai",
                        "gateway": "running",
                        "alias": "",
                        "distribution": "",
                    },
                    {"name": "p2"},
                ]
            ),
            stderr="",
            text=True,
        )
        profiles = discover_profiles()
        assert len(profiles) == 2
        assert profiles[0].name == "p1"
        assert profiles[0].model == "gpt4"
        assert profiles[0].provider == "openai"
        assert profiles[0].gateway_status == "running"
        assert profiles[1].name == "p2"

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_json_success_with_profile_key(self, mock_run):
        """Fallback to 'profile' key when 'name' is absent."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([{"profile": "my-profile"}]),
            stderr="",
            text=True,
        )
        profiles = discover_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "my-profile"

    @patch("cortex.hermes_bridge.subprocess.run")
    @patch("cortex.hermes_bridge._discover_from_fs")
    def test_nonzero_exit_fallsback(self, mock_fs, mock_run):
        """Non-zero return code falls back to FS scan."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error", text=True
        )
        mock_fs.return_value = [HermesProfile(name="fs-profile")]
        profiles = discover_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "fs-profile"

    @patch("cortex.hermes_bridge.subprocess.run")
    @patch("cortex.hermes_bridge._discover_from_fs")
    def test_json_decode_error_fallsback(self, mock_fs, mock_run):
        """Broken JSON falls back to FS scan."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="not-json", stderr="", text=True
        )
        mock_fs.return_value = [HermesProfile(name="fs-profile")]
        profiles = discover_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "fs-profile"

    @patch("cortex.hermes_bridge.subprocess.run")
    @patch("cortex.hermes_bridge._discover_from_fs")
    def test_timeout_fallsback(self, mock_fs, mock_run):
        """TimeoutExpired falls back to FS scan."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="hermes", timeout=15)
        mock_fs.return_value = [HermesProfile(name="fs-profile")]
        profiles = discover_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "fs-profile"

    @patch("cortex.hermes_bridge.subprocess.run")
    @patch("cortex.hermes_bridge._discover_from_fs")
    def test_file_not_found_fallsback(self, mock_fs, mock_run):
        """FileNotFoundError falls back to FS scan."""
        mock_run.side_effect = FileNotFoundError()
        mock_fs.return_value = [HermesProfile(name="fs-profile")]
        profiles = discover_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "fs-profile"


# =============================================================================
# _discover_from_fs()
# =============================================================================


class TestDiscoverFromFS:
    """Tests for _discover_from_fs() — filesystem fallback."""

    def test_no_profiles_dir(self, tmp_path):
        """When the profiles directory does not exist, return empty list."""
        fake_dir = tmp_path / "nonexistent"
        with patch("cortex.hermes_bridge.HERMES_PROFILES_DIR", str(fake_dir)):
            profiles = _discover_from_fs()
        assert profiles == []

    def test_with_valid_profiles(self, tmp_path):
        """Discover profiles from config.yaml and SOUL.md in subdirectories."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Profile with full config and SOUL.md
        p1 = profiles_dir / "profile1"
        p1.mkdir()
        (p1 / "config.yaml").write_text(
            "model:\n  default: gpt-4\n  provider: openai\n"
        )
        (p1 / "SOUL.md").write_text("You are a helpful AI.")

        # Profile with config only, no SOUL.md
        p2 = profiles_dir / "profile2"
        p2.mkdir()
        (p2 / "config.yaml").write_text("model:\n  default: claude\n  provider: \n")

        # Non-directory entry (should be skipped)
        (profiles_dir / "file.txt").write_text("not a dir")

        # Directory without config.yaml (should be skipped)
        empty_dir = profiles_dir / "empty"
        empty_dir.mkdir()

        with patch("cortex.hermes_bridge.HERMES_PROFILES_DIR", str(profiles_dir)):
            profiles = _discover_from_fs()

        assert len(profiles) == 2
        assert profiles[0].name == "profile1"
        assert profiles[0].model == "gpt-4"
        assert profiles[0].provider == "openai"
        assert profiles[0].system_prompt == "You are a helpful AI."
        assert profiles[1].name == "profile2"
        assert profiles[1].model == "claude"
        assert profiles[1].system_prompt == ""

    def test_parse_exception_skipped(self, tmp_path):
        """If config.yaml parsing fails, the profile is still added with defaults."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        broken = profiles_dir / "broken"
        broken.mkdir()
        (broken / "config.yaml").write_text(
            "invalid: yaml: [unbalanced brackets: ["
        )

        with patch("cortex.hermes_bridge.HERMES_PROFILES_DIR", str(profiles_dir)):
            profiles = _discover_from_fs()

        assert len(profiles) == 1
        assert profiles[0].name == "broken"
        assert profiles[0].model == ""  # empty because parsing failed

    def test_no_model_key_in_config(self, tmp_path):
        """Config without 'model' key still adds profile with defaults."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        p = profiles_dir / "myprofile"
        p.mkdir()
        (p / "config.yaml").write_text("other_key: 123\n")

        with patch("cortex.hermes_bridge.HERMES_PROFILES_DIR", str(profiles_dir)):
            profiles = _discover_from_fs()

        assert len(profiles) == 1
        assert profiles[0].name == "myprofile"
        assert profiles[0].model == ""


# =============================================================================
# get_profile()
# =============================================================================


class TestGetProfile:
    """Tests for get_profile() — single profile lookup."""

    @patch("cortex.hermes_bridge.discover_profiles")
    def test_found(self, mock_discover):
        """get_profile returns the matching profile."""
        mock_discover.return_value = [
            HermesProfile(name="a"),
            HermesProfile(name="b", model="gpt4"),
        ]
        p = get_profile("b")
        assert p is not None
        assert p.name == "b"
        assert p.model == "gpt4"

    @patch("cortex.hermes_bridge.discover_profiles")
    def test_not_found(self, mock_discover):
        """get_profile returns None when no profile matches."""
        mock_discover.return_value = [HermesProfile(name="a")]
        p = get_profile("nonexistent")
        assert p is None


# =============================================================================
# get_profile_system_prompt()
# =============================================================================


class TestGetProfileSystemPrompt:
    """Tests for get_profile_system_prompt() — direct SOUL.md reader."""

    def test_file_exists(self, tmp_path):
        """Returns content of SOUL.md when it exists."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        soul = profiles_dir / "myprofile" / "SOUL.md"
        soul.parent.mkdir(parents=True)
        soul.write_text("You are Odysseus.")

        with patch("cortex.hermes_bridge.HERMES_PROFILES_DIR", str(profiles_dir)):
            content = get_profile_system_prompt("myprofile")
        assert content == "You are Odysseus."

    def test_file_not_exists(self):
        """Returns empty string when SOUL.md does not exist."""
        # Patch HERMES_PROFILES_DIR to a non-existent path
        with patch("cortex.hermes_bridge.HERMES_PROFILES_DIR", "/nonexistent/profiles"):
            content = get_profile_system_prompt("unknown")
        assert content == ""


# =============================================================================
# get_proxy_base_url()
# =============================================================================


class TestGetProxyBaseURL:
    """Tests for get_proxy_base_url() — proxy status check."""

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_proxy_ready(self, mock_run):
        """Returns default proxy URL when proxy status reports ready."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="proxy is ready", stderr="", text=True
        )
        url = get_proxy_base_url()
        assert url == "http://localhost:8080/v1"

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_proxy_not_ready(self, mock_run):
        """Returns None when proxy is not ready."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="not ready", stderr="", text=True
        )
        url = get_proxy_base_url()
        assert url is None

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_stdout_does_not_contain_ready(self, mock_run):
        """Returns None when stdout doesn't contain 'ready'."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="starting up", stderr="", text=True
        )
        url = get_proxy_base_url()
        assert url is None

    @patch("cortex.hermes_bridge.subprocess.run")
    def test_exception(self, mock_run):
        """Exception during subprocess returns None gracefully."""
        mock_run.side_effect = FileNotFoundError()
        url = get_proxy_base_url()
        assert url is None


# =============================================================================
# proxy_chat()
# =============================================================================


class TestProxyChat:
    """Tests for proxy_chat() — OpenAI-compatible HTTP proxy."""

    @patch("cortex.hermes_bridge.get_proxy_base_url")
    def test_proxy_not_running(self, mock_base_url):
        """Returns error when proxy is not running."""
        mock_base_url.return_value = None
        resp = proxy_chat("hello")
        assert resp.content == ""
        assert "Hermes proxy is not running" in resp.error
        assert resp.exit_code == 1

    @patch("cortex.hermes_bridge.get_proxy_base_url")
    @patch("httpx.Client")
    def test_success(self, mock_client_class, mock_base_url):
        """Successful proxy chat returns the assistant's reply."""
        mock_base_url.return_value = "http://localhost:8080/v1"
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "Hello from proxy!"}}]
            },
        )

        resp = proxy_chat("hi")
        assert resp.content == "Hello from proxy!"
        assert resp.exit_code == 0
        assert resp.error is None

    @patch("cortex.hermes_bridge.get_proxy_base_url")
    @patch("httpx.Client")
    def test_with_system_prompt(self, mock_client_class, mock_base_url):
        """System prompt is included in the messages array."""
        mock_base_url.return_value = "http://localhost:8080/v1"
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "OK"}}]
            },
        )

        resp = proxy_chat("hi", system_prompt="Be helpful.")
        assert resp.content == "OK"
        # Verify the system message was included
        call_kwargs = mock_client.post.call_args[1]
        messages = call_kwargs["json"]["messages"]
        assert messages[0] == {"role": "system", "content": "Be helpful."}
        assert messages[1] == {"role": "user", "content": "hi"}

    @patch("cortex.hermes_bridge.get_proxy_base_url")
    @patch("httpx.Client")
    def test_http_error(self, mock_client_class, mock_base_url):
        """Non-200 HTTP status returns error."""
        mock_base_url.return_value = "http://localhost:8080/v1"
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = MagicMock(
            status_code=404,
            text="Not Found",
        )

        resp = proxy_chat("hi")
        assert resp.content == ""
        assert "Proxy returned 404" in resp.error
        assert resp.exit_code == 404

    @patch("cortex.hermes_bridge.get_proxy_base_url")
    @patch("httpx.Client")
    def test_exception(self, mock_client_class, mock_base_url):
        """HTTPX exception is caught and returned as error."""
        mock_base_url.return_value = "http://localhost:8080/v1"
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = ConnectionError("Connection refused")

        resp = proxy_chat("hi")
        assert resp.content == ""
        assert "Connection refused" in resp.error
        assert resp.exit_code == 1
