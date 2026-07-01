"""Comprehensive tests for cortex/codegen/llm.py — LLM client interface.

Targets >90% coverage across all classes, parsing helpers, factory function,
and error paths. External calls (hermes_bridge.chat_async, httpx.AsyncClient)
are mocked.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from cortex.codegen.llm import (
    LLMResult,
    LLMClient,
    HermesLLMClient,
    DirectLLMClient,
    _parse_text_tool_calls,
    _strip_tool_calls,
    get_llm_client,
)


# =============================================================================
# LLMResult dataclass
# =============================================================================


class TestLLMResult:
    """LLMResult is the unified response type — verify all defaults and fields."""

    def test_defaults(self):
        result = LLMResult(content="hello")
        assert result.content == "hello"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.error is None
        assert result.provider == ""
        assert result.model == ""

    def test_all_fields(self):
        result = LLMResult(
            content="response",
            tool_calls=[{"id": "1", "type": "function"}],
            finish_reason="tool_calls",
            error=None,
            provider="openai",
            model="gpt-4",
        )
        assert result.content == "response"
        assert result.tool_calls == [{"id": "1", "type": "function"}]
        assert result.finish_reason == "tool_calls"
        assert result.error is None
        assert result.provider == "openai"
        assert result.model == "gpt-4"

    def test_error_set(self):
        result = LLMResult(content="", error="timeout", finish_reason="error")
        assert result.error == "timeout"
        assert result.finish_reason == "error"
        assert result.content == ""

    def test_tool_calls_default_factory(self):
        """Each instance gets its own list, not shared."""
        r1 = LLMResult(content="a")
        r2 = LLMResult(content="b")
        r1.tool_calls.append({"id": "x"})
        assert r2.tool_calls == []


# =============================================================================
# LLMClient ABC
# =============================================================================


class TestLLMClient:
    """LLMClient is the base ABC — verify interface contract."""

    async def test_chat_raises_not_implemented(self):
        client = LLMClient()
        with pytest.raises(NotImplementedError):
            await client.chat(messages=[{"role": "user", "content": "hi"}])

    async def test_chat_structured_no_system(self):
        """chat_structured adds a system prompt with schema hint when no existing system message."""

        class FakeClient(LLMClient):
            async def chat(self, messages, tools=None, temperature=0.2, max_tokens=4096):
                assert len(messages) == 2
                assert messages[0]["role"] == "system"
                assert "valid JSON matching this schema" in messages[0]["content"]
                assert messages[1] == {"role": "user", "content": "do it"}
                return LLMResult(content='{"ok": true}')

        client = FakeClient()
        result = await client.chat_structured(
            messages=[{"role": "user", "content": "do it"}],
            response_schema={"type": "object"},
        )
        assert result.content == '{"ok": true}'

    async def test_chat_structured_with_system(self):
        """chat_structured appends schema hint to existing system message."""

        class FakeClient(LLMClient):
            async def chat(self, messages, tools=None, temperature=0.2, max_tokens=4096):
                assert messages[0]["role"] == "system"
                assert "original" in messages[0]["content"]
                assert "valid JSON matching this schema" in messages[0]["content"]
                return LLMResult(content='{"ok": true}')

        client = FakeClient()
        result = await client.chat_structured(
            messages=[{"role": "system", "content": "original prompt"}, {"role": "user", "content": "do it"}],
            response_schema={"type": "object"},
        )
        assert result.content == '{"ok": true}'

    async def test_chat_structured_empty_messages(self):
        """chat_structured with empty messages just adds system schema hint."""

        class FakeClient(LLMClient):
            async def chat(self, messages, tools=None, temperature=0.2, max_tokens=4096):
                assert len(messages) == 1
                assert messages[0]["role"] == "system"
                return LLMResult(content="ok")

        client = FakeClient()
        result = await client.chat_structured(
            messages=[],
            response_schema={"type": "object"},
        )
        assert result.content == "ok"


# =============================================================================
# HermesLLMClient
# =============================================================================


class TestHermesLLMClient:
    """HermesLLMClient wraps hermes_bridge.chat_async — mock the bridge."""

    def test_init_defaults(self):
        client = HermesLLMClient()
        assert client.provider == "hermes"
        assert client.profile == "swarm1"
        assert client._timeout == 120
        assert client.model == "hermes/swarm1"

    def test_init_custom(self):
        client = HermesLLMClient(profile="custom", timeout=60)
        assert client.profile == "custom"
        assert client._timeout == 60
        assert client.model == "hermes/custom"

    async def test_chat_success_no_tools(self):
        """Happy path: plain text response is parsed correctly."""
        mock_response = MagicMock()
        mock_response.content = "Hello, I am Hermes."
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response):
            client = HermesLLMClient()
            result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "Hello, I am Hermes."
        assert result.tool_calls == []
        assert result.provider == "hermes"
        assert result.model == "hermes/swarm1"
        assert result.finish_reason == "stop"

    async def test_chat_with_tools(self):
        """When tools are provided, the prompt includes tool instructions."""
        mock_response = MagicMock()
        mock_response.content = "I'll use a tool.\nTOOL_CALL: read_file\n{\"path\": \"test.txt\"}\nEND_TOOL_CALL"
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            client = HermesLLMClient()
            result = await client.chat(
                messages=[{"role": "user", "content": "read file"}],
                tools=[{"function": {"name": "read_file", "description": "Read a file"}}],
            )

        # Tool instructions go into the prompt, not the response content
        call_kwargs = mock_chat.call_args[1]
        prompt = call_kwargs["prompt"]
        assert "Available tools:" in prompt
        assert "read_file: Read a file" in prompt
        # Response content has tool calls stripped
        assert result.content == "I'll use a tool."
        assert result.tool_calls == [
            {
                "id": "read_file_0",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "test.txt"}},
            }
        ]

    async def test_chat_with_tool_calls_stripped_from_content(self):
        """Tool call blocks are stripped from content, leaving natural language."""
        mock_response = MagicMock()
        mock_response.content = "Here is the result.\nTOOL_CALL: write_patch\n{\"path\": \"main.py\"}\nEND_TOOL_CALL\nDone."
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response):
            client = HermesLLMClient()
            result = await client.chat(messages=[{"role": "user", "content": "patch it"}])

        assert "TOOL_CALL:" not in result.content
        assert "Here is the result." in result.content
        assert "Done." in result.content

    async def test_chat_error_response(self):
        """When the bridge returns an error, it's propagated."""
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.error = "API key not found"

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response):
            client = HermesLLMClient()
            result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.content == ""
        assert result.error == "API key not found"
        assert result.finish_reason == "error"

    async def test_chat_system_user_assistant_roles(self):
        """Different message roles are labelled correctly in the prompt."""
        mock_response = MagicMock()
        mock_response.content = "OK"
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock) as mock_chat:
            client = HermesLLMClient()
            await client.chat(
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "Do something"},
                ]
            )
            call_kwargs = mock_chat.call_args[1]
            prompt = call_kwargs["prompt"]

        assert "System: You are helpful." in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt
        assert "User: Do something" in prompt

    async def test_chat_passes_profile_and_timeout(self):
        """Profile and timeout are forwarded to chat_async."""
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            client = HermesLLMClient(profile="myprof", timeout=99)
            await client.chat(messages=[{"role": "user", "content": "hi"}])

        mock_chat.assert_awaited_once_with(
            prompt=AnyStr(),
            profile="myprof",
            timeout=99,
        )

    async def test_chat_empty_messages(self):
        """Works with empty messages list."""
        mock_response = MagicMock()
        mock_response.content = "empty"
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response):
            client = HermesLLMClient()
            result = await client.chat(messages=[])
        assert result.content == "empty"


class AnyStr:
    """Helper matcher that matches any string."""

    def __eq__(self, other):
        return isinstance(other, str)


# =============================================================================
# DirectLLMClient — initialization & config
# =============================================================================


class TestDirectLLMClientInit:
    """DirectLLMClient construction and provider validation."""

    def test_valid_providers(self):
        for provider in ["openai", "gemini", "ollama", "openrouter", "deepseek"]:
            client = DirectLLMClient(provider=provider, model="test-model")
            assert client.provider == provider
            assert client.model == "test-model"

    def test_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            DirectLLMClient(provider="unknown", model="x")

    def test_custom_base_url(self):
        client = DirectLLMClient(provider="openai", model="gpt-4", base_url="https://my-proxy.example.com/v1")
        assert client._config["base_url"] == "https://my-proxy.example.com/v1"

    def test_custom_base_url_strips_trailing_slash(self):
        client = DirectLLMClient(provider="openai", model="gpt-4", base_url="https://example.com/")
        assert client._config["base_url"] == "https://example.com"

    def test_default_api_key_empty(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        assert client._api_key == ""

    def test_api_key_set(self):
        client = DirectLLMClient(provider="openai", model="gpt-4", api_key="sk-123")
        assert client._api_key == "sk-123"

    def test_timeout_default(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        assert client._timeout == 120

    def test_timeout_custom(self):
        client = DirectLLMClient(provider="openai", model="gpt-4", timeout=30)
        assert client._timeout == 30

    def test_config_is_copied_not_shared(self):
        c1 = DirectLLMClient(provider="openai", model="gpt-4")
        c2 = DirectLLMClient(provider="openai", model="gpt-4", base_url="https://other.com")
        assert c1._config["base_url"] == "https://api.openai.com/v1"
        assert c2._config["base_url"] == "https://other.com"


# =============================================================================
# DirectLLMClient — _build_url
# =============================================================================


class TestDirectLLMClientBuildUrl:
    """URL construction for each provider type."""

    def test_openai_url(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        assert client._build_url() == "https://api.openai.com/v1/chat/completions"

    def test_openrouter_url(self):
        client = DirectLLMClient(provider="openrouter", model="mistral")
        assert client._build_url() == "https://openrouter.ai/api/v1/chat/completions"

    def test_deepseek_url(self):
        client = DirectLLMClient(provider="deepseek", model="deepseek-chat")
        assert client._build_url() == "https://api.deepseek.com/v1/chat/completions"

    def test_gemini_url(self):
        """Gemini uses {model} in the endpoint path."""
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        assert client._build_url() == "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    def test_ollama_url(self):
        client = DirectLLMClient(provider="ollama", model="llama3")
        assert client._build_url() == "http://localhost:11434/api/chat"

    def test_custom_base_affects_url(self):
        client = DirectLLMClient(provider="openai", model="gpt-4", base_url="https://gateway.example.com")
        assert client._build_url() == "https://gateway.example.com/chat/completions"


# =============================================================================
# DirectLLMClient — _build_headers
# =============================================================================


class TestDirectLLMClientBuildHeaders:
    """Header construction for each provider type."""

    def test_openai_headers(self):
        client = DirectLLMClient(provider="openai", model="gpt-4", api_key="sk-abc")
        headers = client._build_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer sk-abc"

    def test_openrouter_headers(self):
        client = DirectLLMClient(provider="openrouter", model="mistral", api_key="or-key")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer or-key"

    def test_deepseek_headers(self):
        client = DirectLLMClient(provider="deepseek", model="deepseek-chat", api_key="ds-key")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer ds-key"

    def test_gemini_headers(self):
        """Gemini uses x-goog-api-key header instead of Authorization."""
        client = DirectLLMClient(provider="gemini", model="gemini-pro", api_key="gm-key")
        headers = client._build_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["x-goog-api-key"] == "gm-key"
        assert "Authorization" not in headers

    def test_ollama_no_api_key_header(self):
        """Ollama has no api_key_header configured."""
        client = DirectLLMClient(provider="ollama", model="llama3", api_key="whatever")
        headers = client._build_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers
        assert "x-goog-api-key" not in headers

    def test_no_api_key_no_auth_header(self):
        """Without api_key, no auth header is emitted for providers that would normally have one."""
        client = DirectLLMClient(provider="openai", model="gpt-4")
        headers = client._build_headers()
        assert "Authorization" not in headers

    def test_gemini_no_api_key(self):
        """Gemini without api_key omits the x-goog-api-key header."""
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        headers = client._build_headers()
        assert "x-goog-api-key" not in headers


# =============================================================================
# DirectLLMClient — _build_payload
# =============================================================================


class TestDirectLLMClientBuildPayload:
    """Payload construction for each provider type."""

    def test_openai_payload_defaults(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        payload = client._build_payload(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert payload["model"] == "gpt-4"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 4096
        assert "tools" not in payload

    def test_openai_payload_with_tools(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        tools = [{"function": {"name": "read_file"}}]
        payload = client._build_payload(
            messages=[{"role": "user", "content": "read"}],
            tools=tools,
        )
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "auto"

    def test_openai_payload_custom_temperature(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        payload = client._build_payload(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=2048,
        )
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 2048

    def test_gemini_payload_basic(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        payload = client._build_payload(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert "contents" in payload
        assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
        assert payload["generationConfig"]["temperature"] == 0.2
        assert payload["generationConfig"]["maxOutputTokens"] == 4096
        assert "systemInstruction" not in payload

    def test_gemini_payload_with_system_message(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        payload = client._build_payload(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
        )
        assert payload["systemInstruction"]["parts"][0]["text"] == "You are helpful."
        assert len(payload["contents"]) == 1
        assert payload["contents"][0]["role"] == "user"

    def test_gemini_payload_assistant_role_mapped_to_model(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        payload = client._build_payload(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        )
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][1]["role"] == "model"

    def test_gemini_payload_with_tools(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        tools = [{"function": {"name": "read_file", "description": "Read", "parameters": {"type": "object"}}}]
        payload = client._build_payload(
            messages=[{"role": "user", "content": "read"}],
            tools=tools,
        )
        assert "tools" in payload
        gemini_tool = payload["tools"][0]
        assert gemini_tool["functionDeclarations"][0]["name"] == "read_file"

    def test_ollama_payload(self):
        client = DirectLLMClient(provider="ollama", model="llama3")
        payload = client._build_payload(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert payload["model"] == "llama3"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["options"]["temperature"] == 0.2
        assert payload["stream"] is False
        assert "max_tokens" not in payload

    def test_ollama_payload_with_tools(self):
        client = DirectLLMClient(provider="ollama", model="llama3")
        tools = [{"function": {"name": "read_file"}}]
        payload = client._build_payload(
            messages=[{"role": "user", "content": "read"}],
            tools=tools,
        )
        assert payload["tools"] == tools


# =============================================================================
# DirectLLMClient — _parse_response
# =============================================================================


class TestDirectLLMClientParseResponse:
    """Response parsing for each provider type."""

    def test_openai_success(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        data = {
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop",
                }
            ]
        }
        result = client._parse_response(data)
        assert result.content == "Hello!"
        assert result.finish_reason == "stop"
        assert result.tool_calls == []

    def test_openai_with_tool_calls(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        data = {
            "choices": [
                {
                    "message": {
                        "content": "Using tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "test.txt"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = client._parse_response(data)
        assert result.content == "Using tool"
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "read_file"
        assert result.tool_calls[0]["function"]["arguments"] == {"path": "test.txt"}

    def test_openai_tool_call_invalid_json_args(self):
        """Invalid JSON in arguments is caught and replaced with {}."""
        client = DirectLLMClient(provider="openai", model="gpt-4")
        data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "bad_tool",
                                    "arguments": "not-json",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = client._parse_response(data)
        assert result.tool_calls[0]["function"]["arguments"] == {}

    def test_openai_empty_choices(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        data = {"choices": []}
        result = client._parse_response(data)
        assert result.content == ""
        assert result.error == "empty response"
        assert result.finish_reason == "error"

    def test_openai_no_choices_key(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        data = {}
        result = client._parse_response(data)
        assert result.error == "empty response"

    def test_openai_content_none(self):
        """OpenAI can return content=null; should become empty string."""
        client = DirectLLMClient(provider="openai", model="gpt-4")
        data = {
            "choices": [
                {
                    "message": {"content": None},
                    "finish_reason": "stop",
                }
            ]
        }
        result = client._parse_response(data)
        assert result.content == ""

    def test_gemini_success(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello Gemini"}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = client._parse_response(data)
        assert result.content == "Hello Gemini"
        assert result.finish_reason == "stop"

    def test_gemini_empty_candidates(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        data = {"candidates": []}
        result = client._parse_response(data)
        assert result.error == "empty response"
        assert result.finish_reason == "error"

    def test_gemini_function_call(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Using function"},
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {"path": "test.txt"},
                                }
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = client._parse_response(data)
        assert result.content == "Using function"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "read_file"
        assert result.tool_calls[0]["function"]["arguments"] == {"path": "test.txt"}
        assert result.tool_calls[0]["id"] == "read_file"

    def test_gemini_multiple_parts_text_only(self):
        """Multiple text parts are concatenated."""
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello"}, {"text": " "}, {"text": "World"}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = client._parse_response(data)
        assert result.content == "Hello World"

    def test_ollama_success(self):
        client = DirectLLMClient(provider="ollama", model="llama3")
        data = {
            "message": {"content": "Hello Ollama"},
            "done": True,
        }
        result = client._parse_response(data)
        assert result.content == "Hello Ollama"
        assert result.finish_reason == "stop"

    def test_ollama_with_tool_calls(self):
        client = DirectLLMClient(provider="ollama", model="llama3")
        data = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "test.txt"},
                        }
                    }
                ],
            },
            "done": True,
        }
        result = client._parse_response(data)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "read_file"

    def test_ollama_empty_message(self):
        client = DirectLLMClient(provider="ollama", model="llama3")
        data = {"message": {}}
        result = client._parse_response(data)
        assert result.content == ""


# =============================================================================
# DirectLLMClient — chat method (mocked httpx)
# =============================================================================


class TestDirectLLMClientChat:
    """The chat() method uses httpx.AsyncClient under the hood."""

    @patch("httpx.AsyncClient")
    async def test_chat_success(self, mock_client_cls):
        """Happy path: returns parsed LLMResult."""
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}]
        }
        mock_client.post.return_value = mock_response

        client = DirectLLMClient(provider="openai", model="gpt-4", api_key="sk-123")
        result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "Hello"
        assert result.provider == "openai"
        assert result.model == "gpt-4"
        mock_client.post.assert_awaited_once()

    @patch("httpx.AsyncClient")
    async def test_chat_timeout(self, mock_client_cls):
        """TimeoutException yields an error LLMResult."""
        import httpx

        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = httpx.TimeoutException("timeout")

        client = DirectLLMClient(provider="openai", model="gpt-4", api_key="sk-123")
        result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.content == ""
        assert result.error is not None
        assert "timed out" in result.error
        assert result.finish_reason == "error"

    @patch("httpx.AsyncClient")
    async def test_chat_http_error(self, mock_client_cls):
        """HTTPStatusError yields an error LLMResult with status code info."""
        import httpx

        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        error_response = MagicMock()
        error_response.status_code = 401
        error_response.text = "Unauthorized"
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "401 error", request=MagicMock(), response=error_response
        )

        client = DirectLLMClient(provider="openai", model="gpt-4", api_key="sk-123")
        result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.content == ""
        assert result.error is not None
        assert "HTTP 401" in result.error
        assert result.finish_reason == "error"

    @patch("httpx.AsyncClient")
    async def test_chat_http_error_response_text_fails(self, mock_client_cls):
        """If reading response text fails, the error still has the status code."""
        import httpx

        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 500
        # Make .text property access raise an exception
        type(error_response).text = PropertyMock(side_effect=Exception("read error"))
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "500 error", request=MagicMock(), response=error_response
        )

        client = DirectLLMClient(provider="openai", model="gpt-4", api_key="sk-123")
        result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.content == ""
        assert result.error is not None
        assert "HTTP 500" in result.error
        assert result.finish_reason == "error"

    @patch("httpx.AsyncClient")
    async def test_chat_generic_exception(self, mock_client_cls):
        """Generic exceptions are caught and returned as error."""
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = RuntimeError("something broke")

        client = DirectLLMClient(provider="openai", model="gpt-4", api_key="sk-123")
        result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.content == ""
        assert result.error == "something broke"
        assert result.finish_reason == "error"

    @patch("httpx.AsyncClient")
    async def test_chat_verify_url_and_headers(self, mock_client_cls):
        """Verify the URL, headers, and payload passed to httpx."""
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        mock_client.post.return_value = mock_response

        client = DirectLLMClient(provider="openai", model="gpt-4", api_key="sk-secret")
        await client.chat(messages=[{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=100)

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        url = call_args[0][0]
        kwargs = call_args[1]
        assert url == "https://api.openai.com/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer sk-secret"
        assert kwargs["json"]["model"] == "gpt-4"
        assert kwargs["json"]["temperature"] == 0.5
        assert kwargs["json"]["max_tokens"] == 100

    @patch("httpx.AsyncClient")
    async def test_chat_gemini_url_has_model(self, mock_client_cls):
        """Gemini URL includes the model name in the path."""
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]
        }
        mock_client.post.return_value = mock_response

        client = DirectLLMClient(provider="gemini", model="gemini-pro", api_key="gm-key")
        await client.chat(messages=[{"role": "user", "content": "hi"}])

        call_args = mock_client.post.call_args
        url = call_args[0][0]
        assert "gemini-pro" in url
        assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


# =============================================================================
# _parse_text_tool_calls
# =============================================================================


class TestParseTextToolCalls:
    """Parse tool calls from text-based LLM responses (Hermes format)."""

    def test_single_tool_call(self):
        text = "TOOL_CALL: read_file\n{\"path\": \"test.txt\"}\nEND_TOOL_CALL"
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read_file"
        assert calls[0]["function"]["arguments"] == {"path": "test.txt"}
        assert calls[0]["id"] == "read_file_0"
        assert calls[0]["type"] == "function"

    def test_multiple_tool_calls(self):
        text = (
            "TOOL_CALL: read_file\n{\"path\": \"a.txt\"}\nEND_TOOL_CALL\n"
            "some text\n"
            "TOOL_CALL: write_patch\n{\"path\": \"b.py\", \"content\": \"x\"}\nEND_TOOL_CALL"
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "read_file"
        assert calls[0]["id"] == "read_file_0"
        assert calls[1]["function"]["name"] == "write_patch"
        assert calls[1]["id"] == "write_patch_1"

    def test_no_tool_calls(self):
        text = "Just a normal response without any tool calls."
        calls = _parse_text_tool_calls(text)
        assert calls == []

    def test_empty_text(self):
        calls = _parse_text_tool_calls("")
        assert calls == []

    def test_invalid_json_args(self):
        """When arguments are not valid JSON, fall back to raw string."""
        text = "TOOL_CALL: my_tool\nnot json at all\nEND_TOOL_CALL"
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "my_tool"
        assert calls[0]["function"]["arguments"] == {"raw": "not json at all"}

    def test_malformed_no_name(self):
        """Malformed TOOL_CALL without a valid name is not matched."""
        text = "TOOL_CALL: \n{\"a\": 1}\nEND_TOOL_CALL"
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 0

    def test_missing_end_tag(self):
        """Without END_TOOL_CALL, no match."""
        text = "TOOL_CALL: read_file\n{\"a\": 1}\n"
        calls = _parse_text_tool_calls(text)
        assert calls == []

    def test_tool_call_with_extra_text_around(self):
        text = "Before\nTOOL_CALL: search\n{\"q\": \"test\"}\nEND_TOOL_CALL\nAfter"
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"

    def test_multiline_json_args(self):
        text = "TOOL_CALL: complex_tool\n{\n  \"key\": \"value\",\n  \"nested\": {\"a\": 1}\n}\nEND_TOOL_CALL"
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == {"key": "value", "nested": {"a": 1}}

    def test_tool_call_with_underscore_in_name(self):
        """Tool names can contain underscores."""
        text = "TOOL_CALL: my_custom_tool\n{}\nEND_TOOL_CALL"
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "my_custom_tool"

    def test_tool_call_inside_code_block(self):
        """Tool call markers inside code blocks should still be parsed."""
        text = "```\nTOOL_CALL: test\n{\"x\": 1}\nEND_TOOL_CALL\n```"
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "test"

    def test_case_sensitivity(self):
        """TOOL_CALL must be uppercase."""
        text = "tool_call: read_file\n{\"path\": \"x\"}\nEND_TOOL_CALL"
        calls = _parse_text_tool_calls(text)
        assert calls == []

    def test_extra_whitespace_after_colon(self):
        text = "TOOL_CALL:   read_file\n{\"x\": 1}\nEND_TOOL_CALL"
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read_file"


# =============================================================================
# _strip_tool_calls
# =============================================================================


class TestStripToolCalls:
    """Remove tool call blocks, leaving only natural language."""

    def test_strip_single_tool_call(self):
        text = "Hello\nTOOL_CALL: read_file\n{\"path\": \"x\"}\nEND_TOOL_CALL\nDone"
        result = _strip_tool_calls(text)
        assert "TOOL_CALL:" not in result
        assert "Hello" in result
        assert "Done" in result

    def test_strip_multiple_tool_calls(self):
        text = (
            "Step 1\n"
            "TOOL_CALL: a\n{}\nEND_TOOL_CALL\n"
            "Step 2\n"
            "TOOL_CALL: b\n{}\nEND_TOOL_CALL\n"
            "Done"
        )
        result = _strip_tool_calls(text)
        assert "TOOL_CALL:" not in result
        assert "Step 1" in result
        assert "Step 2" in result
        assert "Done" in result

    def test_no_tool_calls(self):
        text = "Just normal text."
        result = _strip_tool_calls(text)
        assert result == "Just normal text."

    def test_empty_text(self):
        assert _strip_tool_calls("") == ""

    def test_strip_final_marker(self):
        text = "FINAL: here is the result"
        result = _strip_tool_calls(text)
        assert result == "here is the result"

    def test_strip_final_marker_multiline(self):
        text = "Some text\nFINAL: The answer is 42"
        result = _strip_tool_calls(text)
        assert result == "Some text\nThe answer is 42"

    def test_only_tool_call(self):
        text = "TOOL_CALL: foo\n{\"x\": 1}\nEND_TOOL_CALL"
        result = _strip_tool_calls(text)
        assert result == ""

    def test_tool_call_and_final(self):
        text = "TOOL_CALL: foo\n{}\nEND_TOOL_CALL\nFINAL: done"
        result = _strip_tool_calls(text)
        # Tool call removed, FINAL stripped, nothing left
        assert result == "done"

    def test_strips_leading_trailing_whitespace(self):
        text = "  hello world  \n"
        result = _strip_tool_calls(text)
        assert result == "hello world"


# =============================================================================
# get_llm_client factory
# =============================================================================


class TestGetLLMClient:
    """Factory function returns the right client type for each provider."""

    def test_hermes_provider(self):
        client = get_llm_client(provider="hermes")
        assert isinstance(client, HermesLLMClient)
        assert client.profile == "swarm1"
        assert client._timeout == 120

    def test_hermes_provider_custom_profile(self):
        client = get_llm_client(provider="hermes", profile="custom", timeout=60)
        assert isinstance(client, HermesLLMClient)
        assert client.profile == "custom"
        assert client._timeout == 60

    def test_openai_provider(self):
        client = get_llm_client(provider="openai", model="gpt-4", api_key="sk-key")
        assert isinstance(client, DirectLLMClient)
        assert client.provider == "openai"
        assert client.model == "gpt-4"

    def test_gemini_provider(self):
        client = get_llm_client(provider="gemini", model="gemini-pro", api_key="gm-key")
        assert isinstance(client, DirectLLMClient)
        assert client.provider == "gemini"

    def test_ollama_provider(self):
        client = get_llm_client(provider="ollama", model="llama3")
        assert isinstance(client, DirectLLMClient)
        assert client.provider == "ollama"

    def test_openrouter_provider(self):
        client = get_llm_client(provider="openrouter", model="mistral", api_key="or-key")
        assert isinstance(client, DirectLLMClient)
        assert client.provider == "openrouter"

    def test_deepseek_provider(self):
        client = get_llm_client(provider="deepseek", model="deepseek-chat", api_key="ds-key")
        assert isinstance(client, DirectLLMClient)
        assert client.provider == "deepseek"

    def test_custom_base_url_passed_through(self):
        client = get_llm_client(
            provider="openai",
            model="gpt-4",
            api_key="sk-key",
            base_url="https://custom.example.com",
        )
        assert isinstance(client, DirectLLMClient)
        assert client._config["base_url"] == "https://custom.example.com"

    def test_timeout_passed_through(self):
        client = get_llm_client(provider="openai", model="gpt-4", api_key="sk-key", timeout=10)
        assert isinstance(client, DirectLLMClient)
        assert client._timeout == 10

    def test_unknown_provider_without_model_raises(self):
        """Any non-hermes provider requires a model."""
        with pytest.raises(ValueError, match="model is required"):
            get_llm_client(provider="openai")

    def test_unknown_provider_raises(self):
        """Completely unknown provider without model raises."""
        with pytest.raises(ValueError, match="model is required"):
            get_llm_client(provider="nonexistent")

    def test_hermes_provider_ignores_model(self):
        """Hermes provider doesn't require a model parameter."""
        client = get_llm_client(provider="hermes")
        assert isinstance(client, HermesLLMClient)

    def test_hermes_provider_ignores_api_key(self):
        """Hermes provider doesn't use api_key."""
        client = get_llm_client(provider="hermes", api_key="should-be-ignored")
        assert isinstance(client, HermesLLMClient)
        # HermesLLMClient has no _api_key attribute
        assert not hasattr(client, "_api_key")


# =============================================================================
# DirectLLMClient — _convert_tools_to_gemini
# =============================================================================


class TestConvertToolsToGemini:
    """OpenAI tool format → Gemini tool format conversion."""

    def test_basic_conversion(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        tools = [
            {
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            }
        ]
        result = client._convert_tools_to_gemini(tools)
        assert len(result) == 1
        decl = result[0]["functionDeclarations"][0]
        assert decl["name"] == "read_file"
        assert decl["description"] == "Read a file"
        assert decl["parameters"] == {"type": "object", "properties": {"path": {"type": "string"}}}

    def test_empty_tools(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        result = client._convert_tools_to_gemini([])
        assert result == []

    def test_missing_function_keys(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        tools = [{"not_function": {}}]
        result = client._convert_tools_to_gemini(tools)
        assert len(result) == 1
        decl = result[0]["functionDeclarations"][0]
        assert decl["name"] == ""
        assert decl["description"] == ""


# =============================================================================
# Integration-style edge cases
# =============================================================================


class TestHermesLLMClientEdgeCases:
    """Additional edge cases for HermesLLMClient."""

    async def test_chat_with_tool_no_content(self):
        """Tool call with no natural language content."""
        mock_response = MagicMock()
        mock_response.content = "TOOL_CALL: search\n{\"q\": \"test\"}\nEND_TOOL_CALL"
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response):
            client = HermesLLMClient()
            result = await client.chat(messages=[{"role": "user", "content": "search"}])

        assert result.content == ""  # everything was stripped
        assert len(result.tool_calls) == 1

    async def test_chat_multiple_tool_calls(self):
        """Multiple sequential tool calls are parsed."""
        mock_response = MagicMock()
        mock_response.content = (
            "TOOL_CALL: read_file\n{\"path\": \"a.txt\"}\nEND_TOOL_CALL\n"
            "TOOL_CALL: write_patch\n{\"path\": \"b.py\"}\nEND_TOOL_CALL"
        )
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response):
            client = HermesLLMClient()
            result = await client.chat(messages=[{"role": "user", "content": "do it"}])

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["function"]["name"] == "read_file"
        assert result.tool_calls[1]["function"]["name"] == "write_patch"

    async def test_chat_profile_forwarded(self):
        """Profile name and timeout are sent to chat_async."""
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.error = None

        with patch("cortex.hermes_bridge.chat_async", new_callable=AsyncMock, return_value=mock_response) as mock:
            client = HermesLLMClient(profile="dev", timeout=45)
            await client.chat(messages=[{"role": "user", "content": "hi"}])

            mock.assert_awaited_once()
            assert mock.call_args[1]["profile"] == "dev"
            assert mock.call_args[1]["timeout"] == 45


class TestDirectLLMClientEdgeCases:
    """Additional edge cases for DirectLLMClient."""

    def test_ollama_done_false(self):
        """Ollama finish_reason is 'stop' regardless of 'done' field."""
        client = DirectLLMClient(provider="ollama", model="llama3")
        data = {
            "message": {"content": "hi"},
            "done": False,
        }
        result = client._parse_response(data)
        assert result.finish_reason == "stop"

    def test_gemini_no_candidates_key(self):
        client = DirectLLMClient(provider="gemini", model="gemini-pro")
        data = {}
        result = client._parse_response(data)
        assert result.error == "empty response"

    def test_openai_content_empty_string(self):
        client = DirectLLMClient(provider="openai", model="gpt-4")
        data = {
            "choices": [
                {
                    "message": {"content": ""},
                    "finish_reason": "stop",
                }
            ]
        }
        result = client._parse_response(data)
        assert result.content == ""
