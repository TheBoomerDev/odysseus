"""
cortex/codegen/llm.py — Unified LLM client interface.

Supports two provider types:
  - hermes: Hermes Agent CLI (via hermes_bridge.chat_async)
  - direct: OpenAI / Gemini / Ollama / OpenRouter via HTTP

Usage:
    client = get_llm_client(provider="openai", model="gpt-4o")
    result = await client.chat([{"role": "user", "content": "hello"}])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified response type
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    """Unified response from any LLM provider."""

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    error: Optional[str] = None
    provider: str = ""
    model: str = ""


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------


class LLMClient:
    """Base class for LLM clients."""

    provider: str = ""
    model: str = ""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResult:
        """Send a chat completion request.

        Args:
            messages: List of {'role': ..., 'content': ...}
            tools: Optional list of tool definitions (OpenAI tool format)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLMResult with content and/or tool_calls
        """
        raise NotImplementedError

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        temperature: float = 0.2,
    ) -> LLMResult:
        """Send a chat request expecting a structured JSON response.

        Uses the response_schema as a hint (via system prompt) for providers
        that don't support native structured output.
        """
        schema_hint = (
            "You MUST respond with valid JSON matching this schema:\n"
            f"{json.dumps(response_schema, indent=2)}"
        )
        enhanced = list(messages)
        if enhanced and enhanced[0].get("role") == "system":
            enhanced[0]["content"] += "\n\n" + schema_hint
        else:
            enhanced.insert(0, {"role": "system", "content": schema_hint})

        return await self.chat(enhanced, temperature=temperature)


# ---------------------------------------------------------------------------
# Hermes LLM Client (CLI-based)
# ---------------------------------------------------------------------------


class HermesLLMClient(LLMClient):
    """LLM client that calls Hermes Agent CLI via hermes_bridge."""

    provider = "hermes"

    def __init__(self, profile: str = "swarm1", timeout: int = 120):
        self.profile = profile
        self._timeout = timeout
        self.model = f"hermes/{profile}"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResult:
        from cortex.hermes_bridge import chat_async

        # Build a single prompt from messages
        prompt_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        if tools:
            prompt_parts.append("\n\nAvailable tools:")
            for tool in tools:
                name = tool.get("function", {}).get("name", "unknown")
                desc = tool.get("function", {}).get("description", "")
                prompt_parts.append(f"  - {name}: {desc}")
            prompt_parts.append(
                "\nRespond with a tool call when needed using "
                "TOOL_CALL: <name>\\n<json args>\\nEND_TOOL_CALL format."
            )

        prompt = "\n".join(prompt_parts)

        resp = await chat_async(
            prompt=prompt,
            profile=self.profile,
            timeout=self._timeout,
        )

        if resp.error:
            return LLMResult(
                content="",
                error=resp.error,
                provider=self.provider,
                model=self.model,
                finish_reason="error",
            )

        # Parse tool calls from text response
        tool_calls = _parse_text_tool_calls(resp.content)
        clean_content = _strip_tool_calls(resp.content)

        return LLMResult(
            content=clean_content,
            tool_calls=tool_calls,
            provider=self.provider,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Direct HTTP LLM Client
# ---------------------------------------------------------------------------


class DirectLLMClient(LLMClient):
    """LLM client that calls provider APIs directly via HTTP.

    Supports: openai, gemini, ollama, openrouter, deepseek, glm
    """

    PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "chat_endpoint": "/chat/completions",
            "api_key_header": "Authorization",
            "api_key_prefix": "Bearer ",
            "model_key": "model",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "chat_endpoint": "/chat/completions",
            "api_key_header": "Authorization",
            "api_key_prefix": "Bearer ",
            "model_key": "model",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "chat_endpoint": "/chat/completions",
            "api_key_header": "Authorization",
            "api_key_prefix": "Bearer ",
            "model_key": "model",
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "chat_endpoint": "/models/{model}:generateContent",
            "api_key_header": "x-goog-api-key",
            "api_key_prefix": "",
            "model_key": "",
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "chat_endpoint": "/api/chat",
            "api_key_header": "",
            "api_key_prefix": "",
            "model_key": "model",
        },
    }

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(
                f"Unsupported provider: {provider!r}. "
                f"Supported: {list(self.PROVIDER_CONFIGS.keys())}"
            )
        self.provider = provider
        self.model = model
        self._api_key = api_key or ""
        self._timeout = timeout
        self._config = dict(self.PROVIDER_CONFIGS[provider])
        if base_url:
            self._config["base_url"] = base_url.rstrip("/")

    def _build_url(self) -> str:
        """Build the API endpoint URL."""
        base = self._config["base_url"]
        endpoint = self._config["chat_endpoint"]
        if "{model}" in endpoint:
            endpoint = endpoint.replace("{model}", self.model)
        return f"{base}{endpoint}"

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        key_header = self._config.get("api_key_header", "")
        key_prefix = self._config.get("api_key_prefix", "")
        if key_header and self._api_key:
            headers[key_header] = f"{key_prefix}{self._api_key}"
        return headers

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Build request payload for the provider."""
        provider = self.provider

        if provider == "gemini":
            return self._build_gemini_payload(messages, tools, temperature, max_tokens)
        if provider == "ollama":
            return self._build_ollama_payload(messages, tools, temperature)

        # OpenAI-compatible (openai, openrouter, deepseek)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _build_gemini_payload(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Build Gemini-specific payload."""
        # Convert OpenAI-style messages to Gemini format
        contents: List[Dict[str, Any]] = []
        system_prompt = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt += content + "\n"
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt.strip()}]
            }
        if tools:
            payload["tools"] = self._convert_tools_to_gemini(tools)
        return payload

    def _convert_tools_to_gemini(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Gemini tool format."""
        gemini_tools = []
        for tool in tools:
            func = tool.get("function", {})
            gemini_tools.append({
                "functionDeclarations": [{
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }]
            })
        return gemini_tools

    def _build_ollama_payload(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Build Ollama-specific payload."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> LLMResult:
        """Parse provider response into unified LLMResult."""
        provider = self.provider

        if provider == "gemini":
            return self._parse_gemini_response(data)
        if provider == "ollama":
            return self._parse_ollama_response(data)

        # OpenAI-compatible
        choices = data.get("choices", [])
        if not choices:
            return LLMResult(
                content="",
                error="empty response",
                provider=self.provider,
                model=self.model,
                finish_reason="error",
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        finish_reason = choice.get("finish_reason", "stop")

        tool_calls = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                },
            })

        return LLMResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            provider=self.provider,
            model=self.model,
        )

    def _parse_gemini_response(self, data: Dict[str, Any]) -> LLMResult:
        """Parse Gemini response."""
        candidates = data.get("candidates", [])
        if not candidates:
            return LLMResult(
                content="",
                error="empty response",
                provider=self.provider,
                model=self.model,
                finish_reason="error",
            )

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [{}])
        content = "".join(p.get("text", "") for p in content_parts)
        finish_reason = candidate.get("finishReason", "stop")

        # Check for function calls
        tool_calls = []
        for part in content_parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": fc.get("name", "unknown"),
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": fc.get("args", {}),
                    },
                })

        return LLMResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason.lower(),
            provider=self.provider,
            model=self.model,
        )

    def _parse_ollama_response(self, data: Dict[str, Any]) -> LLMResult:
        """Parse Ollama response."""
        message = data.get("message", {})
        content = message.get("content", "") or ""
        finish_reason = "stop" if not data.get("done") else "stop"

        tool_calls = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            tool_calls.append({
                "id": func.get("name", "unknown"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", {}),
                },
            })

        return LLMResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            provider=self.provider,
            model=self.model,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResult:
        import httpx

        url = self._build_url()
        headers = self._build_headers()
        payload = self._build_payload(messages, tools, temperature, max_tokens)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            return LLMResult(
                content="",
                error=f"request timed out after {self._timeout}s",
                provider=self.provider,
                model=self.model,
                finish_reason="error",
            )
        except httpx.HTTPStatusError as exc:
            error_body = ""
            try:
                error_body = exc.response.text[:500]
            except Exception:
                pass
            return LLMResult(
                content="",
                error=f"HTTP {exc.response.status_code}: {error_body}",
                provider=self.provider,
                model=self.model,
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResult(
                content="",
                error=str(exc),
                provider=self.provider,
                model=self.model,
                finish_reason="error",
            )

        return self._parse_response(data)


# ---------------------------------------------------------------------------
# Text-based tool call parsing (for Hermes Agent and other text-only LLMs)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = None  # lazy import to avoid circular deps at module level


def _parse_text_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text-based LLM responses.

    Expects format:
        TOOL_CALL: <name>
        {"arg1": "val1", ...}
        END_TOOL_CALL
    """
    import re

    pattern = r"TOOL_CALL:\s*(\w+)\s*\n(.*?)END_TOOL_CALL"
    matches = re.findall(pattern, text, re.DOTALL)

    tool_calls = []
    for name, args_text in matches:
        try:
            args = json.loads(args_text.strip())
        except json.JSONDecodeError:
            args = {"raw": args_text.strip()}
        tool_calls.append({
            "id": f"{name}_{len(tool_calls)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args,
            },
        })

    return tool_calls


def _strip_tool_calls(text: str) -> str:
    """Remove tool call blocks from text, leaving only natural language."""
    import re

    pattern = r"TOOL_CALL:\s*\w+\s*\n.*?END_TOOL_CALL"
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL)
    # Also strip FINAL: markers
    cleaned = re.sub(r"^FINAL:\s*", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_client(
    provider: str = "hermes",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    profile: str = "swarm1",
    timeout: int = 120,
) -> LLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: 'hermes', 'openai', 'gemini', 'ollama', 'openrouter', 'deepseek'
        model: Model name (required for direct providers)
        api_key: API key (required for cloud providers)
        base_url: Custom base URL (optional)
        profile: Hermes profile name (only for provider='hermes')
        timeout: Request timeout in seconds

    Returns:
        LLMClient instance
    """
    if provider == "hermes":
        return HermesLLMClient(profile=profile, timeout=timeout)

    if not model:
        raise ValueError(f"model is required for provider={provider!r}")

    return DirectLLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
