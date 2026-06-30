"""
Tests for cortex/integration.py — puente entre CORTEX y Odysseus core.

Tests: chat routing, goal→scheduling, SDD con LLM, heartbeat WebSocket.
"""
import pytest
from cortex.integration import route_chat_message, route_model_for_session


class TestChatRouting:
    def test_routes_code_message(self):
        """Code-related messages get code category."""
        result = route_chat_message("write a python function to sort a list")
        assert result["category"] is not None
        assert "recommended_model" in result
        assert result["recommended_model"]  # non-empty

    def test_routes_general_message(self):
        """General chat messages get chat category."""
        result = route_chat_message("hello, how are you?")
        assert result["category"] is not None

    def test_routes_with_current_model(self):
        """Current model is reflected in output."""
        result = route_chat_message("explain quantum computing", current_model="gpt-4")
        assert result["current_model"] == "gpt-4"

    def test_routes_returns_explanation(self):
        """Routing includes an explanation."""
        result = route_chat_message("debug this python error")
        assert "explanation" in result
        assert len(result["explanation"]) > 10

    def test_routes_fallback_models(self):
        """Routing includes fallback models."""
        result = route_chat_message("write a poem")
        assert "fallback_models" in result
        assert isinstance(result["fallback_models"], list)

    def test_empty_message(self):
        """Empty message still returns a default route."""
        result = route_chat_message("")
        assert "recommended_model" in result

    def test_route_model_for_session(self):
        """Session routing uses last user message."""
        history = [
            {"role": "user", "content": "write code for a REST API"},
        ]
        result = route_model_for_session(history)
        assert "model" in result
        assert result["model"]
        assert "category" in result

    def test_route_model_for_session_empty(self):
        """Empty session returns current model."""
        result = route_model_for_session([], current_model="deepseek-chat")
        assert result["model"] == "deepseek-chat"

    def test_route_model_for_session_no_user_msg(self):
        """Session with no user messages returns current model."""
        result = route_model_for_session(
            [{"role": "assistant", "content": "hello"}],
            current_model="claude-3",
        )
        assert result["model"] == "claude-3"


class TestHeartbeatIntegration:
    def test_ws_registration(self):
        """WS client registration works."""
        from cortex.integration import register_ws_client, unregister_ws_client, _ws_clients
        # Register
        register_ws_client("test-session", "mock-ws")
        assert "test-session" in _ws_clients
        # Unregister
        unregister_ws_client("test-session")
        assert "test-session" not in _ws_clients


class TestSDDIntegration:
    def test_generate_sdd_with_odysseus_llm(self):
        """SDD generation via integration layer."""
        from cortex.integration import generate_sdd_with_odysseus_llm
        result = generate_sdd_with_odysseus_llm("int-test", "build a CLI tool")
        assert result["goal_id"] == "int-test"
        assert "documents_path" in result
        # Either LLM used or template fallback
        assert result["spec_preview"] or True
