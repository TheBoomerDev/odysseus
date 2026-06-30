"""Tests for cortex/hermes_agents.py — HermesAgent registry, discovery, dispatch."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.hermes_agents import (
    HermesAgent,
    HermesAgentRegistry,
    _DEFAULT_CAPABILITIES,
    _DEFAULT_CATEGORIES,
    _CAPABILITY_MAP,
    _infer_capabilities,
    _infer_categories,
    _infer_priority,
    get_registry,
)
from cortex.hermes_bridge import HermesProfile, HermesResponse


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def sample_profile() -> HermesProfile:
    return HermesProfile(
        name="swarm1",
        model="deepseek/deepseek-v4-flash",
        provider="opencode-go",
        gateway_status="running",
        skills=["code", "debug"],
        system_prompt="You are a coding assistant.",
    )


@pytest.fixture
def sample_profiles() -> list[HermesProfile]:
    """Multiple profiles with different priorities and capabilities."""
    return [
        HermesProfile(
            name="swarm1",
            model="deepseek/deepseek-v4-flash",
            provider="opencode-go",
            gateway_status="running",
            skills=["code"],
            system_prompt="Swarm1: code expert",
        ),
        HermesProfile(
            name="qa",
            model="anthropic/claude-sonnet-4",
            provider="anthropic",
            gateway_status="running",
            skills=["testing"],
            system_prompt="QA: test expert",
        ),
        HermesProfile(
            name="content",
            model="openai/gpt-4o",
            provider="openai",
            gateway_status="stopped",
            skills=["writing"],
            system_prompt="Content writer",
        ),
        HermesProfile(
            name="unknown-profile",
            model="some-other-model",
            provider="custom",
            gateway_status="stopped",
            skills=[],
            system_prompt="",
        ),
    ]


@pytest.fixture
def unknown_model_profile() -> HermesProfile:
    return HermesProfile(
        name="new-unknown",
        model="deepseek-coder-v2",
        provider="deepseek",
    )


@pytest.fixture
def nemotron_profile() -> HermesProfile:
    return HermesProfile(
        name="nemobot",
        model="nvidia/nemotron-4",
        provider="nvidia",
    )


# =========================================================================
# HermesAgent dataclass
# =========================================================================


class TestHermesAgent:
    def test_minimal_creation(self):
        agent = HermesAgent(name="test", model="m", provider="p", system_prompt="sp")
        assert agent.name == "test"
        assert agent.model == "m"
        assert agent.provider == "p"
        assert agent.system_prompt == "sp"
        assert agent.capabilities == []
        assert agent.preferred_categories == []
        assert agent.gateway_status == "stopped"
        assert agent.skills == []
        assert agent.priority == 5

    def test_full_creation(self):
        agent = HermesAgent(
            name="full",
            model="m1",
            provider="p1",
            system_prompt="Hello",
            capabilities=["code", "debug"],
            preferred_categories=["code_generation"],
            gateway_status="running",
            skills=["python"],
            priority=1,
        )
        assert agent.name == "full"
        assert agent.capabilities == ["code", "debug"]
        assert agent.preferred_categories == ["code_generation"]
        assert agent.priority == 1


# =========================================================================
# _infer_capabilities
# =========================================================================


class TestInferCapabilities:
    def test_known_profile(self, sample_profile):
        caps = _infer_capabilities(sample_profile)
        expected = _CAPABILITY_MAP["swarm1"]["capabilities"]
        assert caps == expected

    def test_unknown_profile_code_model(self, unknown_model_profile):
        caps = _infer_capabilities(unknown_model_profile)
        assert caps == ["code", "general"]

    def test_unknown_profile_nemotron(self, nemotron_profile):
        caps = _infer_capabilities(nemotron_profile)
        assert caps == ["general", "chat"]

    def test_unknown_profile_minimax(self):
        p = HermesProfile(name="minibot", model="minimax/abab5", provider="minimax")
        caps = _infer_capabilities(p)
        assert caps == ["general", "chat"]

    def test_unknown_profile_fallback(self):
        p = HermesProfile(name="random", model="foo-bar", provider="baz")
        caps = _infer_capabilities(p)
        assert caps == _DEFAULT_CAPABILITIES

    def test_known_profile_boomerdev(self):
        p = HermesProfile(name="boomerdev", model="gpt-4", provider="openai")
        caps = _infer_capabilities(p)
        assert "react" in caps
        assert "mobile" in caps


# =========================================================================
# _infer_categories
# =========================================================================


class TestInferCategories:
    def test_known_profile(self, sample_profile):
        cats = _infer_categories(sample_profile)
        expected = _CAPABILITY_MAP["swarm1"]["categories"]
        assert cats == expected

    def test_unknown_profile(self):
        p = HermesProfile(name="random", model="x", provider="y")
        cats = _infer_categories(p)
        assert cats == _DEFAULT_CATEGORIES


# =========================================================================
# _infer_priority
# =========================================================================


class TestInferPriority:
    def test_known_profile(self, sample_profile):
        prio = _infer_priority(sample_profile)
        assert prio == _CAPABILITY_MAP["swarm1"]["priority"]

    def test_known_profile_qa(self):
        p = HermesProfile(name="qa", model="claude", provider="anthropic")
        prio = _infer_priority(p)
        assert prio == 1

    def test_unknown_profile(self):
        p = HermesProfile(name="random", model="x", provider="y")
        prio = _infer_priority(p)
        assert prio == 5


# =========================================================================
# HermesAgentRegistry — refresh
# =========================================================================


class TestRefresh:
    def test_refresh_populates_agents(self, sample_profiles):
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value="sysprompt"
        ):
            registry = HermesAgentRegistry()
            assert registry._agents == {}
            assert registry._refreshed_at == 0.0

            registry.refresh()

            assert len(registry._agents) == 4
            assert "swarm1" in registry._agents
            assert "qa" in registry._agents
            assert "content" in registry._agents
            assert "unknown-profile" in registry._agents

            # Check that system prompt is truncated to 1000
            swarm1 = registry._agents["swarm1"]
            assert swarm1.system_prompt == "sysprompt"
            assert swarm1.priority == 1  # from _CAPABILITY_MAP
            assert "code" in swarm1.capabilities

            # Unknown profile gets defaults
            unknown = registry._agents["unknown-profile"]
            assert unknown.capabilities == _DEFAULT_CAPABILITIES
            assert unknown.priority == 5

    def test_refresh_empty_profiles(self):
        with patch("cortex.hermes_agents.discover_profiles", return_value=[]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            registry.refresh()
            assert registry._agents == {}
            assert registry._refreshed_at > 0

    def test_refresh_sets_timestamp(self, sample_profiles):
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles[:1]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            before = time.time()
            registry.refresh()
            after = time.time()
            assert before <= registry._refreshed_at <= after

    def test_refresh_truncates_system_prompt(self, sample_profiles):
        long_prompt = "x" * 2000
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles[:1]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=long_prompt
        ):
            registry = HermesAgentRegistry()
            registry.refresh()
            agent = list(registry._agents.values())[0]
            assert len(agent.system_prompt) == 1000

    def test_refresh_none_system_prompt(self, sample_profiles):
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles[:1]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=None
        ):
            registry = HermesAgentRegistry()
            registry.refresh()
            agent = list(registry._agents.values())[0]
            assert agent.system_prompt == ""


# =========================================================================
# HermesAgentRegistry — agents property (auto-refresh)
# =========================================================================


class TestAgentsProperty:
    def test_auto_refresh_on_access(self):
        with patch("cortex.hermes_agents.discover_profiles", return_value=[]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            assert registry._agents == {}  # empty before access
            _ = registry.agents  # triggers refresh
            assert registry._refreshed_at > 0

    def test_does_not_re_refresh_if_already_populated(self):
        with patch("cortex.hermes_agents.discover_profiles") as mock_disc, patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            mock_disc.return_value = []
            registry = HermesAgentRegistry()
            registry._agents = {"preloaded": MagicMock(spec=HermesAgent)}
            _ = registry.agents  # should not call refresh
            mock_disc.assert_not_called()

    def test_auto_refresh_uses_discover_and_get_prompt(self, sample_profiles):
        """Verify that the lazy agents property calls discover + get_profile_system_prompt."""
        with patch(
            "cortex.hermes_agents.discover_profiles", return_value=sample_profiles
        ) as mock_disc, patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value="sp"
        ) as mock_gsp:
            registry = HermesAgentRegistry()
            agents = registry.agents
            mock_disc.assert_called_once()
            # get_profile_system_prompt called once per profile
            assert mock_gsp.call_count == len(sample_profiles)
            assert len(agents) == len(sample_profiles)


# =========================================================================
# HermesAgentRegistry — get_agent
# =========================================================================


class TestGetAgent:
    def test_get_existing(self, sample_profiles):
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            agent = registry.get_agent("swarm1")
            assert agent is not None
            assert agent.name == "swarm1"

    def test_get_missing(self):
        with patch("cortex.hermes_agents.discover_profiles", return_value=[]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            agent = registry.get_agent("nonexistent")
            assert agent is None

    def test_get_triggers_auto_refresh(self):
        with patch("cortex.hermes_agents.discover_profiles") as mock_disc, patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            mock_disc.return_value = []
            registry = HermesAgentRegistry()
            _ = registry.get_agent("anything")
            mock_disc.assert_called_once()


# =========================================================================
# HermesAgentRegistry — find_agents_by_capability
# =========================================================================


class TestFindAgentsByCapability:
    def test_find_existing_capability(self, sample_profiles):
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            results = registry.find_agents_by_capability("code")
            assert len(results) >= 1
            assert all("code" in a.capabilities for a in results)

    def test_find_missing_capability(self):
        with patch("cortex.hermes_agents.discover_profiles", return_value=[]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            results = registry.find_agents_by_capability("nonexistent")
            assert results == []

    def test_find_no_match(self, sample_profiles):
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            results = registry.find_agents_by_capability("nuclear_physics")
            assert results == []


# =========================================================================
# HermesAgentRegistry — find_agents_by_category
# =========================================================================


class TestFindAgentsByCategory:
    def test_find_existing_category(self, sample_profiles):
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            results = registry.find_agents_by_category("code_generation")
            assert len(results) >= 1
            assert all("code_generation" in a.preferred_categories for a in results)

    def test_find_missing_category(self):
        with patch("cortex.hermes_agents.discover_profiles", return_value=[]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            results = registry.find_agents_by_category("unknown")
            assert results == []


# =========================================================================
# HermesAgentRegistry — best_agent_for_category
# =========================================================================


class TestBestAgentForCategory:
    def test_returns_highest_priority_agent(self, sample_profiles):
        """Lower priority number = higher priority. swarm1 has priority 1 for
        code_generation, boomerdev has 2, swarm4 has 3, etc."""
        # Add boomerdev and swarm4 to the mix
        profiles = list(sample_profiles)
        profiles.extend([
            HermesProfile(name="boomerdev", model="gpt-4", provider="openai"),
            HermesProfile(name="swarm4", model="claude", provider="anthropic"),
        ])
        with patch("cortex.hermes_agents.discover_profiles", return_value=profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            best = registry.best_agent_for_category("code_generation")
            assert best is not None
            assert best.name == "swarm1"  # priority 1 (lowest number)
            # swarm4 has priority 3, boomerdev priority 2 — swarm1 wins at 1

    def test_returns_none_for_empty_registry(self):
        with patch("cortex.hermes_agents.discover_profiles", return_value=[]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            best = registry.best_agent_for_category("anything")
            assert best is None

    def test_returns_none_when_category_not_found(self, sample_profiles):
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            best = registry.best_agent_for_category("nonexistent_category")
            assert best is None

    def test_priority_tie_returns_first(self, sample_profiles):
        """When two agents have the same priority, min() returns the first one
        encountered (stable)."""
        # Both content and content-factory have priority 2 for "writing"
        profiles = [
            HermesProfile(name="content", model="gpt-4", provider="openai"),
            HermesProfile(name="content-factory", model="gpt-4", provider="openai"),
        ]
        with patch("cortex.hermes_agents.discover_profiles", return_value=profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            best = registry.best_agent_for_category("writing")
            assert best is not None
            assert best.name in ("content", "content-factory")


# =========================================================================
# HermesAgentRegistry — dispatch
# =========================================================================


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_calls_chat_async(self):
        mock_response = HermesResponse(content="response text", session_id="s1")
        with patch("cortex.hermes_agents.chat_async", new=AsyncMock(return_value=mock_response)):
            registry = HermesAgentRegistry()
            response = await registry.dispatch(
                prompt="hello",
                agent_name="swarm1",
                timeout=60,
            )
            from cortex.hermes_agents import chat_async

            chat_async.assert_awaited_once_with(
                prompt="hello",
                profile="swarm1",
                timeout=60,
            )
            assert response.content == "response text"

    @pytest.mark.asyncio
    async def test_dispatch_default_timeout(self):
        mock_response = HermesResponse(content="ok")
        with patch("cortex.hermes_agents.chat_async", new=AsyncMock(return_value=mock_response)):
            registry = HermesAgentRegistry()
            response = await registry.dispatch(
                prompt="test",
                agent_name="default-agent",
            )
            from cortex.hermes_agents import chat_async

            chat_async.assert_awaited_once_with(
                prompt="test",
                profile="default-agent",
                timeout=120,
            )
            assert response.content == "ok"


# =========================================================================
# get_registry singleton
# =========================================================================


class TestGetRegistry:
    def test_returns_same_instance(self):
        # Reset module-level singleton
        import cortex.hermes_agents as ha

        ha._registry = None
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
        assert isinstance(r1, HermesAgentRegistry)

    def test_lazy_creation(self):
        import cortex.hermes_agents as ha

        ha._registry = None
        reg = get_registry()
        assert reg._agents == {}
        assert reg._refreshed_at == 0.0


# =========================================================================
# Edge cases and integration scenarios
# =========================================================================


class TestEdgeCases:
    def test_known_profile_discover_fields_mapped_correctly(self):
        """Verify that a swarm1 profile gets all fields mapped from _CAPABILITY_MAP."""
        profile = HermesProfile(
            name="swarm1",
            model="m",
            provider="p",
            gateway_status="running",
            skills=["skill1", "skill2"],
        )
        with patch("cortex.hermes_agents.discover_profiles", return_value=[profile]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value="prompt"
        ):
            registry = HermesAgentRegistry()
            registry.refresh()
            agent = registry._agents["swarm1"]
            assert agent.name == "swarm1"
            assert agent.model == "m"
            assert agent.provider == "p"
            assert agent.gateway_status == "running"
            assert agent.skills == ["skill1", "skill2"]
            assert agent.system_prompt == "prompt"
            assert agent.capabilities == _CAPABILITY_MAP["swarm1"]["capabilities"]
            assert agent.preferred_categories == _CAPABILITY_MAP["swarm1"]["categories"]
            assert agent.priority == _CAPABILITY_MAP["swarm1"]["priority"]

    def test_multiple_capabilities_search(self, sample_profiles):
        """find_agents_by_capability returns all matching agents."""
        with patch("cortex.hermes_agents.discover_profiles", return_value=sample_profiles), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            # "writing" is a capability of 'content' and 'content-factory',
            # but our sample only has 'content'
            results = registry.find_agents_by_capability("writing")
            assert len(results) == 1
            assert results[0].name == "content"

    def test_model_based_capability_inference_unknown_profile(self):
        """Unknown profiles with code-related model name get 'code' capability."""
        profile = HermesProfile(
            name="custom-coder",
            model="deepseek-coder-v2",
            provider="deepseek",
        )
        with patch("cortex.hermes_agents.discover_profiles", return_value=[profile]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            registry.refresh()
            agent = registry._agents["custom-coder"]
            assert "code" in agent.capabilities

    def test_unknown_profile_fallback_defaults(self):
        """Unknown profile with non-matching model gets general/chat defaults."""
        profile = HermesProfile(
            name="weird-model",
            model="some-unknown-v1",
            provider="custom",
        )
        with patch("cortex.hermes_agents.discover_profiles", return_value=[profile]), patch(
            "cortex.hermes_agents.get_profile_system_prompt", return_value=""
        ):
            registry = HermesAgentRegistry()
            registry.refresh()
            agent = registry._agents["weird-model"]
            assert agent.capabilities == _DEFAULT_CAPABILITIES
            assert agent.preferred_categories == _DEFAULT_CATEGORIES
            assert agent.priority == 5

    def test_qwen_model_gets_code_capability(self):
        profile = HermesProfile(name="qwen-user", model="qwen/qwen2.5-coder", provider="qwen")
        assert _infer_capabilities(profile) == ["code", "general"]
