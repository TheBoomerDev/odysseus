"""
cortex/hermes_agents.py — Hermes Agent profiles as Odysseus agents.

Maps each Hermes profile to a structured agent with:
  - Capabilities (code, writing, analysis, etc.)
  - Assigned model/provider
  - System prompt / SOUL.md content
  - Compatible task categories

Used by GoalPipeline to dispatch subtasks to the right Hermes profile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cortex.hermes_bridge import (
    HermesProfile,
    HermesResponse,
    chat_async,
    discover_profiles,
    get_profile_system_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent model
# ---------------------------------------------------------------------------


@dataclass
class HermesAgent:
    """A Hermes profile mapped to an Odysseus agent with capabilities."""

    name: str
    model: str
    provider: str
    system_prompt: str
    capabilities: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)
    gateway_status: str = "stopped"
    skills: List[str] = field(default_factory=list)
    priority: int = 5  # 0=highest, 10=lowest


# ---------------------------------------------------------------------------
# Capability inference
# ---------------------------------------------------------------------------

_CAPABILITY_MAP: Dict[str, Dict[str, Any]] = {
    # Dev / code agents
    "swarm1": {
        "capabilities": ["code", "system_design", "debug", "architecture", "cli_automation"],
        "categories": ["code_generation", "code_review", "debugging", "analysis"],
        "priority": 1,
    },
    "swarm4": {
        "capabilities": ["code", "fullstack", "web_dev"],
        "categories": ["code_generation", "testing"],
        "priority": 3,
    },
    "boomerdev": {
        "capabilities": ["code", "fullstack", "react", "mobile"],
        "categories": ["code_generation", "code_review"],
        "priority": 2,
    },
    "devteam": {
        "capabilities": ["code", "review", "testing", "devops"],
        "categories": ["code_review", "testing", "analysis"],
        "priority": 2,
    },
    "mobile-dev": {
        "capabilities": ["code", "mobile", "react_native"],
        "categories": ["code_generation"],
        "priority": 3,
    },
    # Content / writing
    "novelist-atelier": {
        "capabilities": ["writing", "creative", "storytelling", "editing"],
        "categories": ["writing", "creative"],
        "priority": 1,
    },
    "content": {
        "capabilities": ["writing", "content", "marketing", "social_media"],
        "categories": ["writing", "content_creation"],
        "priority": 2,
    },
    "content-factory": {
        "capabilities": ["content", "seo", "writing"],
        "categories": ["writing", "content_creation"],
        "priority": 2,
    },
    "seo-research": {
        "capabilities": ["research", "seo", "analysis"],
        "categories": ["research", "analysis"],
        "priority": 3,
    },
    # Strategy / management
    "strategist": {
        "capabilities": ["strategy", "planning", "analysis"],
        "categories": ["analysis", "planning"],
        "priority": 2,
    },
    "cargoffer": {
        "capabilities": ["business", "cargo", "logistics", "management"],
        "categories": ["analysis", "planning"],
        "priority": 3,
    },
    "growth": {
        "capabilities": ["growth", "marketing", "analytics"],
        "categories": ["analysis", "content_creation"],
        "priority": 3,
    },
    # Domain-specific
    "legal": {
        "capabilities": ["legal", "compliance", "document_review"],
        "categories": ["analysis", "document_review"],
        "priority": 3,
    },
    "angotrucks": {
        "capabilities": ["logistics", "trucking", "fullstack"],
        "categories": ["code_generation", "analysis"],
        "priority": 3,
    },
    "demonk": {
        "capabilities": ["data", "analytics", "visualization"],
        "categories": ["analysis", "code_generation"],
        "priority": 3,
    },
    "hornslegend": {
        "capabilities": ["music", "audio", "creative"],
        "categories": ["creative", "analysis"],
        "priority": 3,
    },
    # Quality / testing
    "qa": {
        "capabilities": ["testing", "qa", "verification", "bug_detection"],
        "categories": ["testing", "code_review", "debugging"],
        "priority": 1,
    },
}

# Default capability for unknown profiles
_DEFAULT_CAPABILITIES = ["general", "chat"]
_DEFAULT_CATEGORIES = ["chat", "analysis"]


def _infer_capabilities(profile: HermesProfile) -> List[str]:
    """Infer capabilities for a profile based on its name and config."""
    mapping = _CAPABILITY_MAP.get(profile.name)
    if mapping:
        return mapping["capabilities"]
    # Fallback: use model tier as capability hint
    model_lower = profile.model.lower()
    if any(k in model_lower for k in ["code", "deepseek", "qwen"]):
        return ["code", "general"]
    if "nemotron" in model_lower or "minimax" in model_lower:
        return ["general", "chat"]
    return _DEFAULT_CAPABILITIES.copy()


def _infer_categories(profile: HermesProfile) -> List[str]:
    """Infer preferred routing categories."""
    mapping = _CAPABILITY_MAP.get(profile.name)
    if mapping:
        return mapping["categories"]
    return _DEFAULT_CATEGORIES.copy()


def _infer_priority(profile: HermesProfile) -> int:
    """Infer dispatch priority (lower = higher priority)."""
    mapping = _CAPABILITY_MAP.get(profile.name)
    if mapping:
        return mapping["priority"]
    return 5


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------


class HermesAgentRegistry:
    """Registry of all Hermes Agent profiles available as Odysseus agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, HermesAgent] = {}
        self._refreshed_at: float = 0.0

    def refresh(self) -> None:
        """Refresh the agent list from Hermes profiles."""
        import time

        profiles = discover_profiles()
        self._agents = {}

        for p in profiles:
            system_prompt = get_profile_system_prompt(p.name)
            agent = HermesAgent(
                name=p.name,
                model=p.model,
                provider=p.provider,
                system_prompt=system_prompt[:1000] if system_prompt else "",
                capabilities=_infer_capabilities(p),
                preferred_categories=_infer_categories(p),
                gateway_status=p.gateway_status,
                skills=p.skills,
                priority=_infer_priority(p),
            )
            self._agents[p.name] = agent

        self._refreshed_at = time.time()
        logger.info(
            "Refreshed HermesAgentRegistry: %d agents loaded", len(self._agents)
        )

    @property
    def agents(self) -> Dict[str, HermesAgent]:
        if not self._agents:
            self.refresh()
        return self._agents

    def get_agent(self, name: str) -> Optional[HermesAgent]:
        return self.agents.get(name)

    def find_agents_by_capability(self, capability: str) -> List[HermesAgent]:
        """Find agents that have a specific capability."""
        return [
            a for a in self.agents.values() if capability in a.capabilities
        ]

    def find_agents_by_category(self, category: str) -> List[HermesAgent]:
        """Find agents that prefer a specific routing category."""
        return [
            a for a in self.agents.values()
            if category in a.preferred_categories
        ]

    def best_agent_for_category(self, category: str) -> Optional[HermesAgent]:
        """Get the highest-priority agent for a category."""
        candidates = self.find_agents_by_category(category)
        if not candidates:
            return None
        return min(candidates, key=lambda a: a.priority)

    async def dispatch(
        self,
        prompt: str,
        agent_name: str,
        timeout: int = 120,
    ) -> HermesResponse:
        """Dispatch a prompt to a specific Hermes agent/profile."""
        return await chat_async(
            prompt=prompt,
            profile=agent_name,
            timeout=timeout,
        )


# Singleton
_registry: Optional[HermesAgentRegistry] = None


def get_registry() -> HermesAgentRegistry:
    global _registry
    if _registry is None:
        _registry = HermesAgentRegistry()
    return _registry
