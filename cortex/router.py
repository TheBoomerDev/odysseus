"""
cortex/router.py — Multi-agent router/scorer.

Scores available agents by quality × cost × recency × affinity × diversity
to pick the best agent for a given prompt. Port of CORTEX router/scorer.ts.

Integrates with Odysseus's existing agent/tool system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentScore:
    id: str
    name: str
    quality: float  # 0..1 — how well this agent matches the task
    cost: float  # 0..1 — lower is cheaper (inverted)
    recency: float  # 0..1 — how recently used
    affinity: float  # 0..1 — domain/task affinity (from history)
    diversity: float  # 0..1 — how different from other candidates
    score: float = 0.0  # weighted composite

    CARRIER_BIAS = 0.05


@dataclass
class RoutingDecision:
    winner: str
    runner_up: Optional[str] = None
    ranked: List[AgentScore] = field(default_factory=list)
    explanation: str = ""


# Default weights (sums to 1.0)
DEFAULT_WEIGHTS = {
    "quality": 0.35,
    "cost": 0.20,
    "recency": 0.15,
    "affinity": 0.20,
    "diversity": 0.10,
}


# Task signature patterns — map prompt keywords to capability tags
_TASK_SIGNATURES: Dict[str, List[str]] = {
    "code": ["code_edit", "code_review", "multi_file_diff"],
    "test": ["test_generation", "test_execution"],
    "refactor": ["code_edit", "architecture_design"],
    "debug": ["code_review", "adversarial_review"],
    "research": ["web_search", "doc_synthesis"],
    "architect": ["architecture_design", "trade_off_analysis"],
    "doc": ["doc_generation"],
    "security": ["adversarial_review", "code_review"],
    "deploy": ["shell_execution", "docker"],
    "data": ["schema_design", "data_analysis"],
}


def _detect_capabilities(prompt: str) -> List[str]:
    """Infer required capabilities from prompt text."""
    prompt_lower = prompt.lower()
    caps: List[str] = []
    for keyword, sig_caps in _TASK_SIGNATURES.items():
        if keyword in prompt_lower:
            caps.extend(sig_caps)
    if not caps:
        caps = ["code_edit"]  # safe default
    return list(set(caps))


def score_agent(
    agent_id: str,
    agent_name: str,
    capabilities: List[str],
    required_caps: List[str],
    cost_tier: float = 0.5,
    last_used_hours: Optional[float] = None,
    task_affinity: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
) -> AgentScore:
    """Score a single agent for a set of required capabilities."""
    w = weights or DEFAULT_WEIGHTS

    # Quality: capability overlap
    if required_caps:
        overlap = len(set(capabilities) & set(required_caps))
        quality = min(1.0, overlap / max(len(required_caps), 1))
    else:
        quality = 0.5

    # Cost: inverted (higher cost_tier = more expensive = lower score)
    cost = 1.0 - min(1.0, cost_tier)

    # Recency: recently used agents get a boost
    if last_used_hours is not None:
        recency = max(0.0, 1.0 - (last_used_hours / 168.0))  # 1 week decay
    else:
        recency = 0.3  # never used = slight penalty

    # Affinity
    affinity = min(1.0, task_affinity)

    # Composite score
    score = (
        w["quality"] * quality
        + w["cost"] * cost
        + w["recency"] * recency
        + w["affinity"] * affinity
        + w["diversity"] * 0.5  # diversity computed after ranking
    )

    return AgentScore(
        id=agent_id,
        name=agent_name,
        quality=quality,
        cost=cost,
        recency=recency,
        affinity=affinity,
        diversity=0.5,
        score=score,
    )


def route(
    prompt: str,
    agents: List[dict],
    weights: Optional[Dict[str, float]] = None,
) -> RoutingDecision:
    """Route a prompt to the best agent.

    ``agents`` is a list of dicts with keys:
        id, name, capabilities (list[str]), cost_tier (float 0..1),
        last_used_hours (float|None), task_affinity (float 0..1)
    """
    if not agents:
        return RoutingDecision(winner="", explanation="no agents available")

    required = _detect_capabilities(prompt)
    scored: List[AgentScore] = []

    for agent in agents:
        s = score_agent(
            agent_id=agent.get("id", ""),
            agent_name=agent.get("name", agent.get("id", "")),
            capabilities=agent.get("capabilities", []),
            required_caps=required,
            cost_tier=agent.get("cost_tier", 0.5),
            last_used_hours=agent.get("last_used_hours"),
            task_affinity=agent.get("task_affinity", 0.5),
            weights=weights,
        )
        scored.append(s)

    # Sort descending by score
    scored.sort(key=lambda s: s.score, reverse=True)

    # Apply diversity bonus: penalize top-2 if they're too similar
    if len(scored) >= 2:
        top = scored[0]
        second = scored[1]
        # Simple diversity: if same quality tier, give second a bump
        if abs(top.quality - second.quality) < 0.2:
            second.diversity = 0.6
            second.score += (weights or DEFAULT_WEIGHTS)["diversity"] * 0.1
            # Re-sort
            scored.sort(key=lambda s: s.score, reverse=True)

    winner = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None

    explanation = (
        f"Routed to '{winner.name}' (score {winner.score:.3f}) "
        f"for prompt type: {', '.join(required)}. "
        f"Quality={winner.quality:.2f} Cost={winner.cost:.2f} "
        f"Recency={winner.recency:.2f} Affinity={winner.affinity:.2f}"
    )

    return RoutingDecision(
        winner=winner.id,
        runner_up=runner_up.id if runner_up else None,
        ranked=scored,
        explanation=explanation,
    )
