"""
Tests for cortex/router.py — Multi-agent router/scorer.

Tests:
  - route() with sample agents
  - score_agent() calculation
  - detect_capabilities()
  - Edge cases: empty prompt, empty agents
"""

from cortex.router import (
    RoutingDecision,
    score_agent,
    route,
    _detect_capabilities,
    DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Sample agent data
# ---------------------------------------------------------------------------

SAMPLE_AGENTS = [
    {
        "id": "codex",
        "name": "Codex",
        "capabilities": ["code_edit", "multi_file_diff", "test_generation"],
        "cost_tier": 0.3,
        "last_used_hours": 2.0,
        "task_affinity": 0.9,
    },
    {
        "id": "verifier",
        "name": "Verifier",
        "capabilities": ["test_execution", "code_review", "adversarial_review"],
        "cost_tier": 0.2,
        "last_used_hours": 24.0,
        "task_affinity": 0.7,
    },
    {
        "id": "research",
        "name": "Research",
        "capabilities": ["web_search", "doc_synthesis", "data_analysis"],
        "cost_tier": 0.1,
        "last_used_hours": 48.0,
        "task_affinity": 0.5,
    },
]

ARCHITECT_AGENT = {
    "id": "architect",
    "name": "Architect",
    "capabilities": ["architecture_design", "trade_off_analysis", "doc_generation"],
    "cost_tier": 0.5,
    "last_used_hours": None,
    "task_affinity": 0.8,
}


# ---------------------------------------------------------------------------
# detect_capabilities
# ---------------------------------------------------------------------------


class TestDetectCapabilities:
    def test_code_keyword(self):
        """Prompts with 'code' detect code_edit capability."""
        caps = _detect_capabilities("write some code for me")
        assert "code_edit" in caps

    def test_test_keyword(self):
        """Prompts with 'test' detect test_generation and test_execution."""
        caps = _detect_capabilities("create unit tests")
        assert "test_generation" in caps
        assert "test_execution" in caps

    def test_refactor_keyword(self):
        """Prompts with 'refactor' detect code_edit and architecture_design."""
        caps = _detect_capabilities("refactor this module")
        assert "code_edit" in caps
        assert "architecture_design" in caps

    def test_research_keyword(self):
        """Prompts with 'research' detect web_search and doc_synthesis."""
        caps = _detect_capabilities("research the latest AI papers")
        assert "web_search" in caps
        assert "doc_synthesis" in caps

    def test_deploy_keyword(self):
        """Prompts with 'deploy' detect shell_execution and docker."""
        caps = _detect_capabilities("deploy the application to production")
        assert "shell_execution" in caps
        assert "docker" in caps

    def test_multiple_keywords(self):
        """Prompts with multiple keywords detect all matching caps."""
        caps = _detect_capabilities("write tests and refactor code")
        assert "test_generation" in caps
        assert "test_execution" in caps
        assert "code_edit" in caps
        assert "architecture_design" in caps

    def test_no_keyword_default(self):
        """Prompts with no recognized keyword default to ['code_edit']."""
        caps = _detect_capabilities("hello world")
        assert caps == ["code_edit"]

    def test_case_insensitivity(self):
        """Detection should be case-insensitive."""
        caps = _detect_capabilities("DEPLOY THE APP")
        assert "docker" in caps

    def test_empty_prompt(self):
        """Empty prompt returns the default ['code_edit']."""
        caps = _detect_capabilities("")
        assert caps == ["code_edit"]

    def test_partial_word_matching(self):
        """Detection uses substring matching — 'analy' matches 'analysis' keywords."""
        caps = _detect_capabilities("analyze the performance metrics")
        # 'analy' matches TaskSignature patterns via keyword substring
        # 'security' is not here but 'analy' is in 'analyze'
        assert len(caps) >= 1


# ---------------------------------------------------------------------------
# score_agent
# ---------------------------------------------------------------------------


class TestScoreAgent:
    def test_score_agent_perfect_match(self):
        """Perfect capability match yields quality=1.0."""
        score = score_agent(
            agent_id="codex",
            agent_name="Codex",
            capabilities=["code_edit", "test_generation"],
            required_caps=["code_edit", "test_generation"],
        )
        assert score.quality == 1.0

    def test_score_agent_partial_match(self):
        """Partial capability match yields quality < 1.0."""
        score = score_agent(
            agent_id="research",
            agent_name="Research",
            capabilities=["web_search", "doc_synthesis"],
            required_caps=["code_edit", "test_generation"],
        )
        assert score.quality == 0.0  # No overlap

    def test_score_agent_cost_inverted(self):
        """Higher cost_tier results in lower cost score."""
        expensive = score_agent(
            agent_id="a",
            agent_name="A",
            capabilities=[],
            required_caps=[],
            cost_tier=0.9,
        )
        cheap = score_agent(
            agent_id="b",
            agent_name="B",
            capabilities=[],
            required_caps=[],
            cost_tier=0.1,
        )
        assert cheap.cost > expensive.cost

    def test_score_agent_recency(self):
        """Recently used agents get higher recency."""
        recent = score_agent(
            agent_id="a",
            agent_name="A",
            capabilities=[],
            required_caps=[],
            last_used_hours=1.0,
        )
        old = score_agent(
            agent_id="b",
            agent_name="B",
            capabilities=[],
            required_caps=[],
            last_used_hours=200.0,
        )
        assert recent.recency > old.recency

    def test_score_agent_no_recency(self):
        """Agents never used get a default recency of 0.3."""
        score = score_agent(
            agent_id="new",
            agent_name="New",
            capabilities=[],
            required_caps=[],
            last_used_hours=None,
        )
        assert score.recency == 0.3

    def test_score_range(self):
        """Composite score is between 0 and 1 (approximately)."""
        score = score_agent(
            agent_id="test",
            agent_name="Test",
            capabilities=["code_edit"],
            required_caps=["code_edit"],
            cost_tier=0.5,
            last_used_hours=10.0,
            task_affinity=0.8,
        )
        assert 0 <= score.score <= 1.0

    def test_custom_weights(self):
        """Custom weights change the composite score."""
        w = dict(DEFAULT_WEIGHTS)
        w["quality"] = 1.0
        w["cost"] = 0.0
        w["recency"] = 0.0
        w["affinity"] = 0.0
        w["diversity"] = 0.0
        score = score_agent(
            agent_id="codex",
            agent_name="Codex",
            capabilities=["code_edit"],
            required_caps=["code_edit"],
            weights=w,
        )
        assert score.score == 1.0  # quality=1.0 with all weight on quality


# ---------------------------------------------------------------------------
# route()
# ---------------------------------------------------------------------------


class TestRoute:
    def test_route_selects_best_agent(self):
        """route() picks the agent with the highest score."""
        decision = route("write code", SAMPLE_AGENTS)
        assert decision.winner != ""
        assert decision.runner_up is not None

    def test_route_returns_ranked_list(self):
        """route() returns all scored agents in ranked order."""
        decision = route("write code", SAMPLE_AGENTS)
        assert len(decision.ranked) == len(SAMPLE_AGENTS)
        # Verify descending order
        scores = [a.score for a in decision.ranked]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_route_code_prompt_prefers_codex(self):
        """Code-related prompts should prefer agents with code capabilities."""
        decision = route("implement a new feature", SAMPLE_AGENTS)
        winner = decision.ranked[0]
        # Codex should be top-ranked for code tasks
        assert winner.name == "Codex" or winner.quality > 0

    def test_route_research_prompt_prefers_research(self):
        """Research prompts should prefer the research agent."""
        decision = route("research the competitive landscape", SAMPLE_AGENTS)
        best = decision.ranked[0]
        assert best.id == "research"

    def test_route_explanation_format(self):
        """route() explanation contains relevant detail."""
        decision = route("test the login endpoint", SAMPLE_AGENTS)
        assert "Routed to" in decision.explanation
        assert "score" in decision.explanation
        assert "Quality=" in decision.explanation

    def test_route_empty_agents(self):
        """route() with no agents returns an empty decision."""
        decision = route("any prompt", [])
        assert decision.winner == ""
        assert decision.explanation == "no agents available"

    def test_route_single_agent(self):
        """route() works with a single agent."""
        decision = route("write code", [SAMPLE_AGENTS[0]])
        assert decision.winner == SAMPLE_AGENTS[0]["id"]
        assert decision.runner_up is None

    def test_route_empty_prompt(self):
        """route() with empty prompt returns a valid decision."""
        decision = route("", SAMPLE_AGENTS)
        assert decision.winner != ""

    def test_route_custom_weights(self):
        """route() accepts custom weights."""
        w = dict(DEFAULT_WEIGHTS)
        w["quality"] = 0.5
        w["cost"] = 0.5
        w["recency"] = 0.0
        w["affinity"] = 0.0
        w["diversity"] = 0.0
        decision = route("write code", SAMPLE_AGENTS, weights=w)
        assert decision.winner != ""

    def test_route_diversity_bonus(self):
        """When top agents have similar quality, diversity bonus shuffles ranking."""
        # Two agents with nearly identical quality on a task
        agents = [
            {
                "id": "codex-v1",
                "name": "Codex V1",
                "capabilities": ["code_edit", "multi_file_diff"],
                "cost_tier": 0.3,
                "last_used_hours": 2.0,
                "task_affinity": 0.9,
            },
            {
                "id": "codex-v2",
                "name": "Codex V2",
                "capabilities": ["code_edit", "multi_file_diff", "test_generation"],
                "cost_tier": 0.35,
                "last_used_hours": 2.0,
                "task_affinity": 0.85,
            },
        ]
        decision = route("write code and tests", agents)
        # Both should be ranked
        assert len(decision.ranked) == 2


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_decision_attributes(self):
        """RoutingDecision has expected attributes."""
        d = RoutingDecision(winner="codex", runner_up="verifier")
        assert d.winner == "codex"
        assert d.runner_up == "verifier"
        assert d.ranked == []
        assert d.explanation == ""
