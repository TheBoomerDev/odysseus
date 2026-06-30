"""
Tests for cortex/csuite.py — direct unit tests of CSuiteOrchestrator.

Covers lines missed by integration/tool tests: update_context, get_role_info
error path, prepare_prompt error path, record_response, get_history branches,
and the _render_context conditional fields.
"""

import pytest
from cortex.csuite import (
    CSuiteRole,
    CSuiteOrchestrator,
    CompanyContext,
)


# =========================================================================
# Helpers
# =========================================================================


def make_orchestrator(**context_kwargs) -> CSuiteOrchestrator:
    """Create an orchestrator with a CompanyContext built from kwargs."""
    return CSuiteOrchestrator(context=CompanyContext(**context_kwargs))


# =========================================================================
# Test update_context  (line 183)
# =========================================================================


class TestUpdateContext:
    """update_context — replaces the shared context."""

    def test_updates_context(self):
        """update_context replaces context in-place."""
        orch = make_orchestrator(name="OldCo", industry="fintech")
        new_ctx = CompanyContext(name="NewCo", industry="healthcare", team_size=50)
        orch.update_context(new_ctx)  # line 183
        assert orch.context.name == "NewCo"
        assert orch.context.industry == "healthcare"
        assert orch.context.team_size == 50

    def test_update_preserves_history(self):
        """update_context does not clear interaction history."""
        orch = make_orchestrator(name="Co")
        orch.record_response(CSuiteRole.CEO, "q1", "a1")
        orch.update_context(CompanyContext(name="Co2"))
        assert CSuiteRole.CEO in orch._history
        assert len(orch._history[CSuiteRole.CEO]) == 1


# =========================================================================
# Test get_role_info  (lines 187-190)
# =========================================================================


class TestGetRoleInfo:
    """get_role_info — returns info for known roles, raises for unknown."""

    def test_known_role_returns_info(self):
        """Valid role returns a dict with role, name, description, capabilities."""
        orch = make_orchestrator()
        info = orch.get_role_info(CSuiteRole.CTO)  # line 187
        assert info["role"] == "cto"
        assert info["name"] == "Chief Technology Officer"
        assert info["description"]
        assert len(info["capabilities"]) > 0  # line 190
        assert info["system_prompt"]  # line 190

    def test_all_roles_are_valid(self):
        """Every CSuiteRole enum value works."""
        orch = make_orchestrator()
        for role in CSuiteRole:
            info = orch.get_role_info(role)
            assert info["role"] == role.value
            assert info["name"]

    def test_unknown_role_raises(self):
        """Nonexistent role raises ValueError."""
        orch = make_orchestrator()
        with pytest.raises(ValueError, match="Unknown role"):
            orch.get_role_info("nonexistent_role")  # line 187, 188-189
        with pytest.raises(ValueError, match="Unknown role"):
            orch.get_role_info(None)  # line 188-189


# =========================================================================
# Test list_roles
# =========================================================================


class TestListRoles:
    """list_roles — returns all roles."""

    def test_list_all_roles(self):
        """Returns all 6 role entries."""
        orch = make_orchestrator()
        roles = orch.list_roles()
        assert len(roles) == 6

    def test_list_role_structure(self):
        """Each role has role, name, description, capabilities."""
        orch = make_orchestrator()
        for entry in orch.list_roles():
            assert "role" in entry
            assert "name" in entry
            assert "description" in entry
            assert "capabilities" in entry
            assert isinstance(entry["capabilities"], list)

    def test_list_expected_roles(self):
        """All 6 expected roles are present."""
        orch = make_orchestrator()
        role_ids = {r["role"] for r in orch.list_roles()}
        for expected in ("ceo", "cto", "cmo", "cfo", "qa", "rd"):
            assert expected in role_ids


# =========================================================================
# Test prepare_prompt  (line 222 — unknown-role error path)
# =========================================================================


class TestPreparePrompt:
    """prepare_prompt — builds system/user prompt."""

    def test_known_role(self):
        """Known role returns structured prompt dict."""
        orch = make_orchestrator(name="Acme")
        result = orch.prepare_prompt(CSuiteRole.CFO, "What is our runway?")
        assert result["role"] == "cfo"
        assert result["role_name"] == "Chief Financial Officer"
        assert "system_prompt" in result
        assert "user_prompt" in result
        assert "Company Context" in result["user_prompt"]
        assert "What is our runway?" in result["user_prompt"]
        assert "capabilities" in result

    def test_unknown_role_raises(self):
        """Unknown role raises ValueError (line 222)."""
        orch = make_orchestrator()
        with pytest.raises(ValueError, match="Unknown role"):
            orch.prepare_prompt("fake_role", "question")


# =========================================================================
# Test record_response  (line 248)
# =========================================================================


class TestRecordResponse:
    """record_response — stores interaction in history."""

    def test_records_response(self):
        """Response is stored under the correct role key."""
        orch = make_orchestrator()
        orch.record_response(CSuiteRole.CTO, "What stack?", "Python + Go")  # line 248
        assert len(orch._history[CSuiteRole.CTO]) == 1
        entry = orch._history[CSuiteRole.CTO][0]
        assert entry["question"] == "What stack?"
        assert entry["response"] == "Python + Go"
        assert "timestamp" in entry

    def test_records_multiple_responses(self):
        """Multiple responses accumulate in order."""
        orch = make_orchestrator()
        orch.record_response(CSuiteRole.CEO, "q1", "a1")
        orch.record_response(CSuiteRole.CEO, "q2", "a2")
        assert len(orch._history[CSuiteRole.CEO]) == 2


# =========================================================================
# Test get_history  (lines 258-260)
# =========================================================================


class TestGetHistory:
    """get_history — returns interaction history."""

    def test_no_history(self):
        """Returns empty lists when nothing has been recorded."""
        orch = make_orchestrator()
        hist = orch.get_history()  # line 260
        assert isinstance(hist, dict)
        assert len(hist) == 6  # one entry per role

    def test_all_history(self):
        """Returns history for all roles."""
        orch = make_orchestrator()
        orch.record_response(CSuiteRole.CEO, "q", "a")
        orch.record_response(CSuiteRole.CTO, "q", "a")
        hist = orch.get_history()
        assert len(hist[CSuiteRole.CEO.value]) == 1
        assert len(hist[CSuiteRole.CTO.value]) == 1

    def test_specific_role(self):
        """Filters history to one role when a role is given (line 259)."""
        orch = make_orchestrator()
        orch.record_response(CSuiteRole.CMO, "market?", "content")
        orch.record_response(CSuiteRole.CFO, "budget?", "plan")
        hist = orch.get_history(role=CSuiteRole.CFO)  # line 258-259
        assert list(hist.keys()) == ["cfo"]
        assert len(hist["cfo"]) == 1

    def test_specific_role_no_history(self):
        """Returns empty list for a role with no history."""
        orch = make_orchestrator()
        hist = orch.get_history(role=CSuiteRole.RD)
        assert hist == {"rd": []}


# =========================================================================
# Test _render_context  (lines 273, 275, 277, 279, 281, 283-285)
# =========================================================================


class TestRenderContext:
    """_render_context — conditional fields in company context."""

    def test_minimal_context(self):
        """Only mandatory fields when optional ones are empty."""
        orch = make_orchestrator(
            name="Mini",
            industry="tech",
            stage="startup",
            team_size=5,
        )
        rendered = orch._render_context()
        assert "**Company:** Mini" in rendered
        assert "**Industry:** tech" in rendered
        assert "**Stage:** startup" in rendered
        assert "**Team Size:** 5" in rendered
        # None of the optional lines should appear
        assert "**Description:**" not in rendered
        assert "**Tech Stack:**" not in rendered
        assert "**Products:**" not in rendered
        assert "**Target Market:**" not in rendered
        assert "**Revenue Model:**" not in rendered
        assert "**Goals:**" not in rendered

    def test_with_description(self):
        """Includes Description when set (line 273)."""
        orch = make_orchestrator(description="A great company")
        rendered = orch._render_context()
        assert "**Description:** A great company" in rendered

    def test_with_tech_stack(self):
        """Includes Tech Stack when set (line 275)."""
        orch = make_orchestrator(tech_stack=["Python", "Go", "React"])
        rendered = orch._render_context()
        assert "**Tech Stack:** Python, Go, React" in rendered

    def test_with_products(self):
        """Includes Products when set (line 277)."""
        orch = make_orchestrator(products=["Odysseus", "Hermes"])
        rendered = orch._render_context()
        assert "**Products:** Odysseus, Hermes" in rendered

    def test_with_target_market(self):
        """Includes Target Market when set (line 279)."""
        orch = make_orchestrator(target_market="Enterprise")
        rendered = orch._render_context()
        assert "**Target Market:** Enterprise" in rendered

    def test_with_revenue_model(self):
        """Includes Revenue Model when set (line 281)."""
        orch = make_orchestrator(revenue_model="SaaS subscriptions")
        rendered = orch._render_context()
        assert "**Revenue Model:** SaaS subscriptions" in rendered

    def test_with_goals(self):
        """Includes Goals list when set (lines 283-285)."""
        orch = make_orchestrator(goals=["Grow revenue", "Hire team", "Launch v2"])
        rendered = orch._render_context()
        assert "**Goals:**" in rendered
        assert "- Grow revenue" in rendered
        assert "- Hire team" in rendered
        assert "- Launch v2" in rendered

    def test_all_optional_fields(self):
        """All optional fields rendered together."""
        orch = make_orchestrator(
            name="FullCo",
            description="Full description",
            industry="ai",
            stage="growth",
            team_size=50,
            tech_stack=["PyTorch", "FastAPI"],
            products=["AgentHub"],
            target_market="Developers",
            revenue_model="Freemium",
            goals=["Scale to 1M users"],
        )
        rendered = orch._render_context()
        assert "**Description:** Full description" in rendered
        assert "**Tech Stack:** PyTorch, FastAPI" in rendered
        assert "**Products:** AgentHub" in rendered
        assert "**Target Market:** Developers" in rendered
        assert "**Revenue Model:** Freemium" in rendered
        assert "**Goals:**" in rendered
        assert "- Scale to 1M users" in rendered


# =========================================================================
# Combined integration — prepare_prompt with context fields (lines 187-190
# also exercised via the context rendering in prepare_prompt)
# =========================================================================


class TestPreparePromptWithContext:
    """prepare_prompt properly embeds all context fields."""

    def test_with_all_context_fields(self):
        """Full context is reflected in the user_prompt."""
        ctx = CompanyContext(
            name="MegaCorp",
            description="We build stuff",
            industry="robotics",
            stage="mature",
            team_size=500,
            tech_stack=["ROS", "C++", "Python"],
            products=["RobotArm", "DroneOS"],
            target_market="Manufacturing",
            revenue_model="Hardware + SaaS",
            goals=["Automate factories", "Global expansion"],
        )
        orch = CSuiteOrchestrator(context=ctx)
        result = orch.prepare_prompt(CSuiteRole.CTO, "Architecture review?")
        up = result["user_prompt"]
        assert "MegaCorp" in up
        assert "We build stuff" in up
        assert "robotics" in up
        assert "mature" in up
        assert "500" in up
        assert "ROS, C++" in up
        assert "RobotArm" in up
        assert "Manufacturing" in up
        assert "Hardware + SaaS" in up
        assert "Automate factories" in up
        assert "Global expansion" in up


# =========================================================================
# format_prompt_for_role — aliased to prepare_prompt usage
# The orchestrator's prepare_prompt method serves this purpose.
# =========================================================================


class TestFormatPromptForRole:
    """format_prompt_for_role — uses prepare_prompt to format a role query."""

    def test_format_via_prepare_prompt(self):
        """prepare_prompt serves as the format_prompt_for_role equivalent."""
        orch = make_orchestrator()
        result = orch.prepare_prompt(CSuiteRole.CTO, "What stack?")
        assert "role" in result
        assert "system_prompt" in result
        assert "user_prompt" in result
        assert result["role"] == "cto"
