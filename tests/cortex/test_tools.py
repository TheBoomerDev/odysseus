"""
Tests for cortex/tools.py — CORTEX tools for Odysseus agent mode.

Tests:
  - do_route_prompt
  - do_csuite_query
  - do_decompose_goal
  - do_list_csuite_roles
  - do_router_categories
"""

import json
import pytest


# ---------------------------------------------------------------------------
# do_route_prompt
# ---------------------------------------------------------------------------

class TestDoRoutePrompt:
    """Tests for do_route_prompt — routes a prompt to the best model."""

    @pytest.mark.asyncio
    async def test_basic_routing(self):
        """Returns a routing decision for a valid prompt."""
        from cortex.tools import do_route_prompt
        result = await do_route_prompt(json.dumps({"prompt": "write code for a login page"}))
        assert "error" not in result
        assert result["category"]
        assert result["recommended_model"]
        assert result["recommended_provider"]
        assert "estimated_cost_usd" in result

    @pytest.mark.asyncio
    async def test_with_priority(self):
        """Accepts priority parameter."""
        from cortex.tools import do_route_prompt
        result = await do_route_prompt(json.dumps({"prompt": "test the API", "priority": "cost"}))
        assert "error" not in result
        assert result["priority"] == "cost"

    @pytest.mark.asyncio
    async def test_missing_prompt(self):
        """Missing prompt returns an error."""
        from cortex.tools import do_route_prompt
        result = await do_route_prompt(json.dumps({}))
        assert "error" in result
        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Invalid JSON returns an error."""
        from cortex.tools import do_route_prompt
        result = await do_route_prompt("not-json")
        assert "error" in result
        assert result["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_invalid_priority(self):
        """Invalid priority returns an error."""
        from cortex.tools import do_route_prompt
        result = await do_route_prompt(json.dumps({"prompt": "hello", "priority": "invalid"}))
        assert "error" in result
        assert result["exit_code"] == 1


# ---------------------------------------------------------------------------
# do_csuite_query
# ---------------------------------------------------------------------------

class TestDoCsuiteQuery:
    """Tests for do_csuite_query — queries a C-Suite executive agent."""

    @pytest.mark.asyncio
    async def test_query_cto(self):
        """Queries the CTO role successfully."""
        from cortex.tools import do_csuite_query
        result = await do_csuite_query(json.dumps({
            "role": "cto",
            "question": "What tech stack should we use?",
        }))
        assert "error" not in result
        assert result["role"] == "cto"
        assert result["role_name"] == "Chief Technology Officer"
        assert "system_prompt" in result
        assert "user_prompt" in result

    @pytest.mark.asyncio
    async def test_query_ceo(self):
        """Queries the CEO role successfully."""
        from cortex.tools import do_csuite_query
        result = await do_csuite_query(json.dumps({
            "role": "ceo",
            "question": "What is our strategy?",
        }))
        assert "error" not in result
        assert result["role"] == "ceo"

    @pytest.mark.asyncio
    async def test_all_roles(self):
        """All 6 roles return valid results."""
        from cortex.tools import do_csuite_query
        for role in ["ceo", "cto", "cmo", "cfo", "qa", "rd"]:
            result = await do_csuite_query(json.dumps({
                "role": role,
                "question": f"Question for {role}",
            }))
            assert "error" not in result, f"Role {role} failed: {result}"
            assert result["role"] == role

    @pytest.mark.asyncio
    async def test_missing_role(self):
        """Missing role returns an error."""
        from cortex.tools import do_csuite_query
        result = await do_csuite_query(json.dumps({"question": "hello"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_question(self):
        """Missing question returns an error."""
        from cortex.tools import do_csuite_query
        result = await do_csuite_query(json.dumps({"role": "cto"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_role(self):
        """Unknown role returns an error."""
        from cortex.tools import do_csuite_query
        result = await do_csuite_query(json.dumps({"role": "cfo", "question": "test"}))
        assert "error" not in result  # 'cfo' is valid
        result = await do_csuite_query(json.dumps({"role": "nonexistent", "question": "test"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Invalid JSON returns an error."""
        from cortex.tools import do_csuite_query
        result = await do_csuite_query("bad-json")
        assert "error" in result


# ---------------------------------------------------------------------------
# do_decompose_goal
# ---------------------------------------------------------------------------

class TestDoDecomposeGoal:
    """Tests for do_decompose_goal — decomposes a goal into tasks."""

    @pytest.mark.asyncio
    async def test_basic_decompose(self):
        """Returns decomposition for a valid goal."""
        from cortex.tools import do_decompose_goal
        result = await do_decompose_goal(json.dumps({"goal": "migrate database to postgres"}))
        assert "error" not in result
        assert result["source"] == "template"
        assert result["template_id"] == "migrate-database"
        assert len(result["tasks"]) > 0

    @pytest.mark.asyncio
    async def test_heuristic_fallback(self):
        """Goals without templates use heuristic."""
        from cortex.tools import do_decompose_goal
        result = await do_decompose_goal(json.dumps({"goal": "plan a team outing"}))
        assert "error" not in result
        assert result["source"] == "heuristic"
        assert result["template_id"] is None

    @pytest.mark.asyncio
    async def test_missing_goal(self):
        """Missing goal returns an error."""
        from cortex.tools import do_decompose_goal
        result = await do_decompose_goal(json.dumps({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_goal(self):
        """Empty goal returns an error."""
        from cortex.tools import do_decompose_goal
        result = await do_decompose_goal(json.dumps({"goal": ""}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_task_structure(self):
        """Each task has the expected fields."""
        from cortex.tools import do_decompose_goal
        result = await do_decompose_goal(json.dumps({"goal": "refactor auth module"}))
        for task in result["tasks"]:
            assert "id" in task
            assert "description" in task
            assert "agent" in task
            assert "depends_on" in task
            assert "optional" in task
        assert result["total_estimated_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Invalid JSON returns an error."""
        from cortex.tools import do_decompose_goal
        result = await do_decompose_goal("bad-json")
        assert "error" in result


# ---------------------------------------------------------------------------
# do_list_csuite_roles
# ---------------------------------------------------------------------------

class TestDoListCsuiteRoles:
    """Tests for do_list_csuite_roles — lists all C-Suite roles."""

    @pytest.mark.asyncio
    async def test_lists_all_roles(self):
        """Returns all 6 C-Suite roles."""
        from cortex.tools import do_list_csuite_roles
        result = await do_list_csuite_roles("ignored")
        assert "error" not in result
        assert len(result["roles"]) == 6

    @pytest.mark.asyncio
    async def test_role_structure(self):
        """Each role has role, name, description, capabilities."""
        from cortex.tools import do_list_csuite_roles
        result = await do_list_csuite_roles("ignored")
        for role in result["roles"]:
            assert "role" in role
            assert "name" in role
            assert "description" in role
            assert "capabilities" in role
            assert isinstance(role["capabilities"], list)

    @pytest.mark.asyncio
    async def test_expected_roles_present(self):
        """CEO, CTO, CMO, CFO, QA, RD are all present."""
        from cortex.tools import do_list_csuite_roles
        result = await do_list_csuite_roles("ignored")
        role_ids = {r["role"] for r in result["roles"]}
        for expected in ["ceo", "cto", "cmo", "cfo", "qa", "rd"]:
            assert expected in role_ids


# ---------------------------------------------------------------------------
# do_router_categories
# ---------------------------------------------------------------------------

class TestDoRouterCategories:
    """Tests for do_router_categories — lists routing categories."""

    @pytest.mark.asyncio
    async def test_lists_categories(self):
        """Returns routing categories with model recommendations."""
        from cortex.tools import do_router_categories
        result = await do_router_categories("ignored")
        assert "error" not in result
        assert len(result["categories"]) > 0

    @pytest.mark.asyncio
    async def test_category_structure(self):
        """Each category has category, label, primary_model, cost_tier."""
        from cortex.tools import do_router_categories
        result = await do_router_categories("ignored")
        for cat in result["categories"]:
            assert "category" in cat
            assert "label" in cat
            assert "primary_model" in cat
            assert "cost_tier" in cat

    @pytest.mark.asyncio
    async def test_expected_categories(self):
        """Key categories like analysis, code_generation are present."""
        from cortex.tools import do_router_categories
        result = await do_router_categories("ignored")
        cats = {c["category"] for c in result["categories"]}
        for expected in ["analysis", "code_generation", "code_review", "testing", "chat"]:
            assert expected in cats
