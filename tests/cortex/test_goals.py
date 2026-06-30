"""
Tests for cortex/goals.py — Goal decomposition engine.

Tests:
  - decompose() with template matching (migrate, audit, refactor keywords)
  - decompose() with heuristic fallback (unmatched goal)
  - render_tree() output format
  - decompose_and_track() integration with GoalTracker
  - Edge cases: empty goal, very long goal
"""

import shutil
import tempfile

import pytest

from cortex.goals import (
    decompose,
    render_tree,
    decompose_and_track,
)
from cortex.goal_state import GoalTracker, GoalStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_goals_dir():
    """Provide a temporary directory for goal persistence (cleanup on teardown)."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def tracker(temp_goals_dir):
    """GoalTracker backed by a temporary directory."""
    return GoalTracker(base_dir=temp_goals_dir)


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------


class TestTemplateMatching:
    def test_migrate_database(self):
        """'migrate postgres database' matches the migrate-database template."""
        decomp = decompose("migrate postgres database to new schema")
        assert decomp.source == "template"
        assert decomp.template_id == "migrate-database"
        assert len(decomp.tasks) == 6
        assert decomp.tasks[0].agent == "research"

    def test_audit_security(self):
        """'audit security' matches the audit-security template."""
        decomp = decompose("audit security of the web application")
        assert decomp.source == "template"
        assert decomp.template_id == "audit-security"
        assert len(decomp.tasks) == 4

    def test_refactor_module(self):
        """'refactor the authentication module' matches the refactor-module template."""
        decomp = decompose("refactor the authentication module into smaller pieces")
        assert decomp.source == "template"
        assert decomp.template_id == "refactor-module"
        assert len(decomp.tasks) == 4

    def test_vulnerability_scan_triggers_audit(self):
        """'scan for vulnerabilities' triggers audit template via 'vulnerabilit' keyword."""
        decomp = decompose("scan for vulnerabilities in dependencies")
        assert decomp.source == "template"
        assert decomp.template_id == "audit-security"

    def test_split_module_triggers_refactor(self):
        """'split module' triggers refactor template."""
        decomp = decompose("split the main module into smaller files")
        assert decomp.source == "template"
        assert decomp.template_id == "refactor-module"

    def test_case_insensitive_matching(self):
        """Template matching should be case-insensitive."""
        for goal in ["MIGRATE DB", "Audit Security", "REFACTOR Module"]:
            decomp = decompose(goal)
            assert decomp.source == "template"

    def test_special_characters_in_goal(self):
        """decompose() handles goals with special characters."""
        decomp = decompose("refactor @#$% module with (parentheses) and [brackets]")
        assert decomp.source == "template"
        assert decomp.template_id == "refactor-module"


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------


class TestHeuristicFallback:
    def test_unmatched_goal_uses_heuristic(self):
        """A goal that doesn't match any template falls back to heuristic."""
        decomp = decompose("order pizza for the team lunch")
        assert decomp.source == "heuristic"
        assert decomp.template_id is None
        assert len(decomp.tasks) == 4
        assert decomp.tasks[0].id == "t1-research"

    def test_heuristic_has_cost_and_duration(self):
        """Heuristic decomposition includes non-zero cost and duration estimates."""
        decomp = decompose("plan a team building event")
        assert decomp.total_estimated_cost_usd > 0
        assert decomp.total_estimated_duration_seconds > 0

    def test_heuristic_task_structure(self):
        """Heuristic tasks follow a standard 4-step structure."""
        decomp = decompose("write documentation for the API")
        task_ids = [t.id for t in decomp.tasks]
        assert task_ids == ["t1-research", "t2-design", "t3-implement", "t4-validate"]
        # Verify dependency chain
        assert decomp.tasks[0].depends_on == []
        assert "t1-research" in decomp.tasks[1].depends_on


# ---------------------------------------------------------------------------
# render_tree output
# ---------------------------------------------------------------------------


class TestRenderTree:
    def test_render_tree_template(self):
        """Render tree output includes goal, source, totals, and task lines."""
        decomp = decompose("migrate production database")
        output = render_tree(decomp)
        assert decomp.goal in output
        assert "Source: template" in output
        assert "migrate-database" in output
        assert "$" in output
        assert "├─" in output

    def test_render_tree_heuristic(self):
        """Render tree for heuristic source has correct labels."""
        decomp = decompose("organize a hackathon")
        output = render_tree(decomp)
        assert "Source: heuristic" in output
        assert "t1-research" in output
        assert "t2-design" in output

    def test_render_tree_all_tasks_present(self):
        """Every task in the decomposition appears in the render output."""
        decomp = decompose("refactor codebase")
        output = render_tree(decomp)
        for task in decomp.tasks:
            assert task.id in output
            assert task.description in output
            assert task.agent in output

    def test_render_tree_optional_tag(self):
        """Optional tasks are tagged with [optional]."""
        decomp = decompose("migrate to postgres")
        for task in decomp.tasks:
            if task.id == "t6":
                assert task.optional
        output = render_tree(decomp)
        assert "[optional]" in output


# ---------------------------------------------------------------------------
# decompose_and_track integration
# ---------------------------------------------------------------------------


class TestDecomposeAndTrack:
    def test_basic_integration(self, tracker):
        """decompose_and_track creates a session and advances to ANALYZING."""
        session = decompose_and_track("goal-1", "build a new login feature", tracker)
        assert session.goal_id == "goal-1"
        assert session.status == GoalStatus.ANALYZING
        assert session.is_active
        assert not session.is_terminal

    def test_preset_detection_audit(self, tracker):
        """Goals with 'audit' get the 'research' preset."""
        session = decompose_and_track(
            "goal-audit", "audit application security", tracker
        )
        assert session.context.get("preset") == "research"

    def test_preset_detection_refactor(self, tracker):
        """Goals with 'refactor' get the 'feature' preset."""
        session = decompose_and_track("goal-ref", "refactor the auth module", tracker)
        assert session.context.get("preset") == "feature"

    def test_preset_detection_bugfix(self, tracker):
        """Goals with 'fix' get the 'bugfix' preset."""
        session = decompose_and_track("goal-fix", "fix the login crash", tracker)
        assert session.context.get("preset") == "bugfix"

    def test_preset_detection_strategy(self, tracker):
        """Goals with 'strategy' get the 'strategy' preset."""
        session = decompose_and_track(
            "goal-strat", "create a product strategy", tracker
        )
        assert session.context.get("preset") == "strategy"

    def test_preset_defaults_to_feature(self, tracker):
        """Goals that don't match any preset keyword default to 'feature'."""
        session = decompose_and_track("goal-gen", "write unit tests", tracker)
        assert session.context.get("preset") == "feature"

    def test_context_passed_through(self, tracker):
        """Extra context is merged into the session context."""
        session = decompose_and_track(
            "goal-ctx",
            "build a dashboard",
            tracker,
            context={"priority": "high", "team": "frontend"},
        )
        assert session.context.get("priority") == "high"
        assert session.context.get("team") == "frontend"
        assert "preset" in session.context  # merged, not replaced


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_goal_raises_valueerror(self):
        """decompose('') should raise ValueError."""
        with pytest.raises(ValueError, match="goal cannot be empty"):
            decompose("")

    def test_whitespace_goal_raises_valueerror(self):
        """decompose('   ') should raise ValueError."""
        with pytest.raises(ValueError, match="goal cannot be empty"):
            decompose("   ")

    def test_very_long_goal_raises_valueerror(self):
        """decompose() with >5000 chars should raise ValueError."""
        long_goal = "x" * 5001
        with pytest.raises(ValueError, match="goal too long"):
            decompose(long_goal)

    def test_just_under_limit_succeeds(self):
        """Goal at exactly 5000 chars should be valid."""
        goal = "migrate database " + "x" * (4980 - len("migrate database "))
        decomp = decompose(goal)
        assert decomp.source == "template" or decomp.source == "heuristic"
