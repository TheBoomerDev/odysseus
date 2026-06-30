"""
Tests for cortex/goals.py — goal decomposition engine.

Tests template matching, heuristic fallback, and state machine integration.
"""
import shutil
import tempfile
import pytest

from cortex.goals import decompose, render_tree, decompose_and_track
from cortex.goal_state import GoalTracker


class TestDecompose:
    def test_template_migration(self):
        """Template matching for database migration goals."""
        result = decompose("migrate database from postgres to mysql")
        assert len(result.tasks) == 6
        assert result.source == "template"
        assert result.template_id == "migrate-database"
        assert result.tasks[0].description.startswith("Research")

    def test_template_audit(self):
        """Template matching for security audit goals."""
        result = decompose("audit security for our web application")
        assert len(result.tasks) == 4
        assert result.source == "template"
        assert result.template_id == "audit-security"

    def test_template_refactor(self):
        """Template matching for refactoring goals."""
        result = decompose("refactor the user module into smaller pieces")
        assert len(result.tasks) == 4
        assert result.source == "template"
        assert result.template_id == "refactor-module"

    def test_heuristic_fallback(self):
        """Unmatched goals fall back to heuristic decomposition."""
        result = decompose("write a blog post about AI")
        assert len(result.tasks) == 4
        assert result.source == "heuristic"
        assert result.template_id is None

    def test_empty_goal_raises(self):
        """Empty goal raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            decompose("")
        with pytest.raises(ValueError, match="cannot be empty"):
            decompose("   ")

    def test_long_goal_raises(self):
        """Very long goal raises ValueError."""
        with pytest.raises(ValueError, match="too long"):
            decompose("x" * 5001)

    def test_render_tree_format(self):
        """render_tree produces the expected output format."""
        result = decompose("migrate database")
        tree = render_tree(result)
        assert tree.startswith("Goal:")
        assert "Source:" in tree
        assert "Total:" in tree
        assert "├─" in tree

    def test_cost_estimation(self):
        """Total cost is sum of task costs."""
        result = decompose("refactor module")
        expected = sum(t.estimated_cost_usd for t in result.tasks)
        assert result.total_estimated_cost_usd == round(expected, 4)

    def test_dependency_chain(self):
        """Tasks have correct dependency ordering."""
        result = decompose("migrate database")
        assert result.tasks[0].depends_on == []  # First task has no deps
        assert "t1" in result.tasks[1].depends_on  # Second depends on first

    def test_optional_tasks(self):
        """Some tasks can be optional."""
        result = decompose("migrate database")
        optional_tasks = [t for t in result.tasks if t.optional]
        assert len(optional_tasks) > 0  # Documentation task is optional


class TestDecomposeAndTrack:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tracker = GoalTracker(base_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_decompose_and_track_creates_session(self):
        """decompose_and_track creates a session in ANALYZING state."""
        session = decompose_and_track("test-1", "refactor auth module", self.tracker)
        assert session.goal_id == "test-1"
        assert session.status.value == "analyzing"
        assert session.progress_pct == 10

    def test_decompose_and_track_with_goals(self):
        """Tasks are accessible from the decomposed goal."""
        session = decompose_and_track("test-2", "migrate database", self.tracker)
        result = decompose(session.goal)
        assert len(result.tasks) > 0

    def test_preset_detection(self):
        """Different goal types get correct presets."""
        s1 = decompose_and_track("s1", "audit security for our app", self.tracker)
        assert s1.context.get("preset") == "research"

        s2 = decompose_and_track("s2", "fix critical login crash", self.tracker)
        assert s2.context.get("preset") == "bugfix"

        s3 = decompose_and_track("s3", "strategy for q3 growth", self.tracker)
        assert s3.context.get("preset") == "strategy"

    def test_duplicate_goal_id(self):
        """Creating a goal with an existing ID raises."""
        decompose_and_track("dup", "first goal", self.tracker)
        with pytest.raises(ValueError, match="already exists"):
            decompose_and_track("dup", "second goal", self.tracker)
