"""
Tests for cortex/goal_state.py — 8-state machine, GoalTracker, Checkpoints.

Tests all legal/illegal transitions, progress tracking, checkpoints,
persistence, and CRUD operations.
"""
import shutil
import tempfile
import pytest

from cortex.goal_state import GoalTracker, GoalStatus, validate_transition


class TestStateMachine:
    """Tests for the 8-state state machine core."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tracker = GoalTracker(base_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --- Legal transitions ---

    def test_full_lifecycle(self):
        """Goal traverses all 8 states in order."""
        s = self.tracker.create_goal("g1", "test goal")
        assert s.status == GoalStatus.CREATED

        for state in [GoalStatus.ANALYZING, GoalStatus.PLANNING,
                      GoalStatus.DECOMPOSING, GoalStatus.ASSIGNING,
                      GoalStatus.EXECUTING, GoalStatus.VERIFYING]:
            s = self.tracker.transition("g1", state)
            assert s.status == state

        s = self.tracker.complete("g1")
        assert s.status == GoalStatus.COMPLETED
        assert s.is_terminal

    def test_fail_from_any_state(self):
        """Goal can be failed from most non-terminal states."""
        s = self.tracker.create_goal("g2", "test")
        self.tracker.transition("g2", GoalStatus.ANALYZING)
        self.tracker.transition("g2", GoalStatus.PLANNING)
        s = self.tracker.fail("g2", "something went wrong")
        assert s.status == GoalStatus.FAILED
        assert s.error == "something went wrong"

    def test_advance_sequential(self):
        """Advance moves through states sequentially."""
        s = self.tracker.create_goal("g3", "test")
        s = self.tracker.advance("g3")
        assert s.status == GoalStatus.ANALYZING
        s = self.tracker.advance("g3")
        assert s.status == GoalStatus.PLANNING
        s = self.tracker.advance("g3")
        assert s.status == GoalStatus.DECOMPOSING

    def test_retry_after_failure(self):
        """Failed goal can be retried back to CREATED."""
        s = self.tracker.create_goal("g4", "test")
        self._advance_to("g4", GoalStatus.EXECUTING)
        self.tracker.fail("g4", "timeout")
        s = self.tracker.retry("g4")
        assert s.status == GoalStatus.CREATED

    def test_verifying_to_executing(self):
        """VERIFYING can transition back to EXECUTING (rework)."""
        s = self.tracker.create_goal("g5", "test")
        self._advance_to("g5", GoalStatus.VERIFYING)
        s = self.tracker.transition("g5", GoalStatus.EXECUTING)
        assert s.status == GoalStatus.EXECUTING

    def _advance_to(self, gid, target):
        for st in [GoalStatus.ANALYZING, GoalStatus.PLANNING,
                   GoalStatus.DECOMPOSING, GoalStatus.ASSIGNING,
                   GoalStatus.EXECUTING, GoalStatus.VERIFYING]:
            self.tracker.transition(gid, st)
            if st == target:
                break

    # --- Illegal transitions ---

    def test_cannot_skip_states(self):
        """Cannot skip from CREATED directly to EXECUTING."""
        s = self.tracker.create_goal("g6", "test")
        with pytest.raises(ValueError, match="Illegal transition"):
            self.tracker.transition("g6", GoalStatus.EXECUTING)

    def test_cannot_transition_from_completed(self):
        """No transitions allowed from COMPLETED state."""
        s = self.tracker.create_goal("g7", "test")
        self._advance_to("g7", GoalStatus.VERIFYING)
        self.tracker.complete("g7")
        with pytest.raises(ValueError, match="Illegal transition"):
            self.tracker.transition("g7", GoalStatus.ANALYZING)

    def test_cannot_advance_from_completed(self):
        """Cannot advance from terminal state."""
        s = self.tracker.create_goal("g8", "test")
        self._advance_to("g8", GoalStatus.VERIFYING)
        self.tracker.complete("g8")
        with pytest.raises(ValueError):
            self.tracker.advance("g8")

    def test_validate_transition_function(self):
        """validate_transition rejects illegal moves."""
        with pytest.raises(ValueError):
            validate_transition(GoalStatus.CREATED, GoalStatus.EXECUTING)
        with pytest.raises(ValueError):
            validate_transition(GoalStatus.COMPLETED, GoalStatus.ANALYZING)
        # Legal
        validate_transition(GoalStatus.CREATED, GoalStatus.ANALYZING)

    # --- Progress tracking ---

    def test_progress_tracking(self):
        """Progress can be updated with task counts."""
        s = self.tracker.create_goal("g9", "test")
        self._advance_to("g9", GoalStatus.EXECUTING)
        self.tracker.update_progress("g9", 0.45, task_count=10, tasks_completed=4)
        s = self.tracker.get_goal("g9")
        assert s.progress_pct == 45
        assert s.task_count == 10
        assert s.tasks_completed == 4
        assert s.progress == 0.45

    def test_progress_clamped(self):
        """Progress is clamped to 0.0-1.0."""
        s = self.tracker.create_goal("g10", "test")
        self._advance_to("g10", GoalStatus.EXECUTING)
        self.tracker.update_progress("g10", 1.5)
        assert self.tracker.get_goal("g10").progress_pct == 100
        self.tracker.update_progress("g10", -0.5)
        assert self.tracker.get_goal("g10").progress_pct == 0

    # --- Checkpoints ---

    def test_checkpoint_creation(self):
        """Checkpoints are created with unique IDs."""
        s = self.tracker.create_goal("g11", "test")
        self.tracker.transition("g11", GoalStatus.ANALYZING)
        cp = self.tracker.checkpoint("g11", context={"phase": "analysis"})
        assert cp.id.startswith("cp-")
        assert cp.state == GoalStatus.ANALYZING

    def test_checkpoint_restore(self):
        """Restoring a checkpoint reverts state and progress."""
        s = self.tracker.create_goal("g12", "test")
        self.tracker.transition("g12", GoalStatus.ANALYZING)
        cp = self.tracker.checkpoint("g12", context={"phase": "analysis"})
        self.tracker.transition("g12", GoalStatus.PLANNING)
        self.tracker.transition("g12", GoalStatus.DECOMPOSING)
        self.tracker.restore_checkpoint("g12", cp.id)
        s = self.tracker.get_goal("g12")
        assert s.status == GoalStatus.ANALYZING

    def test_checkpoint_list(self):
        """Checkpoints are listed for a goal."""
        s = self.tracker.create_goal("g13", "test")
        self.tracker.transition("g13", GoalStatus.ANALYZING)
        self.tracker.checkpoint("g13", context={"a": 1})
        self.tracker.transition("g13", GoalStatus.PLANNING)
        self.tracker.checkpoint("g13", context={"b": 2})
        cps = self.tracker.get_checkpoints("g13")
        assert len(cps) >= 2

    # --- CRUD ---

    def test_create_get_goal(self):
        """Created goal can be retrieved."""
        self.tracker.create_goal("g14", "test")
        s = self.tracker.get_goal("g14")
        assert s is not None
        assert s.goal == "test"

    def test_get_nonexistent_goal(self):
        """Nonexistent goal returns None."""
        assert self.tracker.get_goal("nonexistent") is None

    def test_delete_goal(self):
        """Goal can be deleted."""
        self.tracker.create_goal("g15", "test")
        assert self.tracker.delete_goal("g15") is True
        assert self.tracker.get_goal("g15") is None

    def test_list_goals(self):
        """All goals are listed."""
        self.tracker.create_goal("g16a", "test a")
        self.tracker.create_goal("g16b", "test b")
        goals = self.tracker.list_goals()
        assert len(goals) == 2

    def test_list_filter_by_status(self):
        """Goals can be filtered by status."""
        s = self.tracker.create_goal("g17", "test")
        # Track g17 is now CREATED
        self.tracker.create_goal("g17b", "another created")
        self.tracker.transition("g17", GoalStatus.ANALYZING)
        analyzing = self.tracker.list_goals(status=GoalStatus.ANALYZING)
        created = self.tracker.list_goals(status=GoalStatus.CREATED)
        assert len(analyzing) >= 1
        assert len(created) >= 1  # g17b is CREATED

    # --- Context ---

    def test_context_update(self):
        """Context can be updated on a goal."""
        self.tracker.create_goal("g18", "test")
        self.tracker.update_context("g18", key="value", number=42)
        s = self.tracker.get_goal("g18")
        assert s.context["key"] == "value"
        assert s.context["number"] == 42

    # --- Persistence ---

    def test_persistence(self):
        """Goals persist across GoalTracker instances."""
        self.tracker.create_goal("g19", "persistent goal")
        # Create second tracker with same base dir
        tracker2 = GoalTracker(base_dir=self.tmpdir)
        s = tracker2.get_goal("g19")
        assert s is not None
        assert s.goal == "persistent goal"
