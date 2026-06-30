"""Tests for goal tracking tools in cortex/tools.py.

Tests: do_create_goal, do_goal_status, do_goal_transition,
       do_goal_advance, do_goal_checkpoint, do_goal_update_progress,
       do_list_goals.
"""

import json
import pytest


_COUNTER = 0


def _unique_id(prefix="g"):
    global _COUNTER
    _COUNTER += 1
    return f"{prefix}-{_COUNTER}"


class TestDoCreateGoal:
    """Tests for do_create_goal — creates a tracked goal."""

    @pytest.mark.asyncio
    async def test_create_basic_goal(self):
        """Creates a basic goal successfully."""
        from cortex.tools import do_create_goal

        gid = _unique_id("basic")
        result = await do_create_goal(
            json.dumps({"goal_id": gid, "goal": "test migration"})
        )
        assert "error" not in result, result
        assert result["goal_id"] == gid
        assert result["status"] == "analyzing"
        assert result["progress_pct"] >= 0

    @pytest.mark.asyncio
    async def test_create_duplicate_returns_error(self):
        """Duplicate goal_id returns an error."""
        from cortex.tools import do_create_goal

        gid = _unique_id("dup")
        r1 = await do_create_goal(
            json.dumps({"goal_id": gid, "goal": "first goal"})
        )
        assert "error" not in r1
        result = await do_create_goal(
            json.dumps({"goal_id": gid, "goal": "duplicate"})
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_missing_fields(self):
        """Missing goal_id or goal returns error."""
        from cortex.tools import do_create_goal

        result = await do_create_goal(json.dumps({}))
        assert "error" in result
        result = await do_create_goal(json.dumps({"goal_id": "x"}))
        assert "error" in result
        result = await do_create_goal(json.dumps({"goal": "x"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_invalid_json(self):
        """Invalid JSON returns error."""
        from cortex.tools import do_create_goal

        result = await do_create_goal("bad-json")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_with_context(self):
        """Creates a goal with context."""
        from cortex.tools import do_create_goal

        gid = _unique_id("ctx")
        result = await do_create_goal(
            json.dumps({
                "goal_id": gid,
                "goal": "goal with context",
                "context": {"key": "value"},
            })
        )
        assert "error" not in result
        assert result["goal_id"] == gid


class TestDoGoalStatus:
    """Tests for do_goal_status — checks goal status."""

    @pytest.mark.asyncio
    async def test_get_status_existing_goal(self):
        """Returns status for an existing goal."""
        from cortex.tools import do_create_goal, do_goal_status

        gid = _unique_id("st")
        await do_create_goal(json.dumps({"goal_id": gid, "goal": "check status"}))
        result = await do_goal_status(json.dumps({"goal_id": gid}))
        assert "error" not in result or result.get("error") is None
        assert result["goal_id"] == gid
        assert result["status"] == "analyzing"
        assert "progress_pct" in result
        assert "is_terminal" in result
        assert "is_active" in result

    @pytest.mark.asyncio
    async def test_get_status_nonexistent(self):
        """Returns error for nonexistent goal."""
        from cortex.tools import do_goal_status

        result = await do_goal_status(
            json.dumps({"goal_id": _unique_id("nonexistent")})
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_status_missing_id(self):
        """Missing goal_id returns error."""
        from cortex.tools import do_goal_status

        result = await do_goal_status(json.dumps({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_status_invalid_json(self):
        """Invalid JSON returns error."""
        from cortex.tools import do_goal_status

        result = await do_goal_status("bad-json")
        assert "error" in result


class TestDoGoalTransition:
    """Tests for do_goal_transition — transitions goal states."""

    @pytest.mark.asyncio
    async def test_transition_valid(self):
        """Transitions a goal to a valid state."""
        from cortex.tools import do_create_goal, do_goal_transition

        gid = _unique_id("tr")
        await do_create_goal(json.dumps({"goal_id": gid, "goal": "transition test"}))
        result = await do_goal_transition(
            json.dumps({"goal_id": gid, "status": "planning"})
        )
        assert "error" not in result
        assert result["status"] == "planning"

    @pytest.mark.asyncio
    async def test_transition_invalid_state(self):
        """Transition to invalid state returns error."""
        from cortex.tools import do_create_goal, do_goal_transition

        gid = _unique_id("tr")
        await do_create_goal(json.dumps({"goal_id": gid, "goal": "invalid trans"}))
        result = await do_goal_transition(
            json.dumps({"goal_id": gid, "status": "nonexistent"})
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_transition_missing_fields(self):
        """Missing fields returns error."""
        from cortex.tools import do_goal_transition

        result = await do_goal_transition(json.dumps({}))
        assert "error" in result
        result = await do_goal_transition(json.dumps({"goal_id": "x"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_transition_to_failed(self):
        """Transition to failed with error message."""
        from cortex.tools import do_create_goal, do_goal_transition

        gid = _unique_id("tr")
        await do_create_goal(json.dumps({"goal_id": gid, "goal": "fail goal"}))
        result = await do_goal_transition(
            json.dumps({
                "goal_id": gid,
                "status": "failed",
                "error": "something went wrong",
            })
        )
        assert "error" not in result
        assert result["status"] == "failed"


class TestDoGoalAdvance:
    """Tests for do_goal_advance — advances to next state."""

    @pytest.mark.asyncio
    async def test_advance_goal(self):
        """Advances goal to next logical state."""
        from cortex.tools import do_create_goal, do_goal_advance

        gid = _unique_id("adv")
        await do_create_goal(json.dumps({"goal_id": gid, "goal": "advance test"}))
        result = await do_goal_advance(json.dumps({"goal_id": gid}))
        assert "error" not in result
        assert result["status"] in ("planning", "decomposing", "assigning")

    @pytest.mark.asyncio
    async def test_advance_nonexistent(self):
        """Advance nonexistent goal returns error."""
        from cortex.tools import do_goal_advance

        result = await do_goal_advance(
            json.dumps({"goal_id": _unique_id("nonexistent")})
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_advance_missing_id(self):
        """Missing goal_id returns error."""
        from cortex.tools import do_goal_advance

        result = await do_goal_advance(json.dumps({}))
        assert "error" in result


class TestDoGoalCheckpoint:
    """Tests for do_goal_checkpoint — creates checkpoints."""

    @pytest.mark.asyncio
    async def test_create_checkpoint(self):
        """Creates a checkpoint for a goal."""
        from cortex.tools import do_create_goal, do_goal_checkpoint

        gid = _unique_id("cp")
        await do_create_goal(json.dumps({"goal_id": gid, "goal": "checkpoint test"}))
        result = await do_goal_checkpoint(json.dumps({"goal_id": gid}))
        assert "error" not in result
        assert "checkpoint_id" in result
        assert "state" in result
        assert "progress" in result

    @pytest.mark.asyncio
    async def test_create_checkpoint_with_context(self):
        """Creates checkpoint with context and outputs."""
        from cortex.tools import do_create_goal, do_goal_checkpoint

        gid = _unique_id("cp")
        await do_create_goal(json.dumps({"goal_id": gid, "goal": "cp with data"}))
        result = await do_goal_checkpoint(
            json.dumps({
                "goal_id": gid,
                "context": {"step": 1},
                "outputs": {"result": "ok"},
            })
        )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_checkpoint_nonexistent(self):
        """Checkpoint nonexistent goal returns error."""
        from cortex.tools import do_goal_checkpoint

        result = await do_goal_checkpoint(
            json.dumps({"goal_id": _unique_id("no-goal")})
        )
        assert "error" in result


class TestDoGoalUpdateProgress:
    """Tests for do_goal_update_progress — updates progress."""

    @pytest.mark.asyncio
    async def test_update_progress(self):
        """Updates goal progress."""
        from cortex.tools import do_create_goal, do_goal_update_progress

        gid = _unique_id("prog")
        await do_create_goal(json.dumps({"goal_id": gid, "goal": "progress test"}))
        result = await do_goal_update_progress(
            json.dumps({
                "goal_id": gid,
                "progress": 0.5,
                "task_count": 4,
                "tasks_completed": 2,
            })
        )
        assert "error" not in result
        assert result["progress_pct"] == 50
        assert result["task_count"] == 4
        assert result["tasks_completed"] == 2

    @pytest.mark.asyncio
    async def test_update_progress_missing_id(self):
        """Missing goal_id returns error."""
        from cortex.tools import do_goal_update_progress

        result = await do_goal_update_progress(
            json.dumps({"progress": 0.5})
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_invalid_json(self):
        """Invalid JSON returns error."""
        from cortex.tools import do_goal_update_progress

        result = await do_goal_update_progress("bad-json")
        assert "error" in result


class TestDoListGoals:
    """Tests for do_list_goals — lists tracked goals."""

    @pytest.mark.asyncio
    async def test_list_all_goals(self):
        """Lists all tracked goals."""
        from cortex.tools import do_create_goal, do_list_goals

        gid1 = _unique_id("list")
        gid2 = _unique_id("list")
        await do_create_goal(json.dumps({"goal_id": gid1, "goal": "goal one"}))
        await do_create_goal(json.dumps({"goal_id": gid2, "goal": "goal two"}))
        result = await do_list_goals("ignored")
        assert "error" not in result
        assert result["count"] >= 2

    @pytest.mark.asyncio
    async def test_list_filtered(self):
        """Lists goals filtered by status."""
        from cortex.tools import do_list_goals

        result = await do_list_goals(json.dumps({"status": "analyzing"}))
        assert "error" not in result
        for g in result["goals"]:
            assert g["status"] == "analyzing"

    @pytest.mark.asyncio
    async def test_list_invalid_status(self):
        """Invalid status filter is ignored gracefully."""
        from cortex.tools import do_list_goals

        result = await do_list_goals(json.dumps({"status": "bogus"}))
        assert "error" not in result
        assert result["count"] >= 0
