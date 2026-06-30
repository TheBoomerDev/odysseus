"""Tests for cortex/heartbeat.py — Periodic health checks and goal monitoring.

Tests cover:
  - HeartbeatTickResult and HeartbeatStatus dataclasses
  - HeartbeatManager constructor / defaults
  - start / stop lifecycle (asyncio task management)
  - _tick() with various health-check combinations
  - tick() public wrapper
  - get_status() with and without check_goal_fn
  - _check_stale_goals() with various goal states
  - Error paths: DB check exception, stale goal exception, callback exception
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from cortex.heartbeat import (
    HeartbeatManager,
    HeartbeatTickResult,
    HeartbeatStatus,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestHeartbeatTickResult:
    def test_defaults(self):
        r = HeartbeatTickResult()
        assert r.db_connected is False
        assert r.redis_connected is False
        assert r.stale_goals_found == 0
        assert r.goals_marked_failed == 0
        assert r.errors == []
        assert r.duration_ms == 0.0

    def test_custom_values(self):
        r = HeartbeatTickResult(
            db_connected=True,
            redis_connected=True,
            stale_goals_found=3,
            goals_marked_failed=2,
            errors=["err1"],
            duration_ms=12.5,
        )
        assert r.db_connected is True
        assert r.redis_connected is True
        assert r.stale_goals_found == 3
        assert r.goals_marked_failed == 2
        assert r.errors == ["err1"]
        assert r.duration_ms == 12.5


class TestHeartbeatStatus:
    def test_defaults(self):
        s = HeartbeatStatus()
        assert s.last_tick is None
        assert s.healthy is True
        assert s.uptime_seconds == 0.0
        assert s.active_goals == 0
        assert s.stale_goals_count == 0
        assert s.goals_marked_failed == 0
        assert s.db_connected is False
        assert s.redis_connected is False
        assert s.total_ticks == 0
        assert s.failed_ticks == 0

    def test_custom_values(self):
        s = HeartbeatStatus(
            last_tick="2025-01-01T00:00:00",
            healthy=False,
            uptime_seconds=3600.0,
            active_goals=5,
            stale_goals_count=2,
            goals_marked_failed=1,
            db_connected=True,
            redis_connected=False,
            total_ticks=10,
            failed_ticks=3,
        )
        assert s.last_tick == "2025-01-01T00:00:00"
        assert s.healthy is False
        assert s.uptime_seconds == 3600.0
        assert s.active_goals == 5
        assert s.stale_goals_count == 2
        assert s.goals_marked_failed == 1
        assert s.db_connected is True
        assert s.total_ticks == 10
        assert s.failed_ticks == 3


# ---------------------------------------------------------------------------
# HeartbeatManager — constructor
# ---------------------------------------------------------------------------


class TestHeartbeatManagerInit:
    def test_default_constructor(self):
        mgr = HeartbeatManager()
        assert mgr.tick_interval_s == 60
        assert mgr.stale_threshold_s == 1800
        assert mgr._db_health_check is None
        assert mgr._redis_health_check is None
        assert mgr._on_goal_failed is None
        assert mgr._check_goal_fn is None
        assert mgr._mark_goal_fn is None
        assert mgr._task is None
        assert mgr._healthy is True
        assert mgr._total_ticks == 0
        assert mgr._failed_ticks == 0
        assert mgr._goals_marked_failed_total == 0

    def test_custom_constructor(self):
        db_fn = MagicMock(return_value=True)
        redis_fn = MagicMock(return_value=True)
        on_fail = MagicMock()
        check_fn = MagicMock(return_value=[])
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            tick_interval_s=10,
            stale_threshold_s=300,
            db_health_check=db_fn,
            redis_health_check=redis_fn,
            on_goal_failed=on_fail,
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
        )
        assert mgr.tick_interval_s == 10
        assert mgr.stale_threshold_s == 300
        assert mgr._db_health_check is db_fn
        assert mgr._redis_health_check is redis_fn
        assert mgr._on_goal_failed is on_fail
        assert mgr._check_goal_fn is check_fn
        assert mgr._mark_goal_fn is mark_fn


# ---------------------------------------------------------------------------
# HeartbeatManager — start / stop lifecycle
# ---------------------------------------------------------------------------


class TestHeartbeatManagerLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_task_and_runs_tick(self):
        """start() runs an immediate tick and creates a background loop task."""
        db_fn = MagicMock(return_value=True)
        mgr = HeartbeatManager(tick_interval_s=3600, db_health_check=db_fn)

        assert mgr._task is None
        assert mgr._total_ticks == 0

        await mgr.start()

        # An immediate tick should have run
        assert mgr._total_ticks == 1
        assert mgr._task is not None
        assert mgr._task._coro is not None  # it's a running Task

        # Cleanup
        await mgr.stop()
        assert mgr._task is None

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """start() logs a warning and returns early if already running."""
        mgr = HeartbeatManager(tick_interval_s=3600)
        await mgr.start()
        task = mgr._task

        with patch("cortex.heartbeat.logger") as mock_log:
            await mgr.start()
            mock_log.warning.assert_called_once_with(
                "HeartbeatManager already running"
            )
        # The same task reference is preserved
        assert mgr._task is task

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """stop() cancels the background task and sets _task to None."""
        mgr = HeartbeatManager(tick_interval_s=3600)
        await mgr.start()
        assert mgr._task is not None

        await mgr.stop()
        assert mgr._task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """stop() is a no-op when not running (no crash)."""
        mgr = HeartbeatManager()
        # Should not raise
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_loop_runs_ticks(self):
        """_loop() runs ticks on interval until cancelled."""
        db_fn = MagicMock(return_value=True)
        mgr = HeartbeatManager(tick_interval_s=0.01, db_health_check=db_fn)

        await mgr.start()
        # Let a few ticks happen
        await asyncio.sleep(0.05)
        await mgr.stop()

        # Should have run more than just the initial tick
        assert mgr._total_ticks > 1

    @pytest.mark.asyncio
    async def test_loop_cancelled_error_handled(self):
        """_loop() catches CancelledError gracefully."""
        mgr = HeartbeatManager(tick_interval_s=3600)
        await mgr.start()
        # Simulate the loop cancellation via stop()
        await mgr.stop()
        # No crash, clean state
        assert mgr._task is None


# ---------------------------------------------------------------------------
# HeartbeatManager — _tick() core logic
# ---------------------------------------------------------------------------


class TestHeartbeatManagerTick:
    @pytest.mark.asyncio
    async def test_tick_with_no_checks(self):
        """With no health check fns, db_connected is True (assumed OK)."""
        mgr = HeartbeatManager()

        # Patch time.time to ensure measurable duration
        with patch("cortex.heartbeat.time") as mock_time:
            mock_time.time.side_effect = [100.0, 100.05]  # tick_start, end
            result = await mgr._tick()

        assert result.db_connected is True
        assert result.redis_connected is False  # no redis check configured
        assert result.stale_goals_found == 0
        assert result.goals_marked_failed == 0
        assert result.errors == []
        assert result.duration_ms == 50.0
        assert mgr._total_ticks == 1
        assert mgr._healthy is True

    @pytest.mark.asyncio
    async def test_tick_db_healthy(self):
        db_fn = MagicMock(return_value=True)
        mgr = HeartbeatManager(db_health_check=db_fn)
        result = await mgr._tick()

        assert result.db_connected is True
        db_fn.assert_called_once()
        assert mgr._healthy is True

    @pytest.mark.asyncio
    async def test_tick_db_unhealthy(self):
        db_fn = MagicMock(return_value=False)
        mgr = HeartbeatManager(db_health_check=db_fn)
        result = await mgr._tick()

        assert result.db_connected is False
        assert mgr._healthy is False
        assert mgr._failed_ticks == 1

    @pytest.mark.asyncio
    async def test_tick_db_check_raises(self):
        db_fn = MagicMock(side_effect=RuntimeError("DB down"))
        mgr = HeartbeatManager(db_health_check=db_fn)
        result = await mgr._tick()

        assert result.db_connected is False
        assert len(result.errors) == 1
        assert "Database check failed: DB down" in result.errors[0]
        assert mgr._healthy is False

    @pytest.mark.asyncio
    async def test_tick_redis_healthy(self):
        redis_fn = MagicMock(return_value=True)
        mgr = HeartbeatManager(redis_health_check=redis_fn)
        result = await mgr._tick()

        assert result.redis_connected is True
        redis_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_redis_unhealthy(self):
        redis_fn = MagicMock(return_value=False)
        mgr = HeartbeatManager(redis_health_check=redis_fn)
        result = await mgr._tick()

        assert result.redis_connected is False

    @pytest.mark.asyncio
    async def test_tick_redis_check_raises(self):
        redis_fn = MagicMock(side_effect=ConnectionError("Redis timeout"))
        mgr = HeartbeatManager(redis_health_check=redis_fn)
        result = await mgr._tick()

        assert result.redis_connected is False
        # Exception in redis check is silently caught (no error logged)
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_tick_with_stale_goals(self):
        """Stale goal detection runs when db is connected and fns are provided."""
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "goal-1",
                    "goal": "test goal",
                    "status": "active",
                    "updated_at": "2020-01-01T00:00:00",  # definitely stale
                }
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            db_health_check=MagicMock(return_value=True),
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
        )
        result = await mgr._tick()

        assert result.stale_goals_found == 1
        assert result.goals_marked_failed == 1
        mark_fn.assert_called_once_with("goal-1", "failed")
        assert mgr._goals_marked_failed_total == 1

    @pytest.mark.asyncio
    async def test_tick_with_no_stale_goals(self):
        """Recent goals are not marked as stale."""
        future_ts = "2099-01-01T00:00:00"
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "goal-2",
                    "goal": "recent goal",
                    "status": "active",
                    "updated_at": future_ts,
                }
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            db_health_check=MagicMock(return_value=True),
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
        )
        result = await mgr._tick()

        assert result.stale_goals_found == 0
        assert result.goals_marked_failed == 0
        mark_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_stale_check_skipped_if_db_disconnected(self):
        """Stale goal check is skipped when db is not connected."""
        check_fn = MagicMock(return_value=[])
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            db_health_check=MagicMock(return_value=False),
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
        )
        result = await mgr._tick()

        # Stale check should NOT have been called
        check_fn.assert_not_called()
        mark_fn.assert_not_called()
        assert result.stale_goals_found == 0
        assert result.goals_marked_failed == 0

    @pytest.mark.asyncio
    async def test_tick_stale_check_exception(self):
        """Exception in stale goal check is caught and recorded."""
        check_fn = MagicMock(side_effect=ValueError("corrupt data"))
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            db_health_check=MagicMock(return_value=True),
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
        )
        result = await mgr._tick()

        assert len(result.errors) == 1
        assert "Stale goal check failed: corrupt data" in result.errors[0]

    @pytest.mark.asyncio
    async def test_tick_skips_stale_if_no_check_fn(self):
        """No stale goal detection without _check_goal_fn."""
        mgr = HeartbeatManager(db_health_check=MagicMock(return_value=True))
        result = await mgr._tick()

        assert result.stale_goals_found == 0
        assert result.goals_marked_failed == 0

    @pytest.mark.asyncio
    async def test_tick_skips_stale_if_no_mark_fn(self):
        """No stale goal detection without _mark_goal_fn."""
        mgr = HeartbeatManager(
            db_health_check=MagicMock(return_value=True),
            check_goal_fn=MagicMock(return_value=[]),
        )
        result = await mgr._tick()

        assert result.stale_goals_found == 0
        assert result.goals_marked_failed == 0

    @pytest.mark.asyncio
    async def test_tick_critical_exception(self):
        """A top-level exception in _tick is caught and recorded."""
        mgr = HeartbeatManager()

        # Make datetime.now() raise to trigger the outer try/except
        with patch("cortex.heartbeat.datetime") as mock_dt:
            mock_dt.now.side_effect = RuntimeError("time error")
            mock_dt.fromisoformat = datetime.fromisoformat  # keep real
            mock_dt.timezone = timezone  # keep real

            with patch("cortex.heartbeat.time.time", return_value=100.0):
                result = await mgr._tick()

        assert mgr._healthy is False
        assert mgr._failed_ticks == 1
        assert len(result.errors) == 1
        assert "Critical error: time error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_tick_duration_calculation_failure(self):
        """If duration_ms calculation fails, it defaults to 0.0."""
        mgr = HeartbeatManager()

        # Make time.time raise on the second call (for duration calc)
        real_time = __import__("time").time
        call_count = 0

        def broken_on_duration():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("duration fail")
            return real_time()

        with patch("cortex.heartbeat.time.time", side_effect=broken_on_duration):
            result = await mgr._tick()

        assert result.duration_ms == 0.0
        assert result.db_connected is True  # tick itself succeeded

    @pytest.mark.asyncio
    async def test_tick_errors_logged(self):
        """Errors are logged at warning level."""
        db_fn = MagicMock(side_effect=RuntimeError("DB error"))
        mgr = HeartbeatManager(db_health_check=db_fn)

        with patch("cortex.heartbeat.logger") as mock_log:
            result = await mgr._tick()

        # Verify the DB health check failure was logged
        assert mock_log.warning.call_count >= 1
        # The first warning call should be the DB health check failure
        first_call = mock_log.warning.call_args_list[0]
        assert first_call[0][0] == "Heartbeat DB health check failed: %s"
        # The second arg is the actual RuntimeError object (str(e) = "DB error")
        assert isinstance(first_call[0][1], RuntimeError)
        assert str(first_call[0][1]) == "DB error"

        # Also verify the summary warning is logged
        assert mock_log.warning.call_count >= 2
        summary_call = mock_log.warning.call_args_list[1]
        assert "completed with" in summary_call[0][0]

    @pytest.mark.asyncio
    async def test_tick_no_errors_no_warning_summary(self):
        """When no errors, no summary warning is logged."""
        mgr = HeartbeatManager(db_health_check=MagicMock(return_value=True))
        with patch("cortex.heartbeat.logger") as mock_log:
            await mgr._tick()

        # The 'completed with' message should NOT appear
        for call_args in mock_log.warning.call_args_list:
            msg = call_args[0][0]
            assert "completed with" not in msg


# ---------------------------------------------------------------------------
# HeartbeatManager — public tick() method
# ---------------------------------------------------------------------------


class TestHeartbeatManagerPublicTick:
    @pytest.mark.asyncio
    async def test_tick_returns_result(self):
        mgr = HeartbeatManager(db_health_check=MagicMock(return_value=True))
        result = await mgr.tick()
        assert isinstance(result, HeartbeatTickResult)
        assert result.db_connected is True
        assert mgr._total_ticks == 1


# ---------------------------------------------------------------------------
# HeartbeatManager — get_status()
# ---------------------------------------------------------------------------


class TestHeartbeatManagerGetStatus:
    @pytest.mark.asyncio
    async def test_get_status_no_checks(self):
        mgr = HeartbeatManager()
        status = await mgr.get_status()

        assert isinstance(status, HeartbeatStatus)
        assert status.last_tick is None
        assert status.healthy is True
        assert status.uptime_seconds == 0.0
        assert status.active_goals == 0
        assert status.stale_goals_count == 0
        assert status.goals_marked_failed == 0
        assert status.total_ticks == 0
        assert status.failed_ticks == 0

    @pytest.mark.asyncio
    async def test_get_status_after_tick(self):
        mgr = HeartbeatManager(db_health_check=MagicMock(return_value=True))
        mgr._start_time = 1000.0  # non-zero so uptime is computed
        await mgr._tick()
        status = await mgr.get_status()

        assert status.total_ticks == 1
        assert status.failed_ticks == 0
        assert status.healthy is True
        assert status.last_tick is not None
        assert status.uptime_seconds > 0

    @pytest.mark.asyncio
    async def test_get_status_with_check_goal_fn(self):
        """get_status counts active and stale goals via check_goal_fn."""
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "active goal",
                    "status": "active",
                    "updated_at": "2099-01-01T00:00:00",
                },
                {
                    "id": "g2",
                    "goal": "completed goal",
                    "status": "completed",
                    "updated_at": "2020-01-01T00:00:00",
                },
                {
                    "id": "g3",
                    "goal": "failed goal",
                    "status": "failed",
                    "updated_at": "2020-01-01T00:00:00",
                },
            ]
        )
        mgr = HeartbeatManager(check_goal_fn=check_fn)
        status = await mgr.get_status()

        # Only g1 is active (completed/failed excluded)
        assert status.active_goals == 1
        # g1's updated_at is in the future, so not stale
        assert status.stale_goals_count == 0

    @pytest.mark.asyncio
    async def test_get_status_with_stale_goal(self):
        """A goal older than stale_threshold_s is counted as stale."""
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "stale goal",
                    "status": "active",
                    "updated_at": "2020-01-01T00:00:00",
                }
            ]
        )
        mgr = HeartbeatManager(check_goal_fn=check_fn, stale_threshold_s=1)
        status = await mgr.get_status()

        assert status.active_goals == 1
        assert status.stale_goals_count == 1

    @pytest.mark.asyncio
    async def test_get_status_check_goal_fn_exception(self):
        """Exception in check_goal_fn during get_status is silently caught."""
        check_fn = MagicMock(side_effect=ValueError("oops"))
        mgr = HeartbeatManager(check_goal_fn=check_fn)
        # Should not raise
        status = await mgr.get_status()
        assert status.active_goals == 0
        assert status.stale_goals_count == 0

    @pytest.mark.asyncio
    async def test_get_status_with_no_check_fn(self):
        """Without check_goal_fn, active/stale counts are 0."""
        mgr = HeartbeatManager()
        status = await mgr.get_status()
        assert status.active_goals == 0
        assert status.stale_goals_count == 0

    @pytest.mark.asyncio
    async def test_get_status_bad_date_handling(self):
        """Goals with unparseable dates don't crash get_status."""
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "bad date",
                    "status": "active",
                    "updated_at": "not-a-date",
                }
            ]
        )
        mgr = HeartbeatManager(check_goal_fn=check_fn)
        status = await mgr.get_status()
        assert status.active_goals == 1
        assert status.stale_goals_count == 0

    @pytest.mark.asyncio
    async def test_get_status_missing_date_field(self):
        """Goals without updated_at or created_at are skipped in stale calc."""
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "no date",
                    "status": "active",
                    # no updated_at, no created_at
                }
            ]
        )
        mgr = HeartbeatManager(check_goal_fn=check_fn)
        status = await mgr.get_status()
        assert status.active_goals == 1
        assert status.stale_goals_count == 0

    @pytest.mark.asyncio
    async def test_get_status_uses_created_at_fallback(self):
        """When updated_at is absent, created_at is used as fallback."""
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "fallback",
                    "status": "active",
                    "created_at": "2020-01-01T00:00:00",
                }
            ]
        )
        mgr = HeartbeatManager(check_goal_fn=check_fn, stale_threshold_s=1)
        status = await mgr.get_status()
        assert status.active_goals == 1
        assert status.stale_goals_count == 1

    @pytest.mark.asyncio
    async def test_get_status_after_failed_tick(self):
        db_fn = MagicMock(return_value=False)
        mgr = HeartbeatManager(db_health_check=db_fn)
        await mgr._tick()
        status = await mgr.get_status()

        assert status.healthy is False
        assert status.db_connected is False
        assert status.total_ticks == 1
        assert status.failed_ticks == 1

    @pytest.mark.asyncio
    async def test_get_status_with_start_time(self):
        """When start() was called, uptime_seconds is computed."""
        mgr = HeartbeatManager()
        await mgr.start()
        # Set a well-known start time to avoid rounding issues with fast execution
        mgr._start_time = 1000.0
        status = await mgr.get_status()
        assert status.uptime_seconds > 0
        await mgr.stop()


# ---------------------------------------------------------------------------
# HeartbeatManager — _check_stale_goals()
# ---------------------------------------------------------------------------


class TestCheckStaleGoals:
    def test_no_fns_returns_zero(self):
        mgr = HeartbeatManager()
        result = mgr._check_stale_goals()
        assert result == {"found": 0, "marked_failed": 0}

    def test_only_check_fn_no_mark_fn(self):
        """If _mark_goal_fn is None, returns zero."""
        mgr = HeartbeatManager(check_goal_fn=MagicMock(return_value=[]))
        # _check_stale_goals checks both fns
        result = mgr._check_stale_goals()
        assert result == {"found": 0, "marked_failed": 0}

    def test_skips_terminal_goals(self):
        check_fn = MagicMock(
            return_value=[
                {"id": "g1", "status": "completed", "updated_at": "2020-01-01T00:00:00"},
                {"id": "g2", "status": "failed", "updated_at": "2020-01-01T00:00:00"},
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(check_goal_fn=check_fn, mark_goal_fn=mark_fn)
        result = mgr._check_stale_goals()
        assert result == {"found": 0, "marked_failed": 0}
        mark_fn.assert_not_called()

    def test_skips_goal_without_dates(self):
        check_fn = MagicMock(
            return_value=[{"id": "g1", "status": "active"}]  # no updated_at, no created_at
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(check_goal_fn=check_fn, mark_goal_fn=mark_fn)
        result = mgr._check_stale_goals()
        assert result == {"found": 0, "marked_failed": 0}
        mark_fn.assert_not_called()

    def test_skips_goal_with_bad_date(self):
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "status": "active",
                    "updated_at": "not-a-date",
                }
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(check_goal_fn=check_fn, mark_goal_fn=mark_fn)
        result = mgr._check_stale_goals()
        assert result == {"found": 0, "marked_failed": 0}
        mark_fn.assert_not_called()

    def test_marks_stale_goal_and_calls_callback(self):
        on_fail = MagicMock()
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "stale task",
                    "status": "active",
                    "updated_at": "2020-01-01T00:00:00",
                }
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
            on_goal_failed=on_fail,
            stale_threshold_s=1,
        )
        result = mgr._check_stale_goals()
        assert result == {"found": 1, "marked_failed": 1}
        mark_fn.assert_called_once_with("g1", "failed")
        on_fail.assert_called_once_with("g1", "stale task")

    def test_mark_fn_raises(self):
        """If mark_goal_fn raises, the error is logged and marked_failed stays 0."""
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "stale task",
                    "status": "active",
                    "updated_at": "2020-01-01T00:00:00",
                }
            ]
        )
        mark_fn = MagicMock(side_effect=RuntimeError("mark failed"))
        mgr = HeartbeatManager(
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
            stale_threshold_s=1,
        )
        result = mgr._check_stale_goals()
        # The goal was found stale but marking failed
        assert result == {"found": 1, "marked_failed": 0}

    def test_callback_raises(self):
        """If on_goal_failed raises, error is logged but mark still succeeds."""
        on_fail = MagicMock(side_effect=ValueError("callback error"))
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "stale task",
                    "status": "active",
                    "updated_at": "2020-01-01T00:00:00",
                }
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
            on_goal_failed=on_fail,
            stale_threshold_s=1,
        )
        result = mgr._check_stale_goals()
        assert result == {"found": 1, "marked_failed": 1}
        mark_fn.assert_called_once_with("g1", "failed")

    def test_mixed_goals(self):
        """Mix of stale, recent, and terminal goals."""
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "status": "active",
                    "updated_at": "2020-01-01T00:00:00",
                },  # stale
                {
                    "id": "g2",
                    "status": "completed",
                    "updated_at": "2020-01-01T00:00:00",
                },  # terminal, skipped
                {
                    "id": "g3",
                    "status": "active",
                    "updated_at": "2099-01-01T00:00:00",
                },  # recent
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
            stale_threshold_s=1,
        )
        result = mgr._check_stale_goals()
        assert result == {"found": 1, "marked_failed": 1}
        mark_fn.assert_called_once_with("g1", "failed")

    def test_uses_created_at_fallback(self):
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "status": "active",
                    "created_at": "2020-01-01T00:00:00",
                }
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
            stale_threshold_s=1,
        )
        result = mgr._check_stale_goals()
        assert result == {"found": 1, "marked_failed": 1}
        mark_fn.assert_called_once_with("g1", "failed")

    def test_not_stale(self):
        """A goal updated recently is not marked stale."""
        from datetime import datetime, timezone, timedelta

        recent = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "status": "active",
                    "updated_at": recent,
                }
            ]
        )
        mark_fn = MagicMock()
        mgr = HeartbeatManager(
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
            stale_threshold_s=3600,  # 1 hour threshold
        )
        result = mgr._check_stale_goals()
        assert result == {"found": 0, "marked_failed": 0}
        mark_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: full tick with all components
# ---------------------------------------------------------------------------


class TestHeartbeatManagerIntegration:
    @pytest.mark.asyncio
    async def test_full_tick_all_healthy(self):
        """A full tick with all checks passing."""
        db_fn = MagicMock(return_value=True)
        redis_fn = MagicMock(return_value=True)
        check_fn = MagicMock(
            return_value=[
                {
                    "id": "g1",
                    "goal": "test goal",
                    "status": "active",
                    "updated_at": "2099-01-01T00:00:00",
                }
            ]
        )
        mark_fn = MagicMock()
        on_fail = MagicMock()

        mgr = HeartbeatManager(
            db_health_check=db_fn,
            redis_health_check=redis_fn,
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
            on_goal_failed=on_fail,
        )
        result = await mgr._tick()

        assert result.db_connected is True
        assert result.redis_connected is True
        assert result.stale_goals_found == 0
        assert result.goals_marked_failed == 0
        assert result.errors == []
        assert mgr._healthy is True
        assert mgr._total_ticks == 1
        assert mgr._last_tick is not None

    @pytest.mark.asyncio
    async def test_full_tick_db_down_skips_stale_and_redis(self):
        """When DB check fails, stale goal check and redis check still run (they are independent)."""
        # Actually re-reading the code: Redis check runs regardless.
        # Stale goal check requires result.db_connected to be True.
        db_fn = MagicMock(return_value=False)
        redis_fn = MagicMock(return_value=True)
        check_fn = MagicMock(return_value=[])
        mark_fn = MagicMock()

        mgr = HeartbeatManager(
            db_health_check=db_fn,
            redis_health_check=redis_fn,
            check_goal_fn=check_fn,
            mark_goal_fn=mark_fn,
        )
        result = await mgr._tick()

        assert result.db_connected is False
        # Redis check still runs (independent)
        assert result.redis_connected is True
        # Stale check skipped because db_connected is False
        check_fn.assert_not_called()
        assert mgr._healthy is False
