"""
cortex/heartbeat.py — Periodic health checks and goal monitoring.

Port of ADABA/codeWriter v2 HeartbeatManager.ts.

Checks database connectivity, detects stale goals, and reports
system health. Runs as a background asyncio task.

Integrates with Odysseus's existing database and task scheduler.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatTickResult:
    db_connected: bool = False
    redis_connected: bool = False
    stale_goals_found: int = 0
    goals_marked_failed: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class HeartbeatStatus:
    last_tick: Optional[str] = None
    healthy: bool = True
    uptime_seconds: float = 0.0
    active_goals: int = 0
    stale_goals_count: int = 0
    goals_marked_failed: int = 0
    db_connected: bool = False
    redis_connected: bool = False
    total_ticks: int = 0
    failed_ticks: int = 0


class HeartbeatManager:
    """Periodic health check and stale goal detection.

    Runs heartbeat ticks on a configurable interval. Each tick:
    1. Checks database connectivity
    2. Detects stale goals (>30min without updates → marks as failed)
    3. Optionally checks Redis
    4. Logs results

    Integrates with Odysseus via callback for stale goal notification.
    """

    def __init__(
        self,
        tick_interval_s: int = 60,
        stale_threshold_s: int = 1800,  # 30 min
        db_health_check: Optional[Callable[[], bool]] = None,
        redis_health_check: Optional[Callable[[], bool]] = None,
        on_goal_failed: Optional[Callable[[str, str], None]] = None,
        check_goal_fn: Optional[Callable[[], List[Dict]]] = None,
        mark_goal_fn: Optional[Callable[[str, str], None]] = None,
    ):
        self.tick_interval_s = tick_interval_s
        self.stale_threshold_s = stale_threshold_s

        # Dependency injection for Odysseus integration
        self._db_health_check = db_health_check
        self._redis_health_check = redis_health_check
        self._on_goal_failed = on_goal_failed
        self._check_goal_fn = check_goal_fn
        self._mark_goal_fn = mark_goal_fn

        # State
        self._task: Optional[asyncio.Task] = None
        self._last_tick: Optional[datetime] = None
        self._healthy: bool = True
        self._start_time: float = 0.0
        self._total_ticks: int = 0
        self._failed_ticks: int = 0
        self._goals_marked_failed_total: int = 0

    async def start(self) -> None:
        """Start the heartbeat loop."""
        if self._task is not None:
            logger.warning("HeartbeatManager already running")
            return

        self._start_time = time.time()
        logger.info(
            "HeartbeatManager starting (interval=%ds, stale_threshold=%ds)",
            self.tick_interval_s,
            self.stale_threshold_s,
        )

        # Run first tick immediately, then on interval
        await self._tick()

        self._task = asyncio.create_task(self._loop())
        logger.info("HeartbeatManager started")

    async def stop(self) -> None:
        """Stop the heartbeat loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("HeartbeatManager stopped")

    async def _loop(self) -> None:
        """Background loop that runs ticks on interval."""
        try:
            while True:
                await asyncio.sleep(self.tick_interval_s)
                await self._tick()
        except asyncio.CancelledError:
            pass

    async def tick(self) -> HeartbeatTickResult:
        """Force a single heartbeat tick.

        Public method — call to trigger an immediate health check.
        """
        return await self._tick()

    async def _tick(self) -> HeartbeatTickResult:
        """Execute a single heartbeat tick."""
        tick_start = time.time()
        result = HeartbeatTickResult()
        self._total_ticks += 1

        try:
            # 1. Database health check
            if self._db_health_check:
                try:
                    result.db_connected = self._db_health_check()
                except Exception as e:
                    result.db_connected = False
                    err_msg = f"Database check failed: {e}"
                    result.errors.append(err_msg)
                    logger.warning("Heartbeat DB health check failed: %s", e)
            else:
                result.db_connected = True  # assume OK if no check configured

            # 2. Stale goal detection
            if self._check_goal_fn and self._mark_goal_fn and result.db_connected:
                try:
                    stale_result = self._check_stale_goals()
                    result.stale_goals_found = stale_result["found"]
                    result.goals_marked_failed = stale_result["marked_failed"]
                    self._goals_marked_failed_total += result.goals_marked_failed
                except Exception as e:
                    err_msg = f"Stale goal check failed: {e}"
                    result.errors.append(err_msg)
                    logger.warning("Heartbeat stale goal check error: %s", e)

            # 3. Redis health check
            if self._redis_health_check:
                try:
                    result.redis_connected = self._redis_health_check()
                except Exception:
                    result.redis_connected = False

            # 4. Update health
            self._healthy = result.db_connected
            self._last_tick = datetime.now(timezone.utc)

            if not self._healthy:
                self._failed_ticks += 1

        except Exception as e:
            self._healthy = False
            self._failed_ticks += 1
            logger.error("Heartbeat tick critical error: %s", e)
            result.errors.append(f"Critical error: {e}")

        try:
            result.duration_ms = round((time.time() - tick_start) * 1000, 2)
        except Exception:
            result.duration_ms = 0.0

        if result.errors:
            logger.warning(
                "Heartbeat tick completed with %d error(s): %s",
                len(result.errors),
                "; ".join(result.errors),
            )

        return result

    async def get_status(self) -> HeartbeatStatus:
        """Get current heartbeat status with active goal counts."""
        status = HeartbeatStatus(
            last_tick=self._last_tick.isoformat() if self._last_tick else None,
            healthy=self._healthy,
            uptime_seconds=round(time.time() - self._start_time, 1)
            if self._start_time
            else 0.0,
            db_connected=False,
            redis_connected=False,
            total_ticks=self._total_ticks,
            failed_ticks=self._failed_ticks,
            goals_marked_failed=self._goals_marked_failed_total,
        )

        # Try to count active/stale goals
        if self._check_goal_fn:
            try:
                goals = self._check_goal_fn()
                now = time.time()
                active = 0
                stale = 0
                for g in goals:
                    g_status = g.get("status", "")
                    if g_status not in ("completed", "failed"):
                        active += 1
                        updated = g.get("updated_at") or g.get("created_at", "")
                        if updated:
                            try:
                                updated_ts = datetime.fromisoformat(updated).timestamp()
                                if now - updated_ts > self.stale_threshold_s:
                                    stale += 1
                            except (ValueError, TypeError):
                                pass
                status.active_goals = active
                status.stale_goals_count = stale
            except Exception:
                pass

        return status

    def _check_stale_goals(self) -> Dict[str, int]:
        """Check for stale goals and mark them as failed."""
        if not self._check_goal_fn or not self._mark_goal_fn:
            return {"found": 0, "marked_failed": 0}

        goals = self._check_goal_fn()
        now = time.time()
        found = 0
        marked_failed = 0

        for goal in goals:
            # Skip terminal goals
            g_status = goal.get("status", "")
            if g_status in ("completed", "failed"):
                continue

            # Check age since last update
            updated = goal.get("updated_at") or goal.get("created_at", "")
            if not updated:
                continue

            try:
                updated_ts = datetime.fromisoformat(updated).timestamp()
            except (ValueError, TypeError):
                continue

            age = now - updated_ts
            if age > self.stale_threshold_s:
                found += 1
                try:
                    goal_id = goal.get("id", "")
                    self._mark_goal_fn(goal_id, "failed")
                    marked_failed += 1
                    logger.warning(
                        "Marked stale goal as failed: %s (inactive %d min)",
                        goal_id,
                        round(age / 60),
                    )

                    # Notify callback
                    if self._on_goal_failed:
                        try:
                            self._on_goal_failed(goal_id, goal.get("goal", ""))
                        except Exception as e:
                            logger.error("on_goal_failed callback error: %s", e)
                except Exception as e:
                    logger.error(
                        "Failed to mark goal %s as failed: %s", goal.get("id", "?"), e
                    )

        return {"found": found, "marked_failed": marked_failed}
