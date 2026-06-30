"""
cortex/goal_state.py — 8-state machine for Goal lifecycle.

Port of ADABA's GoalTracker + state machine to Python.
Tracks every goal through its full lifecycle with:
  - 8 explicit states (created → analyzing → planning → decomposing
    → assigning → executing → verifying → completed / failed)
  - Progress percentage per state
  - Checkpoints for resume after interruption
  - State transition validation (which transitions are legal)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

GOALS_DIR = "data/cortex/goals"

# ---------------------------------------------------------------------------
# 8-State Machine
# ---------------------------------------------------------------------------


class GoalStatus(str, Enum):
    """The 8 states a goal passes through during its lifecycle."""

    CREATED = "created"  # Initial state — goal just created
    ANALYZING = "analyzing"  # Analyzing requirements and scope
    PLANNING = "planning"  # Creating implementation plan
    DECOMPOSING = "decomposing"  # Breaking into subtasks
    ASSIGNING = "assigning"  # Assigning subtasks to agents
    EXECUTING = "executing"  # Executing subtasks
    VERIFYING = "verifying"  # Verifying results against criteria
    COMPLETED = "completed"  # Goal completed successfully
    FAILED = "failed"  # Goal failed (non-retryable)


# Legal transitions: state -> set of allowed next states
_TRANSITIONS: Dict[GoalStatus, Set[GoalStatus]] = {
    GoalStatus.CREATED: {GoalStatus.ANALYZING, GoalStatus.FAILED},
    GoalStatus.ANALYZING: {GoalStatus.PLANNING, GoalStatus.FAILED},
    GoalStatus.PLANNING: {GoalStatus.DECOMPOSING, GoalStatus.FAILED},
    GoalStatus.DECOMPOSING: {GoalStatus.ASSIGNING, GoalStatus.FAILED},
    GoalStatus.ASSIGNING: {GoalStatus.EXECUTING, GoalStatus.FAILED},
    GoalStatus.EXECUTING: {GoalStatus.VERIFYING, GoalStatus.FAILED},
    GoalStatus.VERIFYING: {
        GoalStatus.COMPLETED,
        GoalStatus.EXECUTING,
        GoalStatus.FAILED,
    },
    GoalStatus.COMPLETED: set(),  # Terminal
    GoalStatus.FAILED: {GoalStatus.CREATED},  # Can retry from scratch
}

# Progress weights per state (for % calculation)
_STATE_WEIGHTS = {
    GoalStatus.CREATED: 0.0,
    GoalStatus.ANALYZING: 0.1,
    GoalStatus.PLANNING: 0.2,
    GoalStatus.DECOMPOSING: 0.3,
    GoalStatus.ASSIGNING: 0.4,
    GoalStatus.EXECUTING: 0.7,
    GoalStatus.VERIFYING: 0.9,
    GoalStatus.COMPLETED: 1.0,
    GoalStatus.FAILED: 0.0,
}


def validate_transition(from_state: GoalStatus, to_state: GoalStatus) -> None:
    """Raise ValueError if the transition is not allowed."""
    allowed = _TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise ValueError(
            f"Illegal transition: {from_state.value} → {to_state.value}. "
            f"Allowed: {[s.value for s in allowed]}"
        )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """A snapshot of goal state at a point in time."""

    id: str
    timestamp: float
    state: GoalStatus
    progress: float
    context: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)  # step_name -> output


# ---------------------------------------------------------------------------
# GoalSession — persisted goal with full lifecycle
# ---------------------------------------------------------------------------


@dataclass
class GoalSession:
    """A tracked goal with full state machine lifecycle."""

    goal_id: str
    goal: str
    status: GoalStatus = GoalStatus.CREATED
    created_at: float = 0.0
    updated_at: float = 0.0
    progress: float = 0.0
    checkpoints: List[Checkpoint] = field(default_factory=list)
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    task_count: int = 0
    tasks_completed: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED)

    @property
    def is_active(self) -> bool:
        return not self.is_terminal and self.status != GoalStatus.CREATED

    @property
    def progress_pct(self) -> int:
        return int(self.progress * 100)


# ---------------------------------------------------------------------------
# GoalTracker — manages state machine, checkpoints, persistence
# ---------------------------------------------------------------------------


class GoalTracker:
    """Manages goal sessions through their 8-state lifecycle.

    Persists to disk (JSON) for durability across restarts.
    Supports checkpoint save/load for long-running goals.
    """

    def __init__(self, base_dir: str = GOALS_DIR):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, GoalSession] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_goal(
        self, goal_id: str, goal: str, context: Optional[Dict[str, Any]] = None
    ) -> GoalSession:
        """Create a new goal in CREATED state."""
        if goal_id in self._sessions:
            raise ValueError(f"Goal '{goal_id}' already exists")

        now = time.time()
        session = GoalSession(
            goal_id=goal_id,
            goal=goal,
            status=GoalStatus.CREATED,
            created_at=now,
            updated_at=now,
            context=context or {},
        )
        self._sessions[goal_id] = session
        self._save(session)
        logger.info("Goal created: %s — %s", goal_id, goal[:60])
        return session

    def get_goal(self, goal_id: str) -> Optional[GoalSession]:
        return self._sessions.get(goal_id)

    def list_goals(self, status: Optional[GoalStatus] = None) -> List[GoalSession]:
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def delete_goal(self, goal_id: str) -> bool:
        session = self._sessions.pop(goal_id, None)
        if session:
            path = self._goal_path(goal_id)
            if path.exists():
                path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def transition(
        self, goal_id: str, to_status: GoalStatus, error: Optional[str] = None
    ) -> GoalSession:
        """Transition a goal to a new state. Validates legality."""
        session = self._get_or_raise(goal_id)
        validate_transition(session.status, to_status)

        # Create checkpoint before transition
        self._checkpoint(session)

        session.status = to_status
        session.updated_at = time.time()
        session.progress = _STATE_WEIGHTS.get(to_status, session.progress)

        if to_status == GoalStatus.FAILED:
            session.error = error or "Unknown error"

        self._save(session)
        logger.info("Goal %s: %s → %s", goal_id, session.status.value, to_status.value)
        return session

    def advance(self, goal_id: str) -> GoalSession:
        """Advance to the next logical state in the pipeline."""
        session = self._get_or_raise(goal_id)
        order = list(GoalStatus)
        try:
            idx = order.index(session.status)
        except ValueError:
            raise ValueError(f"Unknown state: {session.status}")

        if idx >= len(order) - 1:
            raise ValueError(
                f"Goal '{goal_id}' is already in terminal state '{session.status.value}'"
            )

        next_state = order[idx + 1]
        # Skip over terminal states if trying to advance from penultimate
        if (
            next_state in (GoalStatus.COMPLETED, GoalStatus.FAILED)
            and session.status != GoalStatus.VERIFYING
        ):
            # If we're not at verifying, advance to next logical state
            for s in order[idx + 1 :]:
                if s not in (GoalStatus.COMPLETED, GoalStatus.FAILED):
                    next_state = s
                    break
        return self.transition(goal_id, next_state)

    def fail(self, goal_id: str, error: str) -> GoalSession:
        """Mark a goal as failed with error message."""
        return self.transition(goal_id, GoalStatus.FAILED, error=error)

    def complete(self, goal_id: str) -> GoalSession:
        """Mark a goal as completed."""
        return self.transition(goal_id, GoalStatus.COMPLETED)

    def retry(self, goal_id: str) -> GoalSession:
        """Retry a failed goal from scratch (CREATED state)."""
        session = self._get_or_raise(goal_id)
        if session.status != GoalStatus.FAILED:
            raise ValueError(
                f"Can only retry FAILED goals, got '{session.status.value}'"
            )
        return self.transition(goal_id, GoalStatus.CREATED)

    # ------------------------------------------------------------------
    # Progress & tracking
    # ------------------------------------------------------------------

    def update_progress(
        self,
        goal_id: str,
        progress: float,
        task_count: Optional[int] = None,
        tasks_completed: Optional[int] = None,
    ) -> GoalSession:
        """Update progress percentage and optionally task counts."""
        session = self._get_or_raise(goal_id)
        session.progress = max(0.0, min(1.0, progress))
        if task_count is not None:
            session.task_count = task_count
        if tasks_completed is not None:
            session.tasks_completed = tasks_completed
        session.updated_at = time.time()
        self._save(session)
        return session

    def update_context(self, goal_id: str, **kwargs: Any) -> GoalSession:
        """Update arbitrary context fields on a goal session."""
        session = self._get_or_raise(goal_id)
        session.context.update(kwargs)
        session.updated_at = time.time()
        self._save(session)
        return session

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def checkpoint(
        self,
        goal_id: str,
        context: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, str]] = None,
    ) -> Checkpoint:
        """Create a manual checkpoint for a goal."""
        session = self._get_or_raise(goal_id)
        return self._checkpoint(session, context=context, outputs=outputs)

    def get_checkpoints(self, goal_id: str) -> List[Checkpoint]:
        session = self._get_or_raise(goal_id)
        return list(session.checkpoints)

    def restore_checkpoint(self, goal_id: str, checkpoint_id: str) -> GoalSession:
        """Restore a goal to a previous checkpoint state."""
        session = self._get_or_raise(goal_id)
        cp = next((c for c in session.checkpoints if c.id == checkpoint_id), None)
        if not cp:
            raise ValueError(
                f"Checkpoint '{checkpoint_id}' not found for goal '{goal_id}'"
            )

        # Restore state and progress from checkpoint
        session.status = cp.state
        session.progress = cp.progress
        session.context.update(cp.context)
        session.error = None
        session.updated_at = time.time()
        self._save(session)
        logger.info(
            "Goal %s restored to checkpoint %s (%s)",
            goal_id,
            checkpoint_id,
            cp.state.value,
        )
        return session

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_raise(self, goal_id: str) -> GoalSession:
        session = self._sessions.get(goal_id)
        if not session:
            raise ValueError(f"Goal '{goal_id}' not found")
        return session

    def _checkpoint(
        self,
        session: GoalSession,
        context: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, str]] = None,
    ) -> Checkpoint:
        """Create a checkpoint from the current session state."""
        cp = Checkpoint(
            id=f"cp-{int(time.time() * 1000)}-{len(session.checkpoints)}",
            timestamp=time.time(),
            state=session.status,
            progress=session.progress,
            context=context or dict(session.context),
            outputs=outputs or {},
        )
        session.checkpoints.append(cp)
        self._save(session)
        return cp

    def _goal_path(self, goal_id: str) -> Path:
        return self._base_dir / f"{goal_id}.json"

    def _save(self, session: GoalSession) -> None:
        """Persist a goal session to disk."""
        path = self._goal_path(session.goal_id)
        data = {
            "goal_id": session.goal_id,
            "goal": session.goal,
            "status": session.status.value,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "progress": session.progress,
            "error": session.error,
            "context": session.context,
            "task_count": session.task_count,
            "tasks_completed": session.tasks_completed,
            "checkpoints": [
                {
                    "id": c.id,
                    "timestamp": c.timestamp,
                    "state": c.state.value,
                    "progress": c.progress,
                    "context": c.context,
                    "outputs": c.outputs,
                }
                for c in session.checkpoints
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except PermissionError:
            # Fallback to temp
            import tempfile

            fallback = (
                Path(tempfile.gettempdir()) / "cortex-goals" / f"{session.goal_id}.json"
            )
            fallback.parent.mkdir(parents=True, exist_ok=True)
            with open(fallback, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.warning(
                "Saved goal %s to fallback path %s", session.goal_id, fallback
            )

    def _load_all(self) -> None:
        """Load all persisted goal sessions."""
        if not self._base_dir.exists():
            return
        for path in self._base_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                checkpoints = []
                for c in data.get("checkpoints", []):
                    checkpoints.append(
                        Checkpoint(
                            id=c["id"],
                            timestamp=c["timestamp"],
                            state=GoalStatus(c["state"]),
                            progress=c["progress"],
                            context=c.get("context", {}),
                            outputs=c.get("outputs", {}),
                        )
                    )
                session = GoalSession(
                    goal_id=data["goal_id"],
                    goal=data["goal"],
                    status=GoalStatus(data["status"]),
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                    progress=data.get("progress", 0.0),
                    error=data.get("error"),
                    context=data.get("context", {}),
                    task_count=data.get("task_count", 0),
                    tasks_completed=data.get("tasks_completed", 0),
                    checkpoints=checkpoints,
                )
                self._sessions[session.goal_id] = session
            except Exception as e:
                logger.warning("Failed to load goal from %s: %s", path, e)
        logger.info(
            "Loaded %d goal sessions from %s", len(self._sessions), self._base_dir
        )
