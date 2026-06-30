"""
cortex/tools.py — CORTEX tools for Odysseus agent mode.

Registers C-Suite, SmartRouter, and Goals as Odysseus tools
so they can be invoked from the chat interface.

Each function follows the Odysseus tool pattern:
    async def tool_name(content: str, owner: Optional[str] = None) -> Dict
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from cortex.csuite import CSuiteOrchestrator, CSuiteRole, CompanyContext
from cortex.smart_router import SmartRouter, TaskCategory, RoutingPriority
from cortex.goals import decompose as goals_decompose

logger = logging.getLogger(__name__)

# Shared instances
_smart_router = SmartRouter()
_csuite = CSuiteOrchestrator()


async def do_route_prompt(content: str, owner: Optional[str] = None) -> Dict:
    """Route a prompt to the best model/provider.

    Analyzes the prompt and recommends the optimal LLM model
    based on task category, with cost estimation.

    Args:
        content: JSON with keys:
            prompt (str): The user prompt to route
            priority (str, optional): "quality" | "cost" | "speed"
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    prompt = args.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required", "exit_code": 1}

    priority_str = args.get("priority", "quality")
    try:
        priority = RoutingPriority(priority_str)
    except ValueError:
        return {"error": f"unknown priority: {priority_str}", "exit_code": 1}

    result = _smart_router.route(prompt, priority=priority)
    return {
        "category": result["category"],
        "label": result["label"],
        "recommended_model": result["recommended_model"],
        "recommended_provider": result["recommended_provider"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "priority": result["priority"],
    }


async def do_csuite_query(content: str, owner: Optional[str] = None) -> Dict:
    """Query a C-Suite executive agent.

    Ask a strategic question to one of the 6 executive roles
    (CEO, CTO, CMO, CFO, QA, R&D).

    Args:
        content: JSON with keys:
            role (str): "ceo" | "cto" | "cmo" | "cfo" | "qa" | "rd"
            question (str): The strategic question to ask
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    role_str = args.get("role", "").lower()
    question = args.get("question", "")
    if not role_str:
        return {"error": "role is required (ceo/cto/cmo/cfo/qa/rd)", "exit_code": 1}
    if not question:
        return {"error": "question is required", "exit_code": 1}

    try:
        role_enum = CSuiteRole(role_str)
    except ValueError:
        return {"error": f"unknown role: {role_str}", "exit_code": 1}

    prompt = _csuite.prepare_prompt(role_enum, question)
    return {
        "role": prompt["role"],
        "role_name": prompt["role_name"],
        "system_prompt": prompt["system_prompt"],
        "user_prompt": prompt["user_prompt"],
        "capabilities": prompt["capabilities"],
    }


async def do_decompose_goal(content: str, owner: Optional[str] = None) -> Dict:
    """Decompose a high-level goal into structured subtasks.

    Breaks down a goal into actionable tasks with agent assignments,
    dependencies, and cost estimates.

    Args:
        content: JSON with key:
            goal (str): The high-level goal to decompose
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    goal = args.get("goal", "")
    if not goal:
        return {"error": "goal is required", "exit_code": 1}

    try:
        decomp = goals_decompose(goal)
    except ValueError as e:
        return {"error": str(e), "exit_code": 1}

    return {
        "goal": decomp.goal,
        "source": decomp.source,
        "template_id": decomp.template_id,
        "total_estimated_cost_usd": decomp.total_estimated_cost_usd,
        "total_estimated_duration_seconds": decomp.total_estimated_duration_seconds,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "agent": t.agent,
                "depends_on": t.depends_on,
                "optional": t.optional,
            }
            for t in decomp.tasks
        ],
    }


async def do_list_csuite_roles(content: str, owner: Optional[str] = None) -> Dict:
    """List all available C-Suite executive agent roles.

    Returns each role's name, description, and capabilities.
    No arguments needed.
    """
    roles = _csuite.list_roles()
    return {"roles": roles}


async def do_router_categories(content: str, owner: Optional[str] = None) -> Dict:
    """List all routing categories with their recommended models.

    Shows what model each task category uses and the cost tier.
    No arguments needed.
    """
    categories = _smart_router.list_categories()
    return {"categories": categories}


# ==================================================================
# Goal Tracking Tools (Fase 4)
# ==================================================================

_tracker_instance = None


def _get_tracker():
    global _tracker_instance
    if _tracker_instance is None:
        from cortex.goal_state import GoalTracker
        _tracker_instance = GoalTracker()
    return _tracker_instance


async def do_create_goal(content: str, owner: Optional[str] = None) -> Dict:
    """Create a new tracked goal with state machine lifecycle.

    Creates a goal in CREATED state and auto-advances to ANALYZING.
    The goal can then be transitioned through: analyzing, planning,
    decomposing, assigning, executing, verifying, completed/failed.

    Args:
        content: JSON with keys:
            goal_id (str): Unique identifier for the goal
            goal (str): The goal description
            context (dict, optional): Initial context data
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    goal_id = args.get("goal_id", "")
    goal = args.get("goal", "")
    if not goal_id or not goal:
        return {"error": "goal_id and goal are required", "exit_code": 1}

    try:
        from cortex.goals import decompose_and_track
        tracker = _get_tracker()
        session = decompose_and_track(goal_id, goal, tracker, context=args.get("context"))
        return {
            "goal_id": session.goal_id,
            "status": session.status.value,
            "progress_pct": session.progress_pct,
            "created_at": session.created_at,
        }
    except ValueError as e:
        return {"error": str(e), "exit_code": 1}


async def do_goal_status(content: str, owner: Optional[str] = None) -> Dict:
    """Get the current status of a tracked goal.

    Args:
        content: JSON with key:
            goal_id (str): The goal to check
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    goal_id = args.get("goal_id", "")
    if not goal_id:
        return {"error": "goal_id is required", "exit_code": 1}

    tracker = _get_tracker()
    session = tracker.get_goal(goal_id)
    if not session:
        return {"error": f"goal '{goal_id}' not found", "exit_code": 1}

    return {
        "goal_id": session.goal_id,
        "goal": session.goal[:80],
        "status": session.status.value,
        "progress_pct": session.progress_pct,
        "task_count": session.task_count,
        "tasks_completed": session.tasks_completed,
        "checkpoints": len(session.checkpoints),
        "error": session.error,
        "is_terminal": session.is_terminal,
        "is_active": session.is_active,
    }


async def do_goal_transition(content: str, owner: Optional[str] = None) -> Dict:
    """Transition a goal to a specific state.

    Args:
        content: JSON with keys:
            goal_id (str): The goal to transition
            status (str): Target state (analyzing, planning, decomposing,
                          assigning, executing, verifying, completed, failed)
            error (str, optional): Error message if transitioning to failed
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    goal_id = args.get("goal_id", "")
    status_str = args.get("status", "")
    if not goal_id or not status_str:
        return {"error": "goal_id and status are required", "exit_code": 1}

    from cortex.goal_state import GoalStatus
    try:
        to_status = GoalStatus(status_str)
    except ValueError:
        return {"error": f"unknown status: {status_str}", "exit_code": 1}

    tracker = _get_tracker()
    try:
        session = tracker.transition(goal_id, to_status, error=args.get("error"))
        return {
            "goal_id": session.goal_id,
            "status": session.status.value,
            "progress_pct": session.progress_pct,
        }
    except ValueError as e:
        return {"error": str(e), "exit_code": 1}


async def do_goal_advance(content: str, owner: Optional[str] = None) -> Dict:
    """Advance a goal to the next logical state.

    Args:
        content: JSON with key:
            goal_id (str): The goal to advance
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    goal_id = args.get("goal_id", "")
    if not goal_id:
        return {"error": "goal_id is required", "exit_code": 1}

    tracker = _get_tracker()
    try:
        session = tracker.advance(goal_id)
        return {
            "goal_id": session.goal_id,
            "status": session.status.value,
            "progress_pct": session.progress_pct,
        }
    except ValueError as e:
        return {"error": str(e), "exit_code": 1}


async def do_goal_checkpoint(content: str, owner: Optional[str] = None) -> Dict:
    """Create a checkpoint for a goal to enable rollback.

    Args:
        content: JSON with keys:
            goal_id (str): The goal to checkpoint
            context (dict, optional): Context to save with checkpoint
            outputs (dict, optional): Step outputs to save
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    goal_id = args.get("goal_id", "")
    if not goal_id:
        return {"error": "goal_id is required", "exit_code": 1}

    tracker = _get_tracker()
    try:
        cp = tracker.checkpoint(
            goal_id,
            context=args.get("context"),
            outputs=args.get("outputs"),
        )
        return {
            "checkpoint_id": cp.id,
            "state": cp.state.value,
            "progress": cp.progress,
            "timestamp": cp.timestamp,
        }
    except ValueError as e:
        return {"error": str(e), "exit_code": 1}


async def do_goal_update_progress(content: str, owner: Optional[str] = None) -> Dict:
    """Update a goal's progress percentage and task counts.

    Args:
        content: JSON with keys:
            goal_id (str): The goal to update
            progress (float): Progress 0.0-1.0
            task_count (int, optional): Total task count
            tasks_completed (int, optional): Completed tasks
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return {"error": "Invalid JSON", "exit_code": 1}

    goal_id = args.get("goal_id", "")
    if not goal_id:
        return {"error": "goal_id is required", "exit_code": 1}

    tracker = _get_tracker()
    try:
        session = tracker.update_progress(
            goal_id,
            progress=args.get("progress", 0.0),
            task_count=args.get("task_count"),
            tasks_completed=args.get("tasks_completed"),
        )
        return {
            "goal_id": session.goal_id,
            "progress_pct": session.progress_pct,
            "task_count": session.task_count,
            "tasks_completed": session.tasks_completed,
        }
    except ValueError as e:
        return {"error": str(e), "exit_code": 1}


async def do_list_goals(content: str, owner: Optional[str] = None) -> Dict:
    """List all tracked goals, optionally filtered by status.

    Args:
        content: JSON with key:
            status (str, optional): Filter by status
    """
    try:
        args = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        args = {}

    from cortex.goal_state import GoalStatus
    status_filter = None
    if args.get("status"):
        try:
            status_filter = GoalStatus(args["status"])
        except ValueError:
            pass

    tracker = _get_tracker()
    sessions = tracker.list_goals(status=status_filter)
    return {
        "goals": [
            {
                "goal_id": s.goal_id,
                "goal": s.goal[:80],
                "status": s.status.value,
                "progress_pct": s.progress_pct,
                "created_at": s.created_at,
            }
            for s in sessions
        ],
        "count": len(sessions),
    }

