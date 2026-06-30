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
