"""
routes/cortex_routes.py — FastAPI routes for CORTEX capabilities.

These expose the cortex/ modules as scoped API endpoints.
Integrates with Odysseus auth and skills system.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from cortex import goals, router as cortex_router, improve, skills_sh, cli_invoker
from cortex.smart_router import SmartRouter, TaskCategory, RoutingPriority
from cortex.csuite import CSuiteOrchestrator, CSuiteRole, CompanyContext
from cortex.sdd import SDDGenerator, PipelineOrchestrator

logger = logging.getLogger(__name__)

# Shared instances
_smart_router = SmartRouter()
_csuite = CSuiteOrchestrator()
_sdd_gen = SDDGenerator()
_pipeline = PipelineOrchestrator(_sdd_gen)


def setup_cortex_routes(
    skills_manager: Optional[object] = None,
) -> APIRouter:
    """Create the CORTEX API router.

    Args:
        skills_manager: Odysseus SkillsManager instance (for skills.sh install)
    """
    r = APIRouter(prefix="/api/cortex", tags=["cortex"])

    # ------------------------------------------------------------------
    # GET /api/cortex/capabilities
    # ------------------------------------------------------------------
    @r.get("/capabilities")
    async def get_capabilities():
        """List all CORTEX submodule capabilities."""
        return {
            "goals": {"decompose": True, "templates": 3},
            "router": {"route": True, "agents_supported": 5},
            "improve": {"trajectories": True, "pattern_detection": True},
            "skills_sh": {"search": True, "install": skills_manager is not None},
            "cli_invoker": {"discover": True, "invoke": True},
        }

    # ------------------------------------------------------------------
    # POST /api/cortex/goals/decompose
    # ------------------------------------------------------------------
    @r.post("/goals/decompose")
    async def decompose_goal(body: dict):
        """Decompose a high-level goal into structured subtasks."""
        goal_text = body.get("goal", "")
        if not goal_text.strip():
            raise HTTPException(status_code=400, detail="goal is required")
        try:
            decomp = goals.decompose(goal_text)
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
                        "capabilities": t.capabilities,
                        "depends_on": t.depends_on,
                        "optional": t.optional,
                        "estimated_cost_usd": t.estimated_cost_usd,
                        "estimated_duration_seconds": t.estimated_duration_seconds,
                    }
                    for t in decomp.tasks
                ],
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ------------------------------------------------------------------
    # GET /api/cortex/goals/templates
    # ------------------------------------------------------------------
    @r.get("/goals/templates")
    async def list_templates():
        """List available goal decomposition templates."""
        return {
            "templates": [
                {"id": "migrate-database", "description": "Database migration planning"},
                {"id": "audit-security", "description": "Security audit and vulnerability scan"},
                {"id": "refactor-module", "description": "Code refactoring with module splitting"},
            ]
        }

    # ------------------------------------------------------------------
    # POST /api/cortex/router/route
    # ------------------------------------------------------------------
    @r.post("/router/route")
    async def route_prompt(body: dict):
        """Route a prompt to the best available agent."""
        prompt = body.get("prompt", "")
        agents = body.get("agents", [])
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")
        if not agents:
            raise HTTPException(status_code=400, detail="agents list is required")

        decision = cortex_router.route(prompt, agents)
        return {
            "winner": decision.winner,
            "runner_up": decision.runner_up,
            "explanation": decision.explanation,
            "ranked": [
                {
                    "id": a.id,
                    "name": a.name,
                    "score": round(a.score, 3),
                    "quality": round(a.quality, 2),
                    "cost": round(a.cost, 2),
                    "recency": round(a.recency, 2),
                    "affinity": round(a.affinity, 2),
                }
                for a in decision.ranked
            ],
        }

    # ------------------------------------------------------------------
    # GET /api/cortex/improve/stats
    # ------------------------------------------------------------------
    @r.get("/improve/stats")
    async def improvement_stats():
        """Get self-improvement system statistics."""
        return improve.get_improvement_stats()

    # ------------------------------------------------------------------
    # POST /api/cortex/improve/record
    # ------------------------------------------------------------------
    @r.post("/improve/record")
    async def record_trajectory(body: dict):
        """Record an agent invocation trajectory."""
        traj = improve.Trajectory(
            session_id=body.get("session_id", ""),
            agent=body.get("agent", ""),
            prompt=body.get("prompt", ""),
            status=body.get("status", ""),
            exit_code=body.get("exit_code"),
            duration_seconds=body.get("duration_seconds", 0.0),
            files_changed=body.get("files_changed", []),
            stdout=body.get("stdout", ""),
            stderr=body.get("stderr", ""),
        )
        improve.record_trajectory(traj)
        return {"ok": True, "session_id": traj.session_id}

    # ------------------------------------------------------------------
    # GET /api/cortex/improve/patterns
    # ------------------------------------------------------------------
    @r.get("/improve/patterns")
    async def list_patterns():
        """List detected patterns from trajectories."""
        return {"patterns": [p for p in improve.load_patterns()]}

    # ------------------------------------------------------------------
    # GET /api/cortex/cli
    # ------------------------------------------------------------------
    @r.get("/cli")
    async def list_cli_agents():
        """Discover installed CLI agents on the system."""
        agents = cli_invoker.discover_agents()
        return {
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "binary": a.binary,
                    "path": a.path,
                    "version": a.version,
                    "capabilities": a.capabilities,
                    "cost_tier": a.cost_tier,
                }
                for a in agents
            ]
        }

    # ------------------------------------------------------------------
    # POST /api/cortex/cli/invoke
    # ------------------------------------------------------------------
    @r.post("/cli/invoke")
    async def invoke_cli(body: dict):
        """Invoke a CLI agent with a prompt."""
        agent_id = body.get("agent", "")
        prompt = body.get("prompt", "")
        if not agent_id or not prompt.strip():
            raise HTTPException(status_code=400, detail="agent and prompt are required")

        agents = cli_invoker.discover_agents()
        target = next((a for a in agents if a.id == agent_id), None)
        if not target:
            raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")

        result = cli_invoker.invoke(
            target,
            prompt,
            workdir=body.get("workdir"),
            timeout=body.get("timeout"),
        )
        return result

    # ------------------------------------------------------------------
    # GET /api/cortex/skills-sh/search
    # ------------------------------------------------------------------
    @r.get("/skills-sh/search")
    async def search_skills_sh(q: str = "", limit: int = 20):
        """Search skills.sh for available skills."""
        if not q:
            return {"skills": []}
        results = skills_sh.search(q, limit=limit)
        return {"skills": results, "query": q, "count": len(results)}

    # ------------------------------------------------------------------
    # POST /api/cortex/skills-sh/install
    # ------------------------------------------------------------------
    @r.post("/skills-sh/install")
    async def install_from_skills_sh(body: dict):
        """Install a skill from skills.sh into Odysseus."""
        name = body.get("name", "")
        if not name:
            raise HTTPException(status_code=400, detail="skill name is required")
        if skills_manager is None:
            raise HTTPException(status_code=503, detail="skills manager not available")

        result = skills_sh.install_skill(skills_manager, name)
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail=result.get("error", "install failed"))
        return result

    # ==================================================================
    # SMART ROUTER
    # ==================================================================

    # GET /api/cortex/router/categories
    @r.get("/router/categories")
    async def router_categories():
        """List all routing categories with recommended models."""
        return {"categories": _smart_router.list_categories()}

    # POST /api/cortex/router/route-smart
    @r.post("/router/route-smart")
    async def smart_route(body: dict):
        """Route a prompt to the best model/provider using SmartRouter."""
        prompt = body.get("prompt", "")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")

        # Parse optional category
        cat_str = body.get("category")
        category = None
        if cat_str:
            try:
                category = TaskCategory(cat_str)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"unknown category: {cat_str}")

        priority_str = body.get("priority", "quality")
        try:
            priority = RoutingPriority(priority_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown priority: {priority_str}")

        result = _smart_router.route(prompt, category=category, priority=priority)
        return result

    # ==================================================================
    # C-SUITE
    # ==================================================================

    # GET /api/cortex/csuite
    @r.get("/csuite")
    async def csuite_list_roles():
        """List all available C-Suite executive roles."""
        return {"roles": _csuite.list_roles()}

    # GET /api/cortex/csuite/{role}
    @r.get("/csuite/{role}")
    async def csuite_role_info(role: str):
        """Get information about a specific C-Suite role."""
        try:
            role_enum = CSuiteRole(role)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"unknown role: {role}")
        return _csuite.get_role_info(role_enum)

    # POST /api/cortex/csuite/{role}/query
    @r.post("/csuite/{role}/query")
    async def csuite_query(role: str, body: dict):
        """Query a C-Suite agent with a question.

        Returns the prepared prompt for Odysseus to use with its LLM.
        """
        question = body.get("question", "")
        if not question.strip():
            raise HTTPException(status_code=400, detail="question is required")
        try:
            role_enum = CSuiteRole(role)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"unknown role: {role}")

        # Update context if provided
        ctx = body.get("context")
        if ctx:
            _csuite.update_context(CompanyContext(**ctx))

        result = _csuite.prepare_prompt(role_enum, question)
        return result

    # POST /api/cortex/csuite/context
    @r.post("/csuite/context")
    async def csuite_set_context(body: dict):
        """Set the shared company context for C-Suite agents."""
        ctx = CompanyContext(**body)
        _csuite.update_context(ctx)
        return {"ok": True, "context": {
            "name": ctx.name,
            "industry": ctx.industry,
            "stage": ctx.stage,
            "team_size": ctx.team_size,
        }}

    # ==================================================================
    # SDD PIPELINE
    # ==================================================================

    # GET /api/cortex/sdd/steps
    @r.get("/sdd/steps")
    async def sdd_list_steps():
        """List all SDD pipeline steps."""
        return {"steps": _pipeline.list_steps()}

    # POST /api/cortex/sdd/generate
    @r.post("/sdd/generate")
    async def sdd_generate(body: dict):
        """Generate SDD documents for a goal."""
        goal_id = body.get("goal_id", "")
        goal = body.get("goal", "")
        if not goal_id or not goal:
            raise HTTPException(status_code=400, detail="goal_id and goal are required")
        docs = _sdd_gen.generate_all(goal_id, goal)
        path = docs.save()
        return {
            "goal_id": goal_id,
            "documents_path": path,
            "spec": docs.spec_content[:500],
            "plan": docs.plan_content[:500],
            "tasks": docs.tasks_content[:500],
        }

    # POST /api/cortex/sdd/pipeline
    @r.post("/sdd/pipeline")
    async def sdd_run_pipeline(body: dict):
        """Run the full SDD pipeline."""
        goal_id = body.get("goal_id", "")
        goal = body.get("goal", "")
        skip = body.get("skip")
        only = body.get("only")
        if not goal_id or not goal:
            raise HTTPException(status_code=400, detail="goal_id and goal are required")
        result = _pipeline.run(goal_id, goal, skip=skip, only=only)
        return result

    return r
