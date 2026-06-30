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
from cortex.heartbeat import HeartbeatManager
from cortex.goal_state import GoalTracker, GoalStatus

logger = logging.getLogger(__name__)

# Shared instances
_smart_router = SmartRouter()
_csuite = CSuiteOrchestrator()
_sdd_gen = SDDGenerator()
_pipeline = PipelineOrchestrator(_sdd_gen)
_heartbeat = HeartbeatManager(
    tick_interval_s=60,
    stale_threshold_s=1800,
)


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
            "goals": {"decompose": True, "templates": 3, "state_machine": True,
                      "tracking": True, "checkpoints": True, "presets": 5},
            "router": {"route": True, "smart_route": True, "categories": 9, "agents_supported": 5},
            "improve": {"trajectories": True, "pattern_detection": True},
            "skills_sh": {"search": True, "install": skills_manager is not None},
            "cli_invoker": {"discover": True, "invoke": True},
            "smart_router": {"models": 24, "providers": 8, "categories": 9},
            "csuite": {"roles": 6, "query": True, "context": True},
            "sdd": {"generate": True, "pipeline_steps": 7, "llm_generate": _sdd_gen._llm_available},
            "heartbeat": {"status": True, "tick": True, "interval_s": 60, "ws_notify": True},
            "integration": {"chat_routing": True, "goal_scheduling": True,
                           "sdd_llm": _sdd_gen._llm_available, "heartbeat_ws": True},
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

    # ==================================================================
    # HEARTBEAT
    # ==================================================================

    # GET /api/cortex/heartbeat/status
    @r.get("/heartbeat/status")
    async def heartbeat_status():
        """Get heartbeat system status."""
        status = await _heartbeat.get_status()
        return {
            "last_tick": status.last_tick,
            "healthy": status.healthy,
            "uptime_seconds": status.uptime_seconds,
            "active_goals": status.active_goals,
            "stale_goals_count": status.stale_goals_count,
            "goals_marked_failed": status.goals_marked_failed,
            "total_ticks": status.total_ticks,
            "failed_ticks": status.failed_ticks,
        }

    # POST /api/cortex/heartbeat/tick
    @r.post("/heartbeat/tick")
    async def heartbeat_tick():
        """Force an immediate heartbeat tick."""
        result = await _heartbeat.tick()
        return {
            "db_connected": result.db_connected,
            "redis_connected": result.redis_connected,
            "stale_goals_found": result.stale_goals_found,
            "goals_marked_failed": result.goals_marked_failed,
            "duration_ms": result.duration_ms,
            "errors": result.errors,
        }


    # ==================================================================
    # GOAL STATE MACHINE (Fase 4)
    # ==================================================================

    _tracker = GoalTracker()

    # POST /api/cortex/goals/session
    @r.post("/goals/session")
    async def goal_create(body: dict):
        """Create a new tracked goal session."""
        goal_id = body.get("goal_id", "")
        goal = body.get("goal", "")
        if not goal_id or not goal:
            raise HTTPException(status_code=400, detail="goal_id and goal are required")
        try:
            session = goals.decompose_and_track(
                goal_id, goal, _tracker,
                context=body.get("context"),
            )
            return {
                "goal_id": session.goal_id,
                "status": session.status.value,
                "progress_pct": session.progress_pct,
                "created_at": session.created_at,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # GET /api/cortex/goals/session/{goal_id}
    @r.get("/goals/session/{goal_id}")
    async def goal_get(goal_id: str):
        """Get a goal session with full details."""
        session = _tracker.get_goal(goal_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"goal '{goal_id}' not found")
        return {
            "goal_id": session.goal_id,
            "goal": session.goal,
            "status": session.status.value,
            "progress_pct": session.progress_pct,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "error": session.error,
            "task_count": session.task_count,
            "tasks_completed": session.tasks_completed,
            "checkpoints": len(session.checkpoints),
            "context": session.context,
        }

    # GET /api/cortex/goals/sessions
    @r.get("/goals/sessions")
    async def goal_list(status: Optional[str] = None):
        """List all goal sessions, optionally filtered by status."""
        status_filter = None
        if status:
            try:
                status_filter = GoalStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"unknown status: {status}")
        sessions = _tracker.list_goals(status=status_filter)
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

    # DELETE /api/cortex/goals/session/{goal_id}
    @r.delete("/goals/session/{goal_id}")
    async def goal_delete(goal_id: str):
        """Delete a goal session."""
        if _tracker.delete_goal(goal_id):
            return {"ok": True}
        raise HTTPException(status_code=404, detail=f"goal '{goal_id}' not found")

    # POST /api/cortex/goals/session/{goal_id}/transition
    @r.post("/goals/session/{goal_id}/transition")
    async def goal_transition(goal_id: str, body: dict):
        """Transition a goal to a specific state."""
        to_status_str = body.get("status", "")
        try:
            to_status = GoalStatus(to_status_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown status: {to_status_str}")
        try:
            session = _tracker.transition(goal_id, to_status, error=body.get("error"))
            return {
                "goal_id": session.goal_id,
                "status": session.status.value,
                "progress_pct": session.progress_pct,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # POST /api/cortex/goals/session/{goal_id}/advance
    @r.post("/goals/session/{goal_id}/advance")
    async def goal_advance(goal_id: str):
        """Advance a goal to the next logical state."""
        try:
            session = _tracker.advance(goal_id)
            return {
                "goal_id": session.goal_id,
                "status": session.status.value,
                "progress_pct": session.progress_pct,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # POST /api/cortex/goals/session/{goal_id}/fail
    @r.post("/goals/session/{goal_id}/fail")
    async def goal_fail(goal_id: str, body: dict):
        """Mark a goal as failed."""
        try:
            session = _tracker.fail(goal_id, body.get("error", "Unknown error"))
            return {"goal_id": session.goal_id, "status": "failed"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # POST /api/cortex/goals/session/{goal_id}/complete
    @r.post("/goals/session/{goal_id}/complete")
    async def goal_complete(goal_id: str):
        """Mark a goal as completed."""
        try:
            session = _tracker.complete(goal_id)
            return {"goal_id": session.goal_id, "status": "completed"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # POST /api/cortex/goals/session/{goal_id}/retry
    @r.post("/goals/session/{goal_id}/retry")
    async def goal_retry(goal_id: str):
        """Retry a failed goal from scratch."""
        try:
            session = _tracker.retry(goal_id)
            return {"goal_id": session.goal_id, "status": session.status.value}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # PUT /api/cortex/goals/session/{goal_id}/progress
    @r.put("/goals/session/{goal_id}/progress")
    async def goal_update_progress(goal_id: str, body: dict):
        """Update progress and task counts for a goal."""
        try:
            session = _tracker.update_progress(
                goal_id,
                progress=body.get("progress", 0.0),
                task_count=body.get("task_count"),
                tasks_completed=body.get("tasks_completed"),
            )
            return {
                "goal_id": session.goal_id,
                "progress_pct": session.progress_pct,
                "task_count": session.task_count,
                "tasks_completed": session.tasks_completed,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # PUT /api/cortex/goals/session/{goal_id}/context
    @r.put("/goals/session/{goal_id}/context")
    async def goal_update_context(goal_id: str, body: dict):
        """Update context on a goal session."""
        try:
            session = _tracker.update_context(goal_id, **body)
            return {"goal_id": session.goal_id, "context": session.context}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ==================================================================
    # CHECKPOINTS (Fase 4)
    # ==================================================================

    # POST /api/cortex/goals/session/{goal_id}/checkpoint
    @r.post("/goals/session/{goal_id}/checkpoint")
    async def goal_create_checkpoint(goal_id: str, body: dict):
        """Create a checkpoint for a goal."""
        try:
            cp = _tracker.checkpoint(
                goal_id,
                context=body.get("context"),
                outputs=body.get("outputs"),
            )
            return {
                "checkpoint_id": cp.id,
                "state": cp.state.value,
                "progress": cp.progress,
                "timestamp": cp.timestamp,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # GET /api/cortex/goals/session/{goal_id}/checkpoints
    @r.get("/goals/session/{goal_id}/checkpoints")
    async def goal_list_checkpoints(goal_id: str):
        """List all checkpoints for a goal."""
        try:
            checkpoints = _tracker.get_checkpoints(goal_id)
            return {
                "checkpoints": [
                    {
                        "id": cp.id,
                        "state": cp.state.value,
                        "progress": cp.progress,
                        "timestamp": cp.timestamp,
                    }
                    for cp in checkpoints
                ],
                "count": len(checkpoints),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # POST /api/cortex/goals/session/{goal_id}/checkpoint/{checkpoint_id}/restore
    @r.post("/goals/session/{goal_id}/checkpoint/{checkpoint_id}/restore")
    async def goal_restore_checkpoint(goal_id: str, checkpoint_id: str):
        """Restore a goal to a previous checkpoint."""
        try:
            session = _tracker.restore_checkpoint(goal_id, checkpoint_id)
            return {
                "goal_id": session.goal_id,
                "status": session.status.value,
                "progress_pct": session.progress_pct,
                "restored_to_checkpoint": checkpoint_id,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ==================================================================
    # SDD with LLM (Fase 4)
    # ==================================================================

    # POST /api/cortex/sdd/generate-with-llm
    @r.post("/sdd/generate-with-llm")
    async def sdd_generate_with_llm(body: dict):
        """Generate SDD documents using LLM when available."""
        goal_id = body.get("goal_id", "")
        goal = body.get("goal", "")
        if not goal_id or not goal:
            raise HTTPException(status_code=400, detail="goal_id and goal are required")
        try:
            docs = _sdd_gen.generate_all_with_llm(goal_id, goal)
            path = docs.save()
            return {
                "goal_id": goal_id,
                "documents_path": path,
                "llm_used": _sdd_gen._llm_available,
                "spec_preview": docs.spec_content[:300],
                "plan_preview": docs.plan_content[:300],
                "tasks_preview": docs.tasks_content[:300],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ==================================================================
    # CHAT INTEGRATION (Fase 6)
    # ==================================================================

    # POST /api/cortex/chat/route
    @r.post("/chat/route")
    async def chat_route_message(body: dict):
        """Route a chat message to the best model using SmartRouter."""
        from cortex.integration import route_chat_message
        message = body.get("message", "")
        if not message.strip():
            raise HTTPException(status_code=400, detail="message is required")
        result = route_chat_message(
            message,
            current_model=body.get("current_model", ""),
        )
        return result

    # POST /api/cortex/goals/schedule
    @r.post("/goals/schedule")
    async def goal_decompose_and_schedule(body: dict):
        """Decompose a goal and create ScheduledTasks."""
        from cortex.integration import decompose_and_schedule
        goal_id = body.get("goal_id", "")
        goal_text = body.get("goal", "")
        if not goal_id or not goal_text:
            raise HTTPException(status_code=400, detail="goal_id and goal are required")
        try:
            result = decompose_and_schedule(
                goal_id, goal_text,
                owner=body.get("owner", "admin"),
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # POST /api/cortex/sdd/generate-with-odysseus-llm
    @r.post("/sdd/generate-with-odysseus-llm")
    async def sdd_generate_with_odysseus_llm(body: dict):
        """Generate SDD docs using Odysseus LLM via integration layer."""
        from cortex.integration import generate_sdd_with_odysseus_llm
        goal_id = body.get("goal_id", "")
        goal_text = body.get("goal", "")
        if not goal_id or not goal_text:
            raise HTTPException(status_code=400, detail="goal_id and goal are required")
        result = generate_sdd_with_odysseus_llm(goal_id, goal_text)
        return result

    # POST /api/cortex/heartbeat/check
    @r.post("/heartbeat/check")
    async def heartbeat_run_check():
        """Run heartbeat health check and notify via WebSocket."""
        from cortex.integration import heartbeat_health_check
        result = await heartbeat_health_check()
        return result

    return r
