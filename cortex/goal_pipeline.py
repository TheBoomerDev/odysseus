"""
cortex/goal_pipeline.py — Multi-agent autonomous pipeline.

Takes a goal from the GoalTracker and runs it through:
  1. ANALYZE — SDDGenerator generates spec + plan + tasks
  2. ROUTE — SmartRouter + HermesAgentRegistry pick the best agent per task
  3. DISPATCH — Execute via CLI invoker or Hermes Agent
  4. VERIFY — Check results against criteria
  5. CONSOLIDATE — Update GoalSession with progress + checkpoints

All supported execution backends:
  - Hermes Agent profiles (hermes chat --query --quiet)
  - CLI agents (codex.js, claude, agy via cli_invoker)
  - Direct LLM (via SmartRouter recommendation)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cortex.goal_state import GoalSession, GoalStatus, GoalTracker
from cortex.sdd import SDDGenerator, SDDDocuments
from cortex.hermes_bridge import HermesResponse, chat_async
from cortex.hermes_agents import HermesAgentRegistry, get_registry
from cortex.cli_invoker import invoke
from cortex.smart_router import SmartRouter
from cortex.codegen.engine import CodegenEngine, create_codegen_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step types
# ---------------------------------------------------------------------------


@dataclass
class PipelineStep:
    """A single step in the goal pipeline."""

    id: str
    description: str
    backend: str  # "hermes", "cli", "llm"
    target: str = ""  # agent name, CLI name, or model
    prompt: str = ""
    timeout: int = 120
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | success | failed | skipped
    result: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    goal_id: str
    steps: List[PipelineStep] = field(default_factory=list)
    status: str = "pending"  # pending | running | success | failed
    started_at: float = 0.0
    completed_at: float = 0.0
    total_duration_ms: float = 0.0
    error: Optional[str] = None
    documents_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class GoalPipeline:
    """Autonomous pipeline executor for goals.

    Usage:
        pipeline = GoalPipeline()
        result = await pipeline.run("goal-123")
    """

    def __init__(
        self,
        tracker: Optional[GoalTracker] = None,
        sdd_generator: Optional[SDDGenerator] = None,
        smart_router: Optional[SmartRouter] = None,
        agent_registry: Optional[HermesAgentRegistry] = None,
        codegen_provider: str = "hermes",
        codegen_model: Optional[str] = None,
        codegen_api_key: Optional[str] = None,
        codegen_profile: str = "swarm1",
    ):
        self._tracker = tracker or GoalTracker()
        self._sdd = sdd_generator or SDDGenerator()
        self._router = smart_router or SmartRouter()
        self._agents = agent_registry or get_registry()
        self._agents.refresh()
        self._codegen_provider = codegen_provider
        self._codegen_model = codegen_model
        self._codegen_api_key = codegen_api_key
        self._codegen_profile = codegen_profile

    async def run(
        self,
        goal_id: str,
        goal: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full pipeline for a goal.

        Steps:
          1. SDD generation (spec + plan + tasks)
          2. Task routing via HermesAgentRegistry
          3. Dispatch each task to the best backend
          4. Verify and consolidate results
        """
        result = PipelineResult(goal_id=goal_id, started_at=time.time())

        try:
            # Phase 1: SDD
            await self._phase_sdd(goal_id, goal, result, context or {})

            # Phase 1b: Codegen (if goal is code_generation)
            goal_type = (context or {}).get("goal_type", "") or self._detect_goal_type(goal)
            if goal_type == "code_generation":
                await self._phase_codegen(goal_id, goal, result, context or {})
            else:
                # Phase 2: Route + Dispatch each task
                await self._phase_execute(goal_id, result)

            # Phase 3: Verify
            await self._phase_verify(goal_id, result)

            # Consolidate
            result.status = "success"

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            logger.error("Pipeline failed for goal %s: %s", goal_id, e)
        finally:
            result.completed_at = time.time()
            result.total_duration_ms = (
                result.completed_at - result.started_at
            ) * 1000

        return result

    async def _phase_sdd(
        self,
        goal_id: str,
        goal: str,
        result: PipelineResult,
        context: Dict[str, Any],
    ) -> None:
        """Generate SDD documents (spec + plan + tasks)."""
        if not goal:
            session = self._tracker.get_goal(goal_id)
            if session:
                goal = session.goal
            else:
                goal = goal_id

        step = PipelineStep(
            id=f"{goal_id}-sdd",
            description=f"Generate SDD for: {goal[:60]}",
            backend="llm",
            target="sdd",
            prompt=goal,
        )
        result.steps.append(step)

        try:
            output = self._sdd.generate_all(goal)
            result.documents_path = output.get("documents_path")

            step.status = "success"
            step.result = json.dumps(output, default=str)[:500]

            # Update tracker
            session = self._tracker.get_goal(goal_id)
            if session:
                self._tracker.transition(
                    goal_id, GoalStatus.DECOMPOSING
                )
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            raise

    async def _phase_execute(
        self,
        goal_id: str,
        result: PipelineResult,
    ) -> None:
        """Route tasks to agents and execute."""
        session = self._tracker.get_goal(goal_id)
        if not session:
            return

        # Get SDD tasks if available
        tasks_from_sdd = session.context.get("tasks", [])
        if not tasks_from_sdd:
            # Use a default task
            tasks_from_sdd = [
                {"id": f"{goal_id}-task-1", "description": goal_id, "agent": "auto"}
            ]

        for i, task_def in enumerate(tasks_from_sdd):
            task_id = task_def.get("id", f"{goal_id}-task-{i}")
            description = task_def.get("description", "Execute task")
            preferred_agent = task_def.get("agent", "auto")

            # Route task to best Hermes agent
            agent_name = await self._route_task(description, preferred_agent)

            step = PipelineStep(
                id=task_id,
                description=description[:80],
                backend="hermes" if agent_name else "cli",
                target=agent_name or preferred_agent,
                prompt=description,
            )
            result.steps.append(step)

            start = time.time()
            try:
                if agent_name:
                    # Dispatch via Hermes Agent
                    resp = await chat_async(
                        prompt=description,
                        profile=agent_name,
                        timeout=120,
                    )
                    step.duration_ms = (time.time() - start) * 1000

                    if resp.error:
                        step.status = "failed"
                        step.error = resp.error
                    else:
                        step.status = "success"
                        step.result = resp.content[:500]
                else:
                    # Fallback: use CLI agent
                    cli_result = invoke(
                        preferred_agent,
                        description,
                    )
                    step.duration_ms = (time.time() - start) * 1000
                    step.status = "success" if cli_result.exit_code == 0 else "failed"
                    step.result = cli_result.stdout[:500]
                    step.error = cli_result.stderr[:500] if cli_result.stderr else None

            except Exception as e:
                step.duration_ms = (time.time() - start) * 1000
                step.status = "failed"
                step.error = str(e)

    async def _route_task(
        self, description: str, preferred_agent: str
    ) -> Optional[str]:
        """Route a task to the best Hermes agent.

        Returns:
            Agent/profile name, or None if no suitable agent found.
        """
        # Use SmartRouter to categorize
        route = self._router.route(description)
        category = route.get("category", "chat")

        # Find best agent for this category
        best = self._agents.best_agent_for_category(category)

        if best:
            logger.info(
                "Routed task '%s' → agent '%s' (category: %s)",
                description[:40], best.name, category,
            )
            return best.name

        return preferred_agent if preferred_agent != "auto" else None

    @staticmethod
    def _detect_goal_type(goal: str) -> str:
        """Detect whether a goal is code_generation or general task execution."""
        code_keywords = [
            "implement", "build", "code", "develop", "create app",
            "create page", "landing page", "website", "web app",
            "frontend", "backend", "api", "component", "scaffold",
            "generate", "refactor", "migrate", "starter", "template",
        ]
        goal_lower = goal.lower()
        for kw in code_keywords:
            if kw in goal_lower:
                return "code_generation"
        return "general"

    async def _phase_codegen(
        self,
        goal_id: str,
        goal: str,
        result: PipelineResult,
        context: Dict[str, Any],
    ) -> None:
        """Execute code generation via CodegenEngine.

        Runs the two-stage pipeline (Plan → Tool Loop) and tracks
        progress in the GoalSession.
        """
        step = PipelineStep(
            id=f"{goal_id}-codegen",
            description=f"Code generation for: {goal[:60]}",
            backend=self._codegen_provider,
            target=self._codegen_model or self._codegen_profile,
            prompt=goal,
            timeout=600,
        )
        result.steps.append(step)

        # Create codegen engine
        engine = create_codegen_engine(
            provider=self._codegen_provider,
            model=self._codegen_model,
            api_key=self._codegen_api_key,
            profile=self._codegen_profile,
        )

        start = time.time()
        files_written: List[str] = []
        error: Optional[str] = None

        try:
            async for event in engine.run(
                project_id=goal_id,
                prompt=goal,
                starter_name="next",
                goal=goal,
            ):
                if event.type == "file_write":
                    files_written.append(event.path)
                elif event.type == "error":
                    error = event.message
                    if not getattr(event, "recoverable", True):
                        break
                elif event.type == "status":
                    logger.info("  [%s] %s: %s", goal_id, event.stage, event.note)

        except Exception as e:
            error = str(e)
            logger.exception("Codegen phase failed for %s", goal_id)

        step.duration_ms = (time.time() - start) * 1000

        if error:
            step.status = "failed"
            step.error = error
            step.result = f"Generated {len(files_written)} file(s) before error"
        else:
            step.status = "success"
            step.result = (
                f"Generated {len(files_written)} file(s) via "
                f"{self._codegen_provider}"
            )
            # Update tracker
            session = self._tracker.get_goal(goal_id)
            if session:
                self._tracker.update_progress(
                    goal_id,
                    progress=1.0,
                    task_count=1,
                    tasks_completed=1,
                    metadata={"files_written": files_written},
                )

    async def _phase_verify(
        self,
        goal_id: str,
        result: PipelineResult,
    ) -> None:
        """Verify pipeline results."""
        # Count successes
        success_count = sum(
            1 for s in result.steps if s.status == "success"
        )
        failed_count = sum(
            1 for s in result.steps if s.status == "failed"
        )

        session = self._tracker.get_goal(goal_id)
        if session:
            progress = success_count / max(len(result.steps), 1)
            self._tracker.update_progress(
                goal_id,
                progress=progress,
                task_count=len(result.steps),
                tasks_completed=success_count,
            )

            if failed_count == 0:
                self._tracker.transition(
                    goal_id, GoalStatus.VERIFYING
                )
            else:
                self._tracker.transition(
                    goal_id,
                    GoalStatus.FAILED,
                    error=f"{failed_count} of {len(result.steps)} steps failed",
                )

        logger.info(
            "Pipeline verify: %d success, %d failed / %d total",
            success_count,
            failed_count,
            len(result.steps),
        )
