"""
Tests for cortex/goal_pipeline.py — GoalPipeline orchestrator.

Achieves >90% coverage by mocking GoalTracker, SDDGenerator,
HermesAgentRegistry, SmartRouter, chat_async, and invoke.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from cortex.goal_pipeline import (
    GoalPipeline,
    PipelineResult,
    PipelineStep,
)
from cortex.goal_state import GoalSession, GoalStatus


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_tracker():
    """Mock GoalTracker."""
    tracker = MagicMock()
    return tracker


@pytest.fixture
def mock_sdd():
    """Mock SDDGenerator."""
    sdd = MagicMock()
    return sdd


@pytest.fixture
def mock_router():
    """Mock SmartRouter."""
    router = MagicMock()
    router.route.return_value = {"category": "code_generation"}
    return router


@pytest.fixture
def mock_agent():
    """A mock HermesAgent object returned by the registry."""
    agent = MagicMock()
    agent.name = "swarm1"
    agent.priority = 1
    return agent


@pytest.fixture
def mock_registry(mock_agent):
    """Mock HermesAgentRegistry."""
    registry = MagicMock()
    registry.best_agent_for_category.return_value = mock_agent
    return registry


@pytest.fixture
def mock_chat():
    """Mock chat_async."""
    with patch("cortex.goal_pipeline.chat_async", new_callable=AsyncMock) as m:
        m.return_value = MagicMock(
            content="Task completed successfully",
            error=None,
            exit_code=0,
        )
        yield m


@pytest.fixture
def mock_invoke():
    """Mock invoke (cli_invoker)."""
    with patch("cortex.goal_pipeline.invoke") as m:
        m.return_value = MagicMock(
            exit_code=0,
            stdout="CLI task output",
            stderr="",
        )
        yield m


@pytest.fixture
def happy_session():
    """A GoalSession in EXECUTING state with tasks in context."""
    return GoalSession(
        goal_id="test-goal-1",
        goal="Build a REST API",
        status=GoalStatus.EXECUTING,
        context={
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Implement auth endpoint",
                    "agent": "swarm1",
                },
            ]
        },
    )


@pytest.fixture
def pipeline(mock_tracker, mock_sdd, mock_router, mock_registry):
    """A GoalPipeline instance with all deps mocked."""
    pl = GoalPipeline(
        tracker=mock_tracker,
        sdd_generator=mock_sdd,
        smart_router=mock_router,
        agent_registry=mock_registry,
    )
    return pl


# =========================================================================
# __init__
# =========================================================================


class TestInit:
    def test_defaults_created(self):
        """When no args passed, GoalPipeline creates default instances."""
        with patch("cortex.goal_pipeline.GoalTracker") as GT, \
             patch("cortex.goal_pipeline.SDDGenerator") as SDD, \
             patch("cortex.goal_pipeline.SmartRouter") as SR, \
             patch("cortex.goal_pipeline.get_registry") as GR:
            pl = GoalPipeline()
            GT.assert_called_once()
            SDD.assert_called_once()
            SR.assert_called_once()
            GR.assert_called_once()
            GR.return_value.refresh.assert_called_once()

    def test_deps_passed_through(self, mock_tracker, mock_sdd, mock_router, mock_registry):
        """Provided dependencies are stored and registry is refreshed."""
        pl = GoalPipeline(
            tracker=mock_tracker,
            sdd_generator=mock_sdd,
            smart_router=mock_router,
            agent_registry=mock_registry,
        )
        assert pl._tracker is mock_tracker
        assert pl._sdd is mock_sdd
        assert pl._router is mock_router
        assert pl._agents is mock_registry
        mock_registry.refresh.assert_called_once()


# =========================================================================
# run() — full pipeline
# =========================================================================


class TestRun:
    @pytest.mark.asyncio
    async def test_run_valid_goal(
        self, pipeline, mock_tracker, mock_sdd, mock_router, mock_registry, mock_chat
    ):
        """Full happy path: SDD succeeds → execute succeeds → verify succeeds."""
        # Arrange
        sdd_output = {"documents_path": "/tmp/sdd/test-goal-1"}
        mock_sdd.generate_all.return_value = sdd_output
        mock_tracker.get_goal.return_value = GoalSession(
            goal_id="test-goal-1",
            goal="Build a REST API",
            status=GoalStatus.EXECUTING,
            context={
                "tasks": [
                    {
                        "id": "task-1",
                        "description": "Implement auth endpoint",
                        "agent": "swarm1",
                    },
                ]
            },
        )

        # Act
        result = await pipeline.run(goal_id="test-goal-1", goal="Build a REST API")

        # Assert
        assert result.status == "success"
        assert result.goal_id == "test-goal-1"
        assert len(result.steps) == 2  # SDD step + task step
        assert result.steps[0].status == "success"
        assert result.steps[1].status == "success"
        assert result.documents_path == "/tmp/sdd/test-goal-1"
        assert result.completed_at > 0
        assert result.total_duration_ms > 0

        # Verify phase calls
        mock_sdd.generate_all.assert_called_once_with("Build a REST API")
        mock_router.route.assert_called_once()
        mock_registry.best_agent_for_category.assert_called_once()
        mock_chat.assert_awaited_once()

        # Tracker was updated
        mock_tracker.transition.assert_any_call(
            "test-goal-1", GoalStatus.DECOMPOSING
        )
        mock_tracker.update_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_context_passed(
        self, pipeline, mock_tracker, mock_sdd, mock_chat
    ):
        """Context is passed through to _phase_sdd."""
        mock_sdd.generate_all.return_value = {"documents_path": "/tmp/sdd/goal-1"}
        mock_tracker.get_goal.return_value = GoalSession(
            goal_id="goal-1",
            goal="Test",
            status=GoalStatus.EXECUTING,
            context={"tasks": []},
        )

        result = await pipeline.run(
            goal_id="goal-1",
            goal="Test",
            context={"custom": "data"},
        )
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_run_sdd_failure(self, pipeline, mock_tracker, mock_sdd, mock_chat):
        """SDD phase failure sets result to failed."""
        mock_sdd.generate_all.side_effect = ValueError("SDD generation failed")
        mock_tracker.get_goal.return_value = GoalSession(
            goal_id="bad-goal", goal="Bad", status=GoalStatus.CREATED, context={}
        )

        result = await pipeline.run(goal_id="bad-goal", goal="Bad")

        assert result.status == "failed"
        assert "SDD generation failed" in result.error
        assert result.steps[0].status == "failed"
        assert result.steps[0].error == "SDD generation failed"

    @pytest.mark.asyncio
    async def test_run_execute_exception(
        self, pipeline, mock_tracker, mock_sdd, mock_chat
    ):
        """Unhandled exception during execute phase propagates to run()."""
        mock_sdd.generate_all.return_value = {"documents_path": "/tmp/sdd/ok"}
        # Make execute raise by having get_goal fail with an unexpected error
        mock_tracker.get_goal.side_effect = RuntimeError("Unexpected tracker error")

        result = await pipeline.run(goal_id="fail-goal", goal="Fail goal")

        assert result.status == "failed"
        assert "Unexpected tracker error" in result.error

    @pytest.mark.asyncio
    async def test_run_no_goal_string(self, pipeline, mock_tracker, mock_sdd, mock_chat):
        """When no goal string is given, _phase_sdd fetches it from the tracker."""
        session = GoalSession(
            goal_id="g-42",
            goal="From tracker session",
            status=GoalStatus.EXECUTING,
            context={},
        )
        mock_tracker.get_goal.return_value = session
        mock_sdd.generate_all.return_value = {"documents_path": "/tmp/sdd/g-42"}

        result = await pipeline.run(goal_id="g-42")

        # SDD should have been called with the goal from the session
        mock_sdd.generate_all.assert_called_once_with("From tracker session")
        assert result.status == "success"


# =========================================================================
# _phase_sdd
# =========================================================================


class TestPhaseSDD:
    @pytest.mark.asyncio
    async def test_sdd_with_goal(self, pipeline, mock_tracker, mock_sdd):
        """When goal is provided, it's passed directly to SDDGenerator."""
        mock_sdd.generate_all.return_value = {"documents_path": "/tmp/sdd/test-1"}
        mock_tracker.get_goal.return_value = GoalSession(
            goal_id="test-1", goal="Do something", status=GoalStatus.CREATED, context={}
        )

        result = PipelineResult(goal_id="test-1")
        await pipeline._phase_sdd("test-1", "Do something", result, {})
        mock_sdd.generate_all.assert_called_once_with("Do something")
        assert result.steps[0].status == "success"
        assert result.documents_path == "/tmp/sdd/test-1"

    @pytest.mark.asyncio
    async def test_sdd_without_goal_uses_session(self, pipeline, mock_tracker, mock_sdd):
        """When goal is empty, fetch from tracker session."""
        session = GoalSession(
            goal_id="g-1", goal="Session goal", status=GoalStatus.CREATED, context={}
        )
        mock_tracker.get_goal.return_value = session
        mock_sdd.generate_all.return_value = {"documents_path": "/tmp/sdd/g-1"}

        result = PipelineResult(goal_id="g-1")
        await pipeline._phase_sdd("g-1", "", result, {})
        mock_sdd.generate_all.assert_called_once_with("Session goal")

    @pytest.mark.asyncio
    async def test_sdd_without_goal_no_session(self, pipeline, mock_tracker, mock_sdd):
        """When goal is empty and no session, use goal_id as fallback."""
        mock_tracker.get_goal.return_value = None
        mock_sdd.generate_all.return_value = {"documents_path": "/tmp/sdd/g-99"}

        result = PipelineResult(goal_id="g-99")
        await pipeline._phase_sdd("g-99", "", result, {})
        mock_sdd.generate_all.assert_called_once_with("g-99")

    @pytest.mark.asyncio
    async def test_sdd_failure(self, pipeline, mock_tracker, mock_sdd):
        """When SDD generation raises, step is failed and exception re-raised."""
        mock_sdd.generate_all.side_effect = RuntimeError("LLM error")

        result = PipelineResult(goal_id="bad")
        with pytest.raises(RuntimeError, match="LLM error"):
            await pipeline._phase_sdd("bad", "Goal", result, {})

        assert result.steps[0].status == "failed"
        assert result.steps[0].error == "LLM error"

    @pytest.mark.asyncio
    async def test_sdd_skips_tracker_when_no_session(self, pipeline, mock_tracker, mock_sdd):
        """_phase_sdd does not call transition if get_goal returns None."""
        mock_tracker.get_goal.return_value = None
        mock_sdd.generate_all.return_value = {"documents_path": "/tmp/sdd/nope"}

        result = PipelineResult(goal_id="nope")
        await pipeline._phase_sdd("nope", "Do it", result, {})
        # transition should not have been called
        for call in mock_tracker.transition.mock_calls:
            pytest.fail(f"transition should not have been called but got: {call}")


# =========================================================================
# _route_task
# =========================================================================


class TestRouteTask:
    @pytest.mark.asyncio
    async def test_route_with_agent_found(self, pipeline, mock_router, mock_registry, mock_agent):
        """When best_agent_for_category returns an agent, return its name."""
        agent_name = await pipeline._route_task("Implement login", "auto")
        assert agent_name == "swarm1"
        mock_router.route.assert_called_once_with("Implement login")
        mock_registry.best_agent_for_category.assert_called_once_with("code_generation")

    @pytest.mark.asyncio
    async def test_route_no_agent_found_non_auto(self, pipeline, mock_router, mock_registry):
        """When no best agent and preferred_agent != 'auto', return preferred_agent."""
        mock_registry.best_agent_for_category.return_value = None
        agent_name = await pipeline._route_task("Write docs", "codex")
        assert agent_name == "codex"

    @pytest.mark.asyncio
    async def test_route_no_agent_found_auto(self, pipeline, mock_router, mock_registry):
        """When no best agent and preferred_agent == 'auto', return None."""
        mock_registry.best_agent_for_category.return_value = None
        agent_name = await pipeline._route_task("Chat", "auto")
        assert agent_name is None


# =========================================================================
# _phase_execute
# =========================================================================


class TestPhaseExecute:
    @pytest.mark.asyncio
    async def test_no_session_returns_early(self, pipeline, mock_tracker):
        """When tracker returns None, _phase_execute returns without adding steps."""
        mock_tracker.get_goal.return_value = None
        result = PipelineResult(goal_id="ghost")
        await pipeline._phase_execute("ghost", result)
        assert len(result.steps) == 0

    @pytest.mark.asyncio
    async def test_with_sdd_tasks(
        self, pipeline, mock_tracker, mock_router, mock_registry, mock_agent, mock_chat
    ):
        """Uses tasks from session.context when available."""
        session = GoalSession(
            goal_id="multi-task",
            goal="Multi",
            status=GoalStatus.EXECUTING,
            context={
                "tasks": [
                    {"id": "t1", "description": "Task 1", "agent": "auto"},
                    {"id": "t2", "description": "Task 2", "agent": "swarm4"},
                ]
            },
        )
        mock_tracker.get_goal.return_value = session
        mock_registry.best_agent_for_category.return_value = mock_agent

        result = PipelineResult(goal_id="multi-task")
        await pipeline._phase_execute("multi-task", result)
        assert len(result.steps) == 2
        assert result.steps[0].status == "success"
        assert result.steps[1].status == "success"
        assert mock_chat.await_count == 2

    @pytest.mark.asyncio
    async def test_default_task_when_no_tasks(
        self, pipeline, mock_tracker, mock_router, mock_registry, mock_agent, mock_chat
    ):
        """When no tasks in context, a default task is created."""
        session = GoalSession(
            goal_id="default-task",
            goal="Default",
            status=GoalStatus.EXECUTING,
            context={},  # no tasks key
        )
        mock_tracker.get_goal.return_value = session
        mock_registry.best_agent_for_category.return_value = mock_agent

        result = PipelineResult(goal_id="default-task")
        await pipeline._phase_execute("default-task", result)
        assert len(result.steps) == 1
        # Default task uses goal_id as description
        assert result.steps[0].id == "default-task-task-1"

    @pytest.mark.asyncio
    async def test_hermes_agent_error(
        self, pipeline, mock_tracker, mock_registry, mock_agent, mock_chat
    ):
        """When chat_async returns error, step is marked failed."""
        session = GoalSession(
            goal_id="hermes-err",
            goal="Err",
            status=GoalStatus.EXECUTING,
            context={
                "tasks": [{"id": "t1", "description": "Do it", "agent": "swarm1"}]
            },
        )
        mock_tracker.get_goal.return_value = session
        mock_registry.best_agent_for_category.return_value = mock_agent
        mock_chat.return_value = MagicMock(
            content="", error="Model overloaded", exit_code=1
        )

        result = PipelineResult(goal_id="hermes-err")
        await pipeline._phase_execute("hermes-err", result)
        assert result.steps[0].status == "failed"
        assert result.steps[0].error == "Model overloaded"

    @pytest.mark.asyncio
    async def test_cli_fallback(
        self, pipeline, mock_tracker, mock_router, mock_registry, mock_invoke
    ):
        """When no Hermes agent found and preferred_agent is 'auto', fall back to CLI invoke."""
        session = GoalSession(
            goal_id="cli-task",
            goal="CLI",
            status=GoalStatus.EXECUTING,
            context={
                "tasks": [{"id": "t1", "description": "Run analysis", "agent": "auto"}]
            },
        )
        mock_tracker.get_goal.return_value = session
        mock_registry.best_agent_for_category.return_value = None  # No hermes agent

        result = PipelineResult(goal_id="cli-task")
        await pipeline._phase_execute("cli-task", result)
        assert len(result.steps) == 1
        assert result.steps[0].backend == "cli"
        assert result.steps[0].target == "auto"
        mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_cli_fallback_failure(
        self, pipeline, mock_tracker, mock_registry, mock_invoke
    ):
        """When CLI invoke returns non-zero exit code, step is failed."""
        session = GoalSession(
            goal_id="cli-fail",
            goal="CLI fail",
            status=GoalStatus.EXECUTING,
            context={
                "tasks": [{"id": "t1", "description": "Crash", "agent": "auto"}]
            },
        )
        mock_tracker.get_goal.return_value = session
        mock_registry.best_agent_for_category.return_value = None
        mock_invoke.return_value = MagicMock(
            exit_code=1, stdout="", stderr="Command not found"
        )

        result = PipelineResult(goal_id="cli-fail")
        await pipeline._phase_execute("cli-fail", result)
        assert result.steps[0].status == "failed"
        assert result.steps[0].error == "Command not found"

    @pytest.mark.asyncio
    async def test_dispatch_exception(
        self, pipeline, mock_tracker, mock_registry, mock_agent, mock_chat
    ):
        """Exception during dispatch is caught and step marked failed."""
        session = GoalSession(
            goal_id="exc",
            goal="Exc",
            status=GoalStatus.EXECUTING,
            context={
                "tasks": [{"id": "t1", "description": "Boom", "agent": "swarm1"}]
            },
        )
        mock_tracker.get_goal.return_value = session
        mock_registry.best_agent_for_category.return_value = mock_agent
        mock_chat.side_effect = ConnectionError("Network down")

        result = PipelineResult(goal_id="exc")
        await pipeline._phase_execute("exc", result)
        assert result.steps[0].status == "failed"
        assert result.steps[0].error == "Network down"
        assert result.steps[0].duration_ms > 0


# =========================================================================
# _phase_verify
# =========================================================================


class TestPhaseVerify:
    @pytest.mark.asyncio
    async def test_all_success(self, pipeline, mock_tracker):
        """When all steps succeed, update_progress and transition to VERIFYING."""
        session = GoalSession(
            goal_id="v-ok", goal="OK", status=GoalStatus.EXECUTING, context={}
        )
        mock_tracker.get_goal.return_value = session

        result = PipelineResult(goal_id="v-ok")
        result.steps = [
            PipelineStep(id="s1", description="Step 1", backend="llm", status="success"),
            PipelineStep(id="s2", description="Step 2", backend="llm", status="success"),
        ]
        await pipeline._phase_verify("v-ok", result)

        mock_tracker.update_progress.assert_called_once_with(
            "v-ok",
            progress=1.0,
            task_count=2,
            tasks_completed=2,
        )
        mock_tracker.transition.assert_called_once_with(
            "v-ok", GoalStatus.VERIFYING
        )

    @pytest.mark.asyncio
    async def test_some_failed(self, pipeline, mock_tracker):
        """When some steps fail, transition to FAILED with error message."""
        session = GoalSession(
            goal_id="v-fail", goal="Fail", status=GoalStatus.EXECUTING, context={}
        )
        mock_tracker.get_goal.return_value = session

        result = PipelineResult(goal_id="v-fail")
        result.steps = [
            PipelineStep(id="s1", description="Step 1", backend="llm", status="success"),
            PipelineStep(id="s2", description="Step 2", backend="llm", status="failed"),
            PipelineStep(id="s3", description="Step 3", backend="llm", status="failed"),
        ]
        await pipeline._phase_verify("v-fail", result)

        mock_tracker.update_progress.assert_called_once_with(
            "v-fail",
            progress=1 / 3,
            task_count=3,
            tasks_completed=1,
        )
        mock_tracker.transition.assert_called_once_with(
            "v-fail", GoalStatus.FAILED, error="2 of 3 steps failed"
        )

    @pytest.mark.asyncio
    async def test_no_session(self, pipeline, mock_tracker):
        """When no session found, update_progress not called."""
        mock_tracker.get_goal.return_value = None

        result = PipelineResult(goal_id="ghost")
        result.steps = [
            PipelineStep(id="s1", description="S1", backend="llm", status="success"),
        ]
        await pipeline._phase_verify("ghost", result)

        mock_tracker.update_progress.assert_not_called()
        mock_tracker.transition.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_steps(self, pipeline, mock_tracker):
        """When result has no steps, progress is 0."""
        session = GoalSession(
            goal_id="empty", goal="Empty", status=GoalStatus.CREATED, context={}
        )
        mock_tracker.get_goal.return_value = session

        result = PipelineResult(goal_id="empty")
        result.steps = []
        await pipeline._phase_verify("empty", result)

        mock_tracker.update_progress.assert_called_once_with(
            "empty", progress=0.0, task_count=0, tasks_completed=0
        )
        # No failures → all good → transition to VERIFYING
        mock_tracker.transition.assert_called_once_with(
            "empty", GoalStatus.VERIFYING
        )


# =========================================================================
# Edge cases: PipelineStep / PipelineResult dataclasses
# =========================================================================


class TestDataClasses:
    def test_pipeline_step_defaults(self):
        """PipelineStep has correct default values."""
        step = PipelineStep(id="step-1", description="test", backend="hermes")
        assert step.status == "pending"
        assert step.backend == "hermes"
        assert step.result == ""
        assert step.error is None
        assert step.duration_ms == 0.0
        assert step.depends_on == []

    def test_pipeline_result_defaults(self):
        """PipelineResult has correct default values."""
        result = PipelineResult(goal_id="g-1")
        assert result.status == "pending"
        assert result.steps == []
        assert result.error is None
        assert result.documents_path is None
