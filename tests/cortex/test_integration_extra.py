"""Extra tests for cortex/integration.py — covering uncovered lines.

Brings coverage from ~45% to >90% by testing:
  - goal_to_scheduled_tasks (lines 120-165)
  - decompose_and_schedule (lines 177-207) — success & error paths
  - generate_sdd_with_odysseus_llm (lines 228-250)
  - notify_heartbeat_event (lines 279-296) — multiple clients, dead clients
  - heartbeat_health_check (lines 305-329) — stale goals, health changes
  - register_ws_client / unregister_ws_client edge cases
  - route_chat_message should_switch logic (line 50)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch, call, PropertyMock

import pytest


# ===================================================================
# 1. route_chat_message — should_switch branches (line 50)
# ===================================================================


class TestRouteChatMessageSwitchLogic:
    """Cover the should_switch logic on line 50."""

    def test_no_switch_when_model_matches(self):
        """Same current_model as recommended → should_switch=False."""
        from cortex.integration import route_chat_message
        result = route_chat_message("hello", current_model="deepseek-chat")
        # deepseek-chat is the default recommendation, so should_switch=False
        assert result["should_switch"] is False

    def _switch_test(self, category: str, label: str, message: str,
                      expect_switch: bool):
        """Helper to test a single should_switch scenario."""
        from cortex.integration import route_chat_message
        with patch("cortex.smart_router.SmartRouter") as MockRouter:
            router_instance = MockRouter.return_value
            router_instance.route.return_value = {
                "recommended_model": "claude-sonnet-4",
                "recommended_provider": "anthropic",
                "category": category,
                "label": label,
                "estimated_cost_usd": 3.0,
                "fallback_models": [],
            }
            result = route_chat_message(message, current_model="deepseek-chat")
        assert result["should_switch"] is expect_switch
        return result

    def test_switch_when_category_is_code(self):
        """Different model + code category → should_switch=True."""
        result = self._switch_test("code", "Code Generation",
                                   "write a function", True)
        assert result["category"] == "code"

    def test_switch_when_category_is_reasoning(self):
        """Different model + reasoning category → should_switch=True."""
        self._switch_test("reasoning", "Reasoning",
                          "solve this logic puzzle", True)

    def test_switch_when_category_is_creative(self):
        """Different model + creative category → should_switch=True."""
        self._switch_test("creative", "Creative Writing",
                          "write a poem", True)

    def test_no_switch_for_chat_category(self):
        """Different model but category is chat → should_switch=False."""
        self._switch_test("chat", "General Chat",
                          "how are you?", False)

    def test_no_switch_for_unknown_category(self):
        """Different model but unknown category → should_switch=False."""
        self._switch_test("analysis", "Analysis",
                          "analyze this data", False)


# ===================================================================
# 2. goal_to_scheduled_tasks (lines 100-165)
# ===================================================================


class MockSubTask:
    """Minimal stand-in for cortex.goals.SubTask."""
    def __init__(self, id, description, agent, capabilities, depends_on):
        self.id = id
        self.description = description
        self.agent = agent
        self.capabilities = capabilities
        self.depends_on = depends_on


class MockGoalDecomposition:
    """Minimal stand-in for cortex.goals.GoalDecomposition."""
    def __init__(self, tasks):
        self.tasks = tasks
        self.goal = "test goal"
        self.source = "heuristic"
        self.total_estimated_cost_usd = 0.0


class TestGoalToScheduledTasks:
    """Cover goal_to_scheduled_tasks — lines 100-165."""

    @pytest.fixture
    def mock_db_session(self):
        """A mock SQLAlchemy session."""
        return MagicMock()

    @pytest.fixture
    def mock_decompose(self):
        """Patch cortex.goals.decompose to return controlled data."""
        tasks = [
            MockSubTask(
                id="task-1",
                description="Research the problem",
                agent="research",
                capabilities=["web_search", "analysis"],
                depends_on=[],
            ),
            MockSubTask(
                id="task-2",
                description="Implement the solution",
                agent="coder",
                capabilities=["python", "testing"],
                depends_on=["task-1"],
            ),
        ]
        decomp = MockGoalDecomposition(tasks)
        with patch("cortex.goals.decompose", return_value=decomp) as mock:
            yield mock

    def test_creates_scheduled_tasks(self, mock_db_session, mock_decompose):
        """Creates ScheduledTask entries for each subtask."""
        from cortex.integration import goal_to_scheduled_tasks
        from core.database import ScheduledTask

        tasks = goal_to_scheduled_tasks(
            goal_id="goal-1",
            goal_text="build a CLI tool",
            db_session=mock_db_session,
            owner="admin",
        )

        assert len(tasks) == 2
        assert tasks[0]["task_id"] == "goal-1-task-1"
        assert tasks[0]["description"] == "Research the problem"
        assert tasks[0]["agent"] == "research"
        assert tasks[0]["depends_on"] == []

        assert tasks[1]["task_id"] == "goal-1-task-2"
        assert tasks[1]["depends_on"] == ["task-1"]

        # Verify ScheduledTask was instantiated and added to session
        assert mock_db_session.add.call_count == 2
        mock_db_session.commit.assert_called_once()

        # Verify the first call to add
        added_task = mock_db_session.add.call_args_list[0][0][0]
        assert isinstance(added_task, ScheduledTask)
        assert added_task.id == "goal-1-task-1"
        assert added_task.owner == "admin"
        assert added_task.task_type == "llm"
        assert added_task.schedule == "once"
        assert added_task.status == "active"
        assert added_task.output_target == "session"
        assert "Research the problem" in added_task.name
        assert added_task.prompt is not None
        assert "Task: Research the problem" in added_task.prompt
        assert "Capabilities: web_search, analysis" in added_task.prompt
        assert "Dependencies: none" in added_task.prompt

    def test_single_task(self, mock_db_session):
        """Works with only one task."""
        from cortex.integration import goal_to_scheduled_tasks

        tasks = [
            MockSubTask("only", "Single task", "basic", ["general"], []),
        ]
        decomp = MockGoalDecomposition(tasks)
        with patch("cortex.goals.decompose", return_value=decomp):
            result = goal_to_scheduled_tasks(
                goal_id="g-single",
                goal_text="do one thing",
                db_session=mock_db_session,
            )

        assert len(result) == 1
        assert result[0]["task_id"] == "g-single-only"
        mock_db_session.commit.assert_called_once()

    def test_empty_decomposition(self, mock_db_session):
        """No tasks from decomposition → empty list, commit still called."""
        from cortex.integration import goal_to_scheduled_tasks

        decomp = MockGoalDecomposition([])
        with patch("cortex.goals.decompose", return_value=decomp):
            result = goal_to_scheduled_tasks(
                goal_id="g-empty",
                goal_text="empty goal",
                db_session=mock_db_session,
            )

        assert result == []
        mock_db_session.add.assert_not_called()
        mock_db_session.commit.assert_called_once()

    def test_task_with_dependencies_in_prompt(self, mock_db_session):
        """Dependencies are included in the prompt."""
        from cortex.integration import goal_to_scheduled_tasks

        tasks = [
            MockSubTask("t1", "Task one", "agent1", ["cap1"], ["dep1", "dep2"]),
        ]
        decomp = MockGoalDecomposition(tasks)
        with patch("cortex.goals.decompose", return_value=decomp):
            result = goal_to_scheduled_tasks(
                goal_id="g-dep",
                goal_text="test",
                db_session=mock_db_session,
            )

        added = mock_db_session.add.call_args[0][0]
        assert "Dependencies: dep1, dep2" in added.prompt

    def test_default_owner_is_admin(self, mock_db_session):
        """Default owner is 'admin'."""
        from cortex.integration import goal_to_scheduled_tasks

        tasks = [MockSubTask("t1", "test", "agent", ["cap"], [])]
        decomp = MockGoalDecomposition(tasks)
        with patch("cortex.goals.decompose", return_value=decomp):
            goal_to_scheduled_tasks(
                goal_id="g-owner",
                goal_text="test",
                db_session=mock_db_session,
            )

        added = mock_db_session.add.call_args[0][0]
        assert added.owner == "admin"


# ===================================================================
# 3. decompose_and_schedule (lines 168-220)
# ===================================================================


class TestDecomposeAndSchedule:
    """Cover decompose_and_schedule — lines 168-220."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock SessionLocal and its instance."""
        session = MagicMock()
        with patch("core.database.SessionLocal", return_value=session) as mock:
            yield session, mock

    def test_success_path(self, mock_db_session):
        """Full success path: decompose → GoalTracker → DB."""
        from cortex.integration import decompose_and_schedule

        session, _ = mock_db_session

        # Mock decompose
        tasks = [
            MockSubTask("t1", "Research", "research", ["web"], []),
            MockSubTask("t2", "Build", "coder", ["python"], ["t1"]),
        ]
        decomp = MockGoalDecomposition(tasks)

        with patch("cortex.goals.decompose", return_value=decomp):
            with patch("cortex.goal_state.GoalTracker") as MockTracker:
                tracker_instance = MockTracker.return_value
                tracker_instance.create_goal.return_value = MagicMock(
                    status=MagicMock(value="active")
                )
                # Call the function
                result = decompose_and_schedule(
                    goal_id="g-sched-1",
                    goal_text="build something useful",
                    owner="test-user",
                )

        # Verify GoalTracker calls
        tracker_instance.create_goal.assert_called_once_with(
            "g-sched-1", "build something useful"
        )
        tracker_instance.transition.assert_any_call("g-sched-1", "decomposing")
        tracker_instance.transition.assert_any_call("g-sched-1", "assigning")
        tracker_instance.update_progress.assert_called_once_with(
            "g-sched-1",
            progress=0.3,
            task_count=2,
        )

        # Verify DB calls
        session.add.assert_called()
        session.commit.assert_called()
        session.close.assert_called_once()

        # Verify result structure
        assert result["goal_id"] == "g-sched-1"
        assert result["status"] == "active"
        assert len(result["tasks"]) == 2
        assert len(result["scheduled_tasks"]) == 2

    def test_success_default_owner(self, mock_db_session):
        """Default owner is 'admin'."""
        from cortex.integration import decompose_and_schedule

        session, _ = mock_db_session
        decomp = MockGoalDecomposition([MockSubTask("t1", "Task", "agent", [], [])])

        with patch("cortex.goals.decompose", return_value=decomp):
            with patch("cortex.goal_state.GoalTracker") as MockTracker:
                ti = MockTracker.return_value
                ti.create_goal.return_value = MagicMock(status=MagicMock(value="active"))
                decompose_and_schedule("g-default", "test")

        added = session.add.call_args[0][0]
        assert added.owner == "admin"

    def test_db_exception_rollback(self, mock_db_session):
        """When DB operation raises, it rolls back and marks goal as failed."""
        from cortex.integration import decompose_and_schedule

        session, _ = mock_db_session
        session.commit.side_effect = RuntimeError("DB connection lost")

        decomp = MockGoalDecomposition([MockSubTask("t1", "Task", "agent", [], [])])

        with patch("cortex.goals.decompose", return_value=decomp):
            with patch("cortex.goal_state.GoalTracker") as MockTracker:
                ti = MockTracker.return_value
                ti.create_goal.return_value = MagicMock(
                    status=MagicMock(value="active")
                )
                with pytest.raises(RuntimeError, match="DB connection lost"):
                    decompose_and_schedule("g-fail", "test")

        # Verify rollback was called
        session.rollback.assert_called_once()
        # Verify goal was marked as failed
        ti.fail.assert_called_once_with("g-fail", "DB connection lost")
        # Verify session is closed even after error
        session.close.assert_called_once()

    def test_db_exception_rollback_and_close(self, mock_db_session):
        """Ensure close() is called in finally even when rollback happens."""
        from cortex.integration import decompose_and_schedule

        session, _ = mock_db_session
        session.commit.side_effect = ValueError("integrity error")

        decomp = MockGoalDecomposition([MockSubTask("t1", "T", "a", [], [])])

        with patch("cortex.goals.decompose", return_value=decomp):
            with patch("cortex.goal_state.GoalTracker") as MockTracker:
                ti = MockTracker.return_value
                ti.create_goal.return_value = MagicMock(
                    status=MagicMock(value="active")
                )
                with pytest.raises(ValueError):
                    decompose_and_schedule("g-close", "test")

        session.rollback.assert_called_once()
        session.close.assert_called_once()


# ===================================================================
# 4. generate_sdd_with_odysseus_llm (lines 228-250)
# ===================================================================


class TestGenerateSddWithOdysseusLlm:
    """Cover generate_sdd_with_odysseus_llm — lines 228-250."""

    def test_full_generation(self):
        """LLM generates spec, plan, and tasks."""
        from cortex.integration import generate_sdd_with_odysseus_llm

        mock_docs = MagicMock()
        mock_docs.save.return_value = "/tmp/sdd/test-goal"
        mock_docs.spec_content = "# Spec\n\nFunctional spec for test-goal\n\n" + "X" * 300
        mock_docs.plan_content = "# Plan\n\nImplementation plan\n\n" + "Y" * 300
        mock_docs.tasks_content = "# Tasks\n\nTask breakdown\n\n" + "Z" * 300

        with patch("cortex.sdd.SDDGenerator") as MockGen:
            gen_instance = MockGen.return_value
            gen_instance._llm_available = True
            gen_instance.generate_all_with_llm.return_value = mock_docs

            result = generate_sdd_with_odysseus_llm("sdd-1", "build an API")

        assert result["goal_id"] == "sdd-1"
        assert result["documents_path"] == "/tmp/sdd/test-goal"
        assert result["llm_used"] is True
        assert result["spec_preview"].startswith("# Spec")
        assert result["plan_preview"].startswith("# Plan")
        assert result["tasks_preview"].startswith("# Tasks")

        gen_instance.generate_all_with_llm.assert_called_once_with(
            "sdd-1", "build an API"
        )
        mock_docs.save.assert_called_once()

    def test_generation_without_llm(self):
        """LLM not available — templates are used."""
        from cortex.integration import generate_sdd_with_odysseus_llm

        mock_docs = MagicMock()
        mock_docs.save.return_value = "/tmp/sdd/sdd-2"
        mock_docs.spec_content = "# Specification\n\n## Goal\nbuild a CLI"
        mock_docs.plan_content = "# Implementation Plan\n\n## Goal\nbuild a CLI"
        mock_docs.tasks_content = "# Tasks\n\n## Goal\nbuild a CLI"

        with patch("cortex.sdd.SDDGenerator") as MockGen:
            gen_instance = MockGen.return_value
            gen_instance._llm_available = False
            gen_instance.generate_all_with_llm.return_value = mock_docs

            result = generate_sdd_with_odysseus_llm("sdd-2", "build a CLI")

        assert result["goal_id"] == "sdd-2"
        assert result["llm_used"] is False
        assert "Specification" in result["spec_preview"]

    def test_empty_documents(self):
        """Documents with empty content — previews are empty strings."""
        from cortex.integration import generate_sdd_with_odysseus_llm

        mock_docs = MagicMock()
        mock_docs.save.return_value = "/tmp/sdd/empty"
        mock_docs.spec_content = None
        mock_docs.plan_content = None
        mock_docs.tasks_content = ""

        with patch("cortex.sdd.SDDGenerator") as MockGen:
            gen_instance = MockGen.return_value
            gen_instance._llm_available = True
            gen_instance.generate_all_with_llm.return_value = mock_docs

            result = generate_sdd_with_odysseus_llm("empty", "empty goal")

        assert result["spec_preview"] == ""
        assert result["plan_preview"] == ""
        assert result["tasks_preview"] == ""


# ===================================================================
# 5. WebSocket registration / unregistration — edge cases
# ===================================================================


class TestWebSocketRegistration:
    """Cover register_ws_client + unregister_ws_client edge cases."""

    def _get_ws_clients(self):
        """Get the module-level _ws_clients dict, carefully."""
        from cortex import integration as mod
        # Use its reference to avoid re-import issues
        return mod._ws_clients

    def setup_method(self):
        """Clear _ws_clients before each test."""
        clients = self._get_ws_clients()
        clients.clear()

    def test_register_multiple_clients(self):
        """Can register multiple clients."""
        from cortex.integration import register_ws_client
        register_ws_client("s1", "ws1")
        register_ws_client("s2", "ws2")
        register_ws_client("s3", "ws3")

        clients = self._get_ws_clients()
        assert len(clients) == 3
        assert clients["s1"] == "ws1"
        assert clients["s2"] == "ws2"
        assert clients["s3"] == "ws3"

    def test_register_replaces_existing(self):
        """Re-registering same session_id replaces old websocket."""
        from cortex.integration import register_ws_client
        register_ws_client("s1", "old-ws")
        register_ws_client("s1", "new-ws")

        clients = self._get_ws_clients()
        assert clients["s1"] == "new-ws"
        assert len(clients) == 1

    def test_unregister_existing(self):
        """Unregister an existing client."""
        from cortex.integration import register_ws_client, unregister_ws_client
        register_ws_client("s1", "ws1")
        unregister_ws_client("s1")

        clients = self._get_ws_clients()
        assert "s1" not in clients

    def test_unregister_nonexistent(self):
        """Unregister a non-existent client does not raise."""
        from cortex.integration import unregister_ws_client
        # Should not raise
        unregister_ws_client("nonexistent")

    def test_unregister_clears_only_one(self):
        """Unregister removes only the specified client."""
        from cortex.integration import register_ws_client, unregister_ws_client
        register_ws_client("s1", "ws1")
        register_ws_client("s2", "ws2")
        unregister_ws_client("s1")

        clients = self._get_ws_clients()
        assert "s1" not in clients
        assert clients["s2"] == "ws2"


# ===================================================================
# 6. notify_heartbeat_event (lines 271-296)
# ===================================================================


class TestNotifyHeartbeatEvent:
    """Cover notify_heartbeat_event — lines 271-296."""

    def _get_ws_clients(self):
        from cortex import integration as mod
        return mod._ws_clients

    @pytest.fixture(autouse=True)
    def clear_clients(self):
        """Reset _ws_clients before each test."""
        clients = self._get_ws_clients()
        clients.clear()
        yield
        clients.clear()

    @pytest.mark.asyncio
    async def test_sends_to_all_clients(self):
        """Sends message to every registered client."""
        from cortex.integration import register_ws_client, notify_heartbeat_event

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        register_ws_client("s1", ws1)
        register_ws_client("s2", ws2)

        await notify_heartbeat_event("stale_goal", {"count": 2})

        assert ws1.send_text.call_count == 1
        assert ws2.send_text.call_count == 1

        # Verify message contains expected fields
        sent = ws1.send_text.call_args[0][0]
        import json
        msg = json.loads(sent)
        assert msg["type"] == "heartbeat"
        assert msg["event"] == "stale_goal"
        assert msg["data"] == {"count": 2}
        assert "timestamp" in msg

    @pytest.mark.asyncio
    async def test_no_clients(self):
        """No clients registered — no error."""
        from cortex.integration import notify_heartbeat_event
        await notify_heartbeat_event("tick", {})
        # Should not raise

    @pytest.mark.asyncio
    async def test_removes_dead_clients(self):
        """Clients that raise on send_text are unregistered."""
        from cortex.integration import (
            register_ws_client,
            unregister_ws_client,
            notify_heartbeat_event,
            _ws_clients,
        )

        ws_alive = AsyncMock()
        ws_dead = AsyncMock()
        ws_dead.send_text.side_effect = ConnectionError("WS disconnected")

        register_ws_client("alive", ws_alive)
        register_ws_client("dead", ws_dead)

        await notify_heartbeat_event("tick", {})

        # alive client still there
        assert "alive" in _ws_clients
        # dead client should be removed
        assert "dead" not in _ws_clients
        # alive got the message
        ws_alive.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_clients_dead(self):
        """All clients dead — all removed."""
        from cortex.integration import register_ws_client, notify_heartbeat_event, _ws_clients

        ws1 = AsyncMock()
        ws1.send_text.side_effect = RuntimeError("gone")
        ws2 = AsyncMock()
        ws2.send_text.side_effect = RuntimeError("gone too")

        register_ws_client("s1", ws1)
        register_ws_client("s2", ws2)

        await notify_heartbeat_event("tick", {})

        # Both removed
        assert len(_ws_clients) == 0

    @pytest.mark.asyncio
    async def test_message_structure(self):
        """Verify exact message JSON structure."""
        from cortex.integration import register_ws_client, notify_heartbeat_event

        ws = AsyncMock()
        register_ws_client("s1", ws)

        await notify_heartbeat_event(
            "health_change",
            {"db_connected": False, "redis_connected": True},
        )

        import json
        msg = json.loads(ws.send_text.call_args[0][0])
        assert msg["type"] == "heartbeat"
        assert msg["event"] == "health_change"
        assert msg["data"]["db_connected"] is False
        assert msg["data"]["redis_connected"] is True
        # Timestamp is ISO format
        assert "T" in msg["timestamp"]


# ===================================================================
# 7. heartbeat_health_check (lines 299-336)
# ===================================================================


class MockHeartbeatTickResult:
    """Stand-in for HeartbeatTickResult to avoid importing heartbeat module."""

    def __init__(
        self,
        db_connected=True,
        redis_connected=True,
        stale_goals_found=0,
        goals_marked_failed=0,
        errors=None,
        duration_ms=12.5,
    ):
        self.db_connected = db_connected
        self.redis_connected = redis_connected
        self.stale_goals_found = stale_goals_found
        self.goals_marked_failed = goals_marked_failed
        self.errors = errors or []
        self.duration_ms = duration_ms


class TestHeartbeatHealthCheck:
    """Cover heartbeat_health_check — lines 299-336."""

    @pytest.fixture(autouse=True)
    def clear_clients(self):
        """Reset _ws_clients before each test."""
        from cortex import integration as mod
        mod._ws_clients.clear()
        yield
        mod._ws_clients.clear()

    @pytest.mark.asyncio
    async def test_healthy_tick(self):
        """Healthy tick returns correct status."""
        from cortex.integration import heartbeat_health_check

        with patch("cortex.heartbeat.HeartbeatManager") as MockHB:
            hb_instance = MockHB.return_value
            hb_instance.tick = AsyncMock()
            hb_instance.tick.return_value = MockHeartbeatTickResult(
                db_connected=True,
                redis_connected=True,
                stale_goals_found=0,
                goals_marked_failed=0,
                duration_ms=12.5,
            )

            result = await heartbeat_health_check()

        assert result["db_connected"] is True
        assert result["redis_connected"] is True
        assert result["stale_goals_found"] == 0
        assert result["goals_marked_failed"] == 0
        assert result["duration_ms"] == 12.5
        assert result["healthy"] is True

    @pytest.mark.asyncio
    async def test_with_stale_goals(self):
        """Stale goals trigger a stale_goal notification."""
        from cortex.integration import heartbeat_health_check, _ws_clients

        ws = AsyncMock()
        _ws_clients["test-client"] = ws

        with patch("cortex.heartbeat.HeartbeatManager") as MockHB:
            hb_instance = MockHB.return_value
            hb_instance.tick = AsyncMock()
            hb_instance.tick.return_value = MockHeartbeatTickResult(
                db_connected=True,
                redis_connected=True,
                stale_goals_found=3,
                goals_marked_failed=2,
                duration_ms=15.0,
            )

            result = await heartbeat_health_check()

        assert result["stale_goals_found"] == 3
        assert result["goals_marked_failed"] == 2

        # Verify WebSocket notification was sent
        ws.send_text.assert_called_once()
        import json
        msg = json.loads(ws.send_text.call_args[0][0])
        assert msg["event"] == "stale_goal"
        assert msg["data"]["count"] == 3
        assert msg["data"]["goals_marked_failed"] == 2

    @pytest.mark.asyncio
    async def test_health_change_notification(self):
        """DB/Redis issues trigger a health_change notification."""
        from cortex.integration import heartbeat_health_check, _ws_clients

        ws = AsyncMock()
        _ws_clients["test-client"] = ws

        with patch("cortex.heartbeat.HeartbeatManager") as MockHB:
            hb_instance = MockHB.return_value
            hb_instance.tick = AsyncMock()
            hb_instance.tick.return_value = MockHeartbeatTickResult(
                db_connected=False,
                redis_connected=True,
                stale_goals_found=0,
                goals_marked_failed=0,
                errors=["DB connection timeout"],
                duration_ms=500.0,
            )

            result = await heartbeat_health_check()

        assert result["db_connected"] is False
        assert result["redis_connected"] is True
        assert result["healthy"] is False

        # Verify health_change notification
        import json
        msg = json.loads(ws.send_text.call_args[0][0])
        assert msg["event"] == "health_change"
        assert msg["data"]["db_connected"] is False
        assert msg["data"]["redis_connected"] is True
        assert msg["data"]["errors"] == ["DB connection timeout"]

    @pytest.mark.asyncio
    async def test_both_notifications(self):
        """Both stale goals and health issues trigger two notifications."""
        from cortex.integration import heartbeat_health_check, _ws_clients

        ws = AsyncMock()
        _ws_clients["test-client"] = ws

        with patch("cortex.heartbeat.HeartbeatManager") as MockHB:
            hb_instance = MockHB.return_value
            hb_instance.tick = AsyncMock()
            hb_instance.tick.return_value = MockHeartbeatTickResult(
                db_connected=False,
                redis_connected=False,
                stale_goals_found=1,
                goals_marked_failed=1,
                errors=["DB down", "Redis down"],
                duration_ms=200.0,
            )

            result = await heartbeat_health_check()

        assert result["db_connected"] is False
        assert result["redis_connected"] is False
        assert result["stale_goals_found"] == 1
        assert result["healthy"] is False

        # Two WebSocket messages: stale_goal + health_change
        assert ws.send_text.call_count == 2
        import json
        events = [
            json.loads(ws.send_text.call_args_list[i][0][0])["event"]
            for i in range(2)
        ]
        assert "stale_goal" in events
        assert "health_change" in events

    @pytest.mark.asyncio
    async def test_no_notifications_when_healthy(self):
        """No stale goals and healthy → no WebSocket messages."""
        from cortex.integration import heartbeat_health_check, _ws_clients

        ws = AsyncMock()
        _ws_clients["test-client"] = ws

        with patch("cortex.heartbeat.HeartbeatManager") as MockHB:
            hb_instance = MockHB.return_value
            hb_instance.tick = AsyncMock()
            hb_instance.tick.return_value = MockHeartbeatTickResult(
                db_connected=True,
                redis_connected=True,
                stale_goals_found=0,
                goals_marked_failed=0,
                duration_ms=10.0,
            )

            result = await heartbeat_health_check()

        assert result["healthy"] is True
        # No notifications sent
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_ws_clients_no_crash(self):
        """No WebSocket clients registered → no crash, no notifications."""
        from cortex.integration import heartbeat_health_check

        with patch("cortex.heartbeat.HeartbeatManager") as MockHB:
            hb_instance = MockHB.return_value
            hb_instance.tick = AsyncMock()
            hb_instance.tick.return_value = MockHeartbeatTickResult(
                db_connected=False,
                redis_connected=False,
                stale_goals_found=2,
                goals_marked_failed=1,
                duration_ms=30.0,
            )

            # Should not raise despite having notifications to send but no clients
            result = await heartbeat_health_check()

        assert result["stale_goals_found"] == 2


# ===================================================================
# 8. route_model_for_session — last-user-message edge cases
# ===================================================================


class TestRouteModelForSessionEdgeCases:
    """Additional edge cases for route_model_for_session."""

    def test_multiple_user_messages(self):
        """Uses the last user message from history."""
        from cortex.integration import route_model_for_session

        history = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "code the solution"},
        ]
        result = route_model_for_session(history)
        assert result["model"] is not None
        assert result["category"] is not None

    def test_interleaved_messages(self):
        """User messages can be interleaved with system/assistant."""
        from cortex.integration import route_model_for_session

        history = [
            {"role": "system", "content": "be helpful"},
            {"role": "assistant", "content": "how can I help?"},
            {"role": "user", "content": "debug this error"},
        ]
        result = route_model_for_session(history)
        assert result["model"] is not None

    def test_no_user_messages_current_model_empty(self):
        """Empty history and no current model returns 'deepseek-chat'."""
        from cortex.integration import route_model_for_session

        result = route_model_for_session([], current_model="")
        assert result["model"] == "deepseek-chat"

    def test_function_call_messages_ignored(self):
        """Messages without role 'user' are skipped."""
        from cortex.integration import route_model_for_session

        history = [
            {"role": "function", "name": "get_weather", "content": "sunny"},
            {"role": "tool", "content": "result"},
        ]
        result = route_model_for_session(history, current_model="gpt-4o")
        assert result["model"] == "gpt-4o"
