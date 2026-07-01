"""Comprehensive tests for cortex/codegen/engine.py — CodegenEngine pipeline.

Targets >90% coverage. All LLM calls are mocked. Tests cover:
  - CodegenConfig defaults and custom config
  - create_codegen_engine factory
  - _build_openai_tools constant
  - _ensure_project scaffolding and storage integration
  - _is_project_empty detection
  - _select_family provider mapping
  - _load_project_context
  - _generate_plan (success and failure)
  - _build_tool_messages content
  - run() full pipeline with event sequence validation
  - _tool_loop with read_file, write_patch, shell_exec, unknown tool, LLM errors,
    max iterations, text-based vs native function calling
  - run_and_collect convenience wrapper
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from cortex.codegen.engine import (
    CodegenConfig,
    CodegenEngine,
    CodegenEvent,
    DEFAULT_MAX_TOOL_ITERATIONS,
    StatusEvent,
    PlanEvent,
    ToolCallEvent,
    ToolResultEvent,
    FileWriteEvent,
    PermissionRequestEvent,
    ErrorEvent,
    DoneEvent,
    create_codegen_engine,
    _build_openai_tools,
    _OPENAI_TOOLS,
)
from cortex.codegen.llm import LLMResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Return an AsyncMock LLM client that yields a default 'ok' response."""
    client = AsyncMock()
    client.provider = "hermes"
    client.model = "hermes/swarm1"
    client.chat = AsyncMock(
        return_value=LLMResult(
            content="Here is the plan.",
            tool_calls=[],
        )
    )
    return client


@pytest.fixture
def mock_get_llm_client(mock_llm_client):
    """Patch get_llm_client to return our mock."""
    with patch("cortex.codegen.engine.get_llm_client", return_value=mock_llm_client) as m:
        yield m


@pytest.fixture
def engine(mock_get_llm_client):
    """Return a basic CodegenEngine with mocked LLM client."""
    return CodegenEngine(provider="hermes")


@pytest.fixture
def engine_openai(mock_get_llm_client):
    """Return an engine using openai provider (native function calling mode)."""
    eng = CodegenEngine(provider="openai", model="gpt-4o", api_key="sk-test")
    eng._llm = mock_get_llm_client.return_value
    return eng


@pytest.fixture
def tmp_projects_dir(tmp_path):
    """Create a temporary projects directory."""
    d = tmp_path / "projects"
    d.mkdir()
    return d


@pytest.fixture
def mock_starter():
    """Return fake starter files."""
    return {
        "package.json": '{"name": "test-app"}',
        "app/page.tsx": "export default function Page() { return <div>Hello</div>; }",
    }


@pytest.fixture
def mock_chat_response():
    """Factory for LLMResult responses."""
    def _make(
        content: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> LLMResult:
        return LLMResult(content=content, tool_calls=tool_calls or [], error=error)
    return _make


# =============================================================================
# CodegenConfig
# =============================================================================


class TestCodegenConfig:
    def test_defaults(self):
        """Default config uses the module-level constants."""
        cfg = CodegenConfig()
        assert cfg.max_tool_iterations == DEFAULT_MAX_TOOL_ITERATIONS
        assert cfg.max_tool_iterations == 20
        assert cfg.shell_output_limit == 8192
        assert cfg.auto_approve_shell is False
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 4096
        assert cfg.project_root is None

    def test_custom_values(self):
        cfg = CodegenConfig(
            max_tool_iterations=5,
            shell_output_limit=1024,
            auto_approve_shell=True,
            temperature=0.8,
            max_tokens=2048,
            project_root=Path("/custom/root"),
        )
        assert cfg.max_tool_iterations == 5
        assert cfg.shell_output_limit == 1024
        assert cfg.auto_approve_shell is True
        assert cfg.temperature == 0.8
        assert cfg.max_tokens == 2048
        assert cfg.project_root == Path("/custom/root")

    def test_frozen_like_behaviour(self):
        """Dataclass fields are mutable by default."""
        cfg = CodegenConfig()
        cfg.max_tool_iterations = 99
        assert cfg.max_tool_iterations == 99


# =============================================================================
# create_codegen_engine factory
# =============================================================================


class TestCreateCodegenEngine:
    def test_default_factory(self, mock_get_llm_client):
        engine = create_codegen_engine()
        assert isinstance(engine, CodegenEngine)
        assert engine.provider == "hermes"
        assert engine._config.max_tool_iterations == DEFAULT_MAX_TOOL_ITERATIONS
        assert engine._config.auto_approve_shell is False
        assert engine._profile == "swarm1"

    def test_custom_factory(self, mock_get_llm_client):
        engine = create_codegen_engine(
            provider="openai",
            model="gpt-4o",
            api_key="sk-123",
            profile="default",
            auto_approve_shell=True,
            max_iterations=10,
        )
        assert engine.provider == "openai"
        assert engine._model == "gpt-4o"
        assert engine._api_key == "sk-123"
        assert engine._profile == "default"
        assert engine._config.max_tool_iterations == 10
        assert engine._config.auto_approve_shell is True

    def test_factory_passes_storage(self, mock_get_llm_client):
        storage = MagicMock()
        engine = create_codegen_engine(storage=storage)
        assert engine._storage is storage


# =============================================================================
# _build_openai_tools
# =============================================================================


class TestBuildOpenaiTools:
    def test_returns_list_of_tool_dicts(self):
        tools = _build_openai_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 3  # read_file, write_patch, shell_exec

        names = {t["function"]["name"] for t in tools}
        assert "read_file" in names
        assert "write_patch" in names
        assert "shell_exec" in names

    def test_each_tool_has_required_keys(self):
        for tool in _OPENAI_TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_parameters_are_json_schema(self):
        for tool in _OPENAI_TOOLS:
            schema = tool["function"]["parameters"]
            assert "type" in schema
            assert schema["type"] == "object"

    def test_read_file_schema(self):
        tools = [t for t in _OPENAI_TOOLS if t["function"]["name"] == "read_file"]
        assert len(tools) == 1
        props = tools[0]["function"]["parameters"]["properties"]
        assert "path" in props

    def test_write_patch_schema(self):
        tools = [t for t in _OPENAI_TOOLS if t["function"]["name"] == "write_patch"]
        assert len(tools) == 1
        props = tools[0]["function"]["parameters"]["properties"]
        assert "path" in props
        assert "content" in props


# =============================================================================
# CodegenEngine.__init__
# =============================================================================


class TestCodegenEngineInit:
    def test_creates_llm_client(self):
        with patch("cortex.codegen.engine.get_llm_client") as m:
            eng = CodegenEngine(provider="gemini", model="gemini-pro", api_key="gk", profile="p1")
            m.assert_called_once_with(
                provider="gemini",
                model="gemini-pro",
                api_key="gk",
                base_url=None,
                profile="p1",
            )
            assert eng._provider == "gemini"
            assert eng._model == "gemini-pro"
            assert eng._api_key == "gk"
            assert eng._profile == "p1"

    def test_default_config_used_when_none(self, mock_get_llm_client):
        eng = CodegenEngine()
        assert isinstance(eng._config, CodegenConfig)
        assert eng._config.max_tool_iterations == DEFAULT_MAX_TOOL_ITERATIONS

    def test_custom_config_passed(self, mock_get_llm_client):
        cfg = CodegenConfig(max_tool_iterations=5)
        eng = CodegenEngine(config=cfg)
        assert eng._config.max_tool_iterations == 5

    def test_storage_passed(self, mock_get_llm_client):
        storage = MagicMock()
        eng = CodegenEngine(storage=storage)
        assert eng._storage is storage

    def test_provider_property(self, mock_get_llm_client):
        eng = CodegenEngine(provider="openai")
        assert eng.provider == "openai"

    def test_model_property_without_model(self, mock_get_llm_client):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.model = "hermes/swarm1"
        eng = CodegenEngine(provider="hermes", model=None)
        assert eng.model == "hermes/swarm1"

    def test_model_property_with_model(self, mock_get_llm_client):
        eng = CodegenEngine(provider="openai", model="gpt-4o")
        assert eng.model == "gpt-4o"


# =============================================================================
# _is_project_empty
# =============================================================================


class TestIsProjectEmpty:
    def test_non_existent_dir(self, tmp_path):
        eng = CodegenEngine.__new__(CodegenEngine)
        d = tmp_path / "nonexistent"
        assert eng._is_project_empty(d) is True

    def test_empty_dir(self, tmp_path):
        eng = CodegenEngine.__new__(CodegenEngine)
        d = tmp_path / "empty"
        d.mkdir()
        assert eng._is_project_empty(d) is True

    def test_only_hidden_files(self, tmp_path):
        eng = CodegenEngine.__new__(CodegenEngine)
        d = tmp_path / "hidden_only"
        d.mkdir()
        (d / ".gitkeep").write_text("")
        assert eng._is_project_empty(d) is True

    def test_non_hidden_file(self, tmp_path):
        eng = CodegenEngine.__new__(CodegenEngine)
        d = tmp_path / "nonempty"
        d.mkdir()
        (d / "readme.md").write_text("# Hello")
        assert eng._is_project_empty(d) is False

    def test_mixed_hidden_and_visible(self, tmp_path):
        eng = CodegenEngine.__new__(CodegenEngine)
        d = tmp_path / "mixed"
        d.mkdir()
        (d / ".gitkeep").write_text("")
        (d / "src").mkdir()
        assert eng._is_project_empty(d) is False


# =============================================================================
# _select_family
# =============================================================================


class TestSelectFamily:
    @pytest.mark.parametrize("provider,expected", [
        ("hermes", "openai-chat"),
        ("openai", "openai-chat"),
        ("openrouter", "openai-chat"),
        ("deepseek", "openai-chat"),
        ("gemini", "gemini"),
        ("ollama", "ollama"),
        ("unknown", "openai-chat"),
        ("", "openai-chat"),
    ])
    def test_maps_correctly(self, mock_get_llm_client, provider, expected):
        eng = CodegenEngine(provider=provider)
        assert eng._select_family() == expected


# =============================================================================
# _load_project_context
# =============================================================================


class TestLoadProjectContext:
    def test_without_storage_returns_none(self, mock_get_llm_client):
        eng = CodegenEngine(provider="hermes")
        eng._storage = None
        ctx = eng._load_project_context("test-proj", "make a page", "next")
        assert ctx is None

    def test_with_storage_calls_load_context(self, mock_get_llm_client):
        eng = CodegenEngine(provider="hermes")
        mock_ctx = MagicMock()
        # _load_project_context does a local import of load_context,
        # so we patch at the source module
        with patch("cortex.codegen.context.load_context", return_value=mock_ctx) as m:
            eng._storage = MagicMock()
            ctx = eng._load_project_context("test-proj", "make a page", "next")
            m.assert_called_once_with(eng._storage, "test-proj", "make a page", "next")
            assert ctx is mock_ctx


# =============================================================================
# _ensure_project
# =============================================================================


class TestEnsureProject:
    @pytest.mark.asyncio
    async def test_creates_project_dir(self, mock_get_llm_client, tmp_path):
        eng = CodegenEngine(provider="hermes")
        eng._config.project_root = tmp_path / "projects"
        result = await eng._ensure_project("my-app", starter_name=None)
        assert result is not None
        assert result.exists()
        assert result.name == "my-app"
        assert result.parent == tmp_path / "projects"

    @pytest.mark.asyncio
    async def test_scaffolds_starter_when_empty(self, mock_get_llm_client, tmp_path, mock_starter):
        with patch("cortex.codegen.engine.get_starter", return_value=mock_starter):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"
            result = await eng._ensure_project("scaffold-test", starter_name="next")
            assert result.exists()
            assert (result / "package.json").exists()
            assert (result / "app/page.tsx").exists()
            assert (result / "package.json").read_text() == '{"name": "test-app"}'

    @pytest.mark.asyncio
    async def test_does_not_scaffold_when_starter_is_none(self, mock_get_llm_client, tmp_path):
        eng = CodegenEngine(provider="hermes")
        eng._config.project_root = tmp_path / "projects"
        result = await eng._ensure_project("no-starter", starter_name=None)
        assert result.exists()
        assert not list(result.iterdir())

    @pytest.mark.asyncio
    async def test_does_not_scaffold_when_dir_not_empty(self, mock_get_llm_client, tmp_path):
        with patch("cortex.codegen.engine.get_starter") as mock_starter:
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"
            d = eng._config.project_root / "existing"
            d.mkdir(parents=True)
            (d / "keep.txt").write_text("stuff")
            result = await eng._ensure_project("existing", starter_name="next")
            mock_starter.assert_not_called()
            assert result == d

    @pytest.mark.asyncio
    async def test_starter_not_found(self, mock_get_llm_client, tmp_path):
        with patch("cortex.codegen.engine.get_starter", return_value=None):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"
            result = await eng._ensure_project("unknown-starter", starter_name="nonexistent")
            assert result.exists()
            assert not list(result.iterdir())

    @pytest.mark.asyncio
    async def test_with_storage(self, mock_get_llm_client, tmp_path):
        storage = MagicMock()
        storage.project_dir.return_value = tmp_path / "storage-projects" / "my-app"
        with patch("cortex.codegen.engine.get_starter") as mock_starter:
            mock_starter.return_value = {"file.txt": "content"}
            eng = CodegenEngine(provider="hermes", storage=storage)
            eng._config.project_root = tmp_path / "projects"
            result = await eng._ensure_project("my-app", starter_name="next")
            assert result.exists()
            assert (result / "file.txt").exists()
            storage.write_file.assert_called_once_with("my-app", "file.txt", "content")

    @pytest.mark.asyncio
    async def test_storage_project_dir_raises_exception(self, mock_get_llm_client, tmp_path):
        """Fallback to config root when storage.project_dir raises."""
        storage = MagicMock()
        storage.project_dir.side_effect = ValueError("no dir")
        with patch("cortex.codegen.engine.get_starter") as mock_starter:
            mock_starter.return_value = {"f.txt": "data"}
            eng = CodegenEngine(provider="hermes", storage=storage)
            eng._config.project_root = tmp_path / "projects"
            result = await eng._ensure_project("fallback-test", starter_name="next")
            assert result is not None
            assert result.parent == tmp_path / "projects"
            # Since starter_name is truthy and dir was just created (empty),
            # it should scaffold
            assert (result / "f.txt").exists()

    @pytest.mark.asyncio
    async def test_storage_write_file_raises(self, mock_get_llm_client, tmp_path):
        """Exception in storage.write_file is silently swallowed."""
        storage = MagicMock()
        storage.project_dir.return_value = tmp_path / "sp" / "my-app"
        storage.write_file.side_effect = OSError("disk full")
        with patch("cortex.codegen.engine.get_starter") as mock_starter:
            mock_starter.return_value = {"f.txt": "data"}
            eng = CodegenEngine(provider="hermes", storage=storage)
            eng._config.project_root = tmp_path / "projects"
            result = await eng._ensure_project("my-app", starter_name="next")
            assert result.exists()
            assert (result / "f.txt").exists()  # still written to disk
            storage.write_file.assert_called_once()


# =============================================================================
# _generate_plan
# =============================================================================


class TestGeneratePlan:
    async def test_successful_plan(self, mock_get_llm_client, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(content="This is the plan")

        with (
            patch("cortex.codegen.engine.get_prompt", return_value="You are a planner."),
            patch("cortex.codegen.engine.render_context_block", return_value="<context>"),
        ):
            eng = CodegenEngine(provider="hermes")
            plan = await eng._generate_plan(prompt="Build a landing page", context="...", goal="goal")
            assert plan == "This is the plan"

    async def test_plan_error_returns_none(self, mock_get_llm_client, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(error="API error")

        with (
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            plan = await eng._generate_plan(prompt="test", context=None, goal="test")
            assert plan is None

    async def test_empty_plan_returns_none(self, mock_get_llm_client, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(content="   ")

        with (
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            plan = await eng._generate_plan(prompt="test", context=None, goal="test")
            assert plan is None

    async def test_passes_context_block_when_context_not_none(self, mock_get_llm_client, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(content="Plan with context")

        with (
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value="[CONTEXT BLOCK]") as mock_render,
        ):
            mock_ctx = MagicMock()
            eng = CodegenEngine(provider="hermes")
            plan = await eng._generate_plan(prompt="test", context=mock_ctx, goal="test")
            assert plan == "Plan with context"
            mock_render.assert_called_once_with(mock_ctx)

    async def test_no_context_block_when_context_none(self, mock_get_llm_client, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(content="Plan no context")

        with (
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block") as mock_render,
        ):
            eng = CodegenEngine(provider="hermes")
            plan = await eng._generate_plan(prompt="test", context=None, goal="test")
            assert plan == "Plan no context"
            mock_render.assert_not_called()

    async def test_provider_maps_to_correct_family(self, mock_get_llm_client, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(content="gemini plan")

        with patch("cortex.codegen.engine.get_prompt", return_value="gemini planner prompt") as m:
            eng = CodegenEngine(provider="gemini")
            plan = await eng._generate_plan(prompt="test", context=None, goal="test")
            assert plan == "gemini plan"
            m.assert_called_once_with("gemini", "planner")


# =============================================================================
# _build_tool_messages
# =============================================================================


class TestBuildToolMessages:
    def test_contains_system_and_user_messages(self, mock_get_llm_client):
        with (
            patch("cortex.codegen.engine.get_prompt", return_value="codegen prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value="<rendered ctx>"),
            patch("cortex.codegen.engine.tool_schemas_for_prompt", return_value="[tool schemas]"),
            patch("cortex.codegen.engine.tool_call_format_instruction", return_value="[format instructions]"),
        ):
            eng = CodegenEngine(provider="hermes")
            messages = eng._build_tool_messages(
                prompt="Build a page",
                plan="1. Create files",
                context=MagicMock(),
                goal="Build",
            )
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert "codegen prompt" in messages[0]["content"]
            assert "[tool schemas]" in messages[0]["content"]
            assert "[format instructions]" in messages[0]["content"]

    def test_no_tool_schemas_for_non_hermes(self, mock_get_llm_client):
        with (
            patch("cortex.codegen.engine.get_prompt", return_value="codegen prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
            patch("cortex.codegen.engine.tool_schemas_for_prompt") as mock_schemas,
            patch("cortex.codegen.engine.tool_call_format_instruction") as mock_fmt,
        ):
            eng = CodegenEngine(provider="openai", model="gpt-4o")
            messages = eng._build_tool_messages(
                prompt="Build",
                plan="Plan",
                context=None,
                goal="Build",
            )
            assert len(messages) == 2
            # System content should NOT have tool schemas appended for openai
            assert messages[0]["content"] == "codegen prompt"  # no extras
            mock_schemas.assert_not_called()
            mock_fmt.assert_not_called()

    def test_plan_in_user_content(self, mock_get_llm_client):
        with (
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            messages = eng._build_tool_messages(
                prompt="Make a login page",
                plan="Add login form",
                context=None,
                goal="Login feature",
            )
            user_msg = messages[1]["content"]
            assert "Make a login page" in user_msg
            assert "Add login form" in user_msg
            assert "read_file" in user_msg
            assert "write_patch" in user_msg

    def test_empty_prompt_and_plan_defaults(self, mock_get_llm_client):
        with (
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            messages = eng._build_tool_messages(
                prompt="",
                plan="",
                context=None,
                goal="Goal",
            )
            user_msg = messages[1]["content"]
            assert "(empty)" in user_msg
            assert "(none)" in user_msg

    def test_context_block_rendered_when_context_present(self, mock_get_llm_client):
        with (
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value="[CONTEXT!]"),
        ):
            eng = CodegenEngine(provider="hermes")
            messages = eng._build_tool_messages(
                prompt="test", plan="plan", context=MagicMock(), goal="goal"
            )
            user_msg = messages[1]["content"]
            assert "[CONTEXT!]" in user_msg


# =============================================================================
# run() - Full pipeline
# =============================================================================


class TestRun:
    """Test the full run() async generator pipeline."""

    async def test_happy_path(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Full pipeline yields correct event sequence."""
        mock_llm = mock_get_llm_client.return_value
        # Plan response
        mock_llm.chat.side_effect = [
            mock_chat_response(content="The implementation plan."),  # _generate_plan
            mock_chat_response(content="Done!", tool_calls=[]),      # _tool_loop -> no tools
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            events: list[CodegenEvent] = []
            async for event in eng.run(project_id="test-run", prompt="Build my app"):
                events.append(event)

            # Verify event types in order
            event_types = [e.type for e in events]
            assert event_types == [
                "status",    # init
                "status",    # context
                "status",    # planning
                "plan",      # plan
                "status",    # generating
                "status",    # complete (tool loop finished)
                "status",    # done
                "done",      # done
            ]

            # Check specific events
            assert events[0].stage == "init"
            assert events[3].type == "plan"
            assert events[3].content == "The implementation plan."
            assert events[-1].type == "done"
            assert isinstance(events[-1], DoneEvent)
            assert events[-2].type == "status"
            assert events[-2].stage == "done"

    async def test_with_file_writes(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Tool loop with write_patch yields FileWriteEvents."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "write_patch",
                    "arguments": {"path": "hello.txt", "content": "Hello world"},
                },
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="The plan."),               # _generate_plan
            mock_chat_response(content="", tool_calls=tool_calls), # _tool_loop iter 0
            mock_chat_response(content="All done.", tool_calls=[]),# _tool_loop iter 1
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            events: list[CodegenEvent] = []
            async for event in eng.run(project_id="wr-test", prompt="Write hello.txt"):
                events.append(event)

            file_write_events = [e for e in events if e.type == "file_write"]
            assert len(file_write_events) == 1
            fwe = file_write_events[0]
            assert fwe.path == "hello.txt"
            assert fwe.content == "Hello world"

            tool_call_events = [e for e in events if e.type == "tool_call"]
            assert len(tool_call_events) == 1
            assert tool_call_events[0].tool_name == "write_patch"

    async def test_plan_failure(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """When plan generation fails, ErrorEvent is yielded."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(error="Plan failed")

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            events: list[CodegenEvent] = []
            async for event in eng.run(project_id="fail-plan", prompt="test"):
                events.append(event)

            errors = [e for e in events if e.type == "error"]
            assert len(errors) == 1
            assert "Failed to generate plan" in errors[0].message
            assert errors[0].recoverable is False
            # Should stop before DoneEvent
            assert events[-1].type == "error"

    async def test_ensure_project_failure(self, mock_get_llm_client, tmp_path):
        """When project dir can't be created, ErrorEvent is yielded."""
        eng = CodegenEngine(provider="hermes")
        # Mock _ensure_project to return None
        with patch.object(eng, "_ensure_project", return_value=None):
            events: list[CodegenEvent] = []
            async for event in eng.run(project_id="fail-init", prompt="test"):
                events.append(event)

            assert len(events) >= 1
            assert events[0].type == "status"
            # Next should be error
            errors = [e for e in events if e.type == "error"]
            assert len(errors) == 1
            assert "Failed to initialize project directory" in errors[0].message

    async def test_without_starter_name_skips_scaffold(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """When starter_name is None, no scaffold happens."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]

        with (
            patch("cortex.codegen.engine.get_starter") as mock_get_starter,
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            events: list[CodegenEvent] = []
            async for event in eng.run(
                project_id="no-starter", prompt="test", starter_name=None
            ):
                events.append(event)

            # get_starter should not be called because starter_name is None
            mock_get_starter.assert_not_called()
            assert events[-1].type == "done"

    async def test_tool_loop_llm_error(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """LLM error in tool loop yields ErrorEvent with the error message from LLM result."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),                    # _generate_plan ok
            mock_chat_response(error="LLM crashed"),                # _tool_loop fails
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            events: list[CodegenEvent] = []
            async for event in eng.run(project_id="llm-error", prompt="test"):
                events.append(event)

            errors = [e for e in events if e.type == "error"]
            assert len(errors) == 1
            assert errors[0].message == "LLM crashed"

    async def test_tool_loop_exception_caught(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Exception in _tool_loop is caught and yielded as ErrorEvent."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            # Force _tool_loop to raise
            async def broken_loop(*a, **kw):
                raise RuntimeError("loop crashed")
                yield  # pragma: no cover

            with patch.object(eng, "_tool_loop", broken_loop):
                events: list[CodegenEvent] = []
                async for event in eng.run(project_id="crash", prompt="test"):
                    events.append(event)

                errors = [e for e in events if e.type == "error"]
                assert len(errors) == 1
                assert "tool loop failed: loop crashed" in errors[0].message

    async def test_non_recoverable_error_stops_pipeline(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Non-recoverable error in tool loop stops the pipeline early."""
        mock_llm = mock_get_llm_client.return_value
        # First iteration: read_file succeeds with a non-existent file
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "nonexistent.py"}},
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),  # finish
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            events: list[CodegenEvent] = []
            async for event in eng.run(project_id="err-stop", prompt="test"):
                events.append(event)

            # read_file of nonexistent file yields a ToolResultEvent with error,
            # but does NOT yield an ErrorEvent — so the pipeline continues
            tool_results = [e for e in events if e.type == "tool_result"]
            read_results = [e for e in tool_results if e.tool_name == "read_file"]
            assert len(read_results) == 1
            assert "error:" in read_results[0].output
            # Pipeline should continue and complete normally
            assert events[-1].type == "done"


# =============================================================================
# _tool_loop
# =============================================================================


class TestToolLoop:
    """Test the tool loop in isolation."""

    async def test_no_tool_calls_returns_immediately(self, mock_get_llm_client, mock_chat_response):
        """LLM returns no tool calls -> StatusEvent(complete) and return."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(content="All done.")
        eng = CodegenEngine(provider="hermes")

        messages = [{"role": "system", "content": "You are a coder."}]
        project_root = Path("/tmp")

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "status"
        assert events[0].stage == "complete"

    async def test_read_file_tool(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """read_file tool execution yields ToolCallEvent + ToolResultEvent."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "hello.txt"}},
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]
        eng = CodegenEngine(provider="hermes")

        # Create the file to read
        project_root = tmp_path / "tool-loop-project"
        project_root.mkdir()
        (project_root / "hello.txt").write_text("file content")

        messages = [{"role": "system", "content": "prompt"}]

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        # First iteration: tool call + tool result, Second: complete
        assert len(events) == 3
        assert events[0].type == "tool_call"
        assert events[0].tool_name == "read_file"
        assert events[0].args == {"path": "hello.txt"}
        assert events[1].type == "tool_result"
        assert events[1].tool_name == "read_file"
        assert events[1].output == "file content"
        assert events[2].type == "status"
        assert events[2].stage == "complete"

    async def test_write_patch_tool(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """write_patch tool execution yields FileWriteEvent + ToolResultEvent."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "write_patch",
                    "arguments": {"path": "output.txt", "content": "new content"},
                },
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]
        eng = CodegenEngine(provider="hermes")

        project_root = tmp_path / "write-patch-test"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        assert len(events) == 4  # tool_call + file_write + tool_result + complete
        assert events[0].type == "tool_call"
        assert events[0].tool_name == "write_patch"
        assert events[1].type == "file_write"
        assert events[1].path == "output.txt"
        assert events[1].content == "new content"
        assert events[2].type == "tool_result"
        assert events[2].tool_name == "write_patch"
        assert events[3].type == "status"
        assert events[3].stage == "complete"

        # Verify file was actually written
        assert (project_root / "output.txt").read_text() == "new content"

    async def test_shell_exec_auto_approve(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """shell_exec with auto_approve_shell=True yields ToolCallEvent + ToolResultEvent
        (no PermissionRequestEvent since auto-approved)."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "arguments": {"command": "echo hello", "reason": "testing"},
                },
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]
        eng = CodegenEngine(provider="hermes")
        eng._config.auto_approve_shell = True

        project_root = tmp_path / "shell-test"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        # Order: tool_call -> tool_result -> complete (no PermissionRequestEvent)
        assert len(events) == 3
        assert events[0].type == "tool_call"
        assert events[0].tool_name == "shell_exec"
        assert events[1].type == "tool_result"
        assert events[1].tool_name == "shell_exec"
        assert events[1].approved is True
        assert "hello" in events[1].output
        assert events[2].type == "status"
        assert events[2].stage == "complete"

    async def test_shell_exec_denied(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """shell_exec without auto_approve yields ToolCallEvent + PermissionRequestEvent
        + ToolResultEvent with approved=False."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_denied",
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "arguments": {"command": "rm -rf /", "reason": "cleanup"},
                },
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]
        eng = CodegenEngine(provider="hermes")
        eng._config.auto_approve_shell = False  # default

        project_root = tmp_path / "shell-denied"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        assert events[0].type == "tool_call"
        assert events[0].tool_name == "shell_exec"
        assert events[1].type == "permission_request"
        assert events[1].command == "rm -rf /"
        assert events[2].type == "tool_result"
        assert events[2].tool_name == "shell_exec"
        assert events[2].approved is False
        assert "denied" in events[2].output
        assert events[3].type == "status"
        assert events[3].stage == "complete"

    async def test_unknown_tool(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Unknown tool name yields ToolResultEvent with error message."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_unknown",
                "type": "function",
                "function": {"name": "unknown_tool", "arguments": {}},
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]
        eng = CodegenEngine(provider="hermes")

        project_root = tmp_path / "unknown-tool"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        tool_results = [e for e in events if e.type == "tool_result"]
        assert len(tool_results) == 1
        assert "unknown tool" in tool_results[0].output
        assert "unknown_tool" in tool_results[0].output

    async def test_llm_call_exception(self, mock_get_llm_client, tmp_path):
        """Exception during LLM.chat in tool loop yields ErrorEvent."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.side_effect = ValueError("Connection refused")
        eng = CodegenEngine(provider="hermes")

        project_root = tmp_path / "llm-exc"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "LLM call failed" in events[0].message
        assert "Connection refused" in events[0].message

    async def test_llm_result_error(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """LLM result with error in tool loop yields ErrorEvent."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(error="Rate limited")
        eng = CodegenEngine(provider="hermes")

        project_root = tmp_path / "llm-error"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert events[0].message == "Rate limited"

    async def test_max_iterations_reached(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Tool loop stops when max_iterations is reached."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_iter",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "a.txt"}},
            },
        ]
        # Return tool_calls for every iteration (more than max)
        mock_llm.chat.return_value = mock_chat_response(content="", tool_calls=tool_calls)

        eng = CodegenEngine(provider="hermes")
        eng._config.max_tool_iterations = 2  # small number for test

        project_root = tmp_path / "max-iter"
        project_root.mkdir()
        (project_root / "a.txt").write_text("data")
        messages = [{"role": "system", "content": "prompt"}]

        events: list[CodegenEvent] = []
        async for event in eng._tool_loop(messages, project_root, "test"):
            events.append(event)

        # Expect: 2 iterations of (tool_call + tool_result) + status(max_iterations) + status(complete)
        status_events = [e for e in events if e.type == "status"]
        max_iter_status = [e for e in status_events if e.stage == "max_iterations_reached"]
        assert len(max_iter_status) == 1
        assert status_events[-1].stage == "complete"

    async def test_text_based_message_format(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Text-based (hermes) uses 'user' role for tool results."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "f.txt"}},
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Tool response with TOOL_CALL marker", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]
        eng = CodegenEngine(provider="hermes")

        project_root = tmp_path / "text-format"
        project_root.mkdir()
        (project_root / "f.txt").write_text("content")
        messages = [{"role": "system", "content": "prompt"}]

        async for _ in eng._tool_loop(messages, project_root, "test"):
            pass

        # After one tool iteration: system (initial) + assistant + user (tool result) = 3
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert "Tool" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "Tool result" in messages[2]["content"]

    async def test_native_function_calling_message_format(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Native function calling (openai) uses 'tool' role for results."""
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_native",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "f.txt"}},
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]
        eng = CodegenEngine(provider="openai", model="gpt-4o", api_key="sk-test")
        eng._llm = mock_llm

        project_root = tmp_path / "native-format"
        project_root.mkdir()
        (project_root / "f.txt").write_text("content")
        messages = [{"role": "system", "content": "prompt"}]

        async for _ in eng._tool_loop(messages, project_root, "test"):
            pass

        # After one tool iteration: system (initial) + assistant (with tool_calls) + tool (result) = 3
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_native"

    async def test_openai_tools_param_passed(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """OpenAI provider passes tools param to LLM."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(content="Done.", tool_calls=[])

        eng = CodegenEngine(provider="openai", model="gpt-4o", api_key="sk-test")
        eng._llm = mock_llm

        project_root = tmp_path / "tools-param"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        async for _ in eng._tool_loop(messages, project_root, "test"):
            pass

        # Verify the tools parameter was passed to chat
        assert mock_llm.chat.called
        _, kwargs = mock_llm.chat.call_args
        assert "tools" in kwargs
        assert kwargs["tools"] is not None  # _OPENAI_TOOLS

    async def test_hermes_does_not_pass_tools_param(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Hermes provider does NOT pass tools param (text-based)."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(content="Done.", tool_calls=[])

        eng = CodegenEngine(provider="hermes")
        eng._llm = mock_llm

        project_root = tmp_path / "no-tools-param"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        async for _ in eng._tool_loop(messages, project_root, "test"):
            pass

        _, kwargs = mock_llm.chat.call_args
        assert kwargs.get("tools") is None

    async def test_storage_write_patch(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """write_patch with storage calls storage.write_file."""
        storage = MagicMock()
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_storage",
                "type": "function",
                "function": {
                    "name": "write_patch",
                    "arguments": {"path": "stored.txt", "content": "stored content"},
                },
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]
        eng = CodegenEngine(provider="hermes", storage=storage)

        project_root = tmp_path / "storage-write"
        project_root.mkdir()
        messages = [{"role": "system", "content": "prompt"}]

        async for _ in eng._tool_loop(messages, project_root, "stored-project"):
            pass

        # The file should be written via storage
        storage.write_file.assert_called_once_with(
            "stored-project", "stored.txt", "stored content"
        )


# =============================================================================
# run_and_collect convenience wrapper
# =============================================================================


class TestRunAndCollect:
    async def test_collects_results(self, mock_get_llm_client, tmp_path, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            files_written, summary, error = await eng.run_and_collect(
                project_id="collect-test", prompt="Build it"
            )
            assert files_written == []
            assert "Generated" in summary
            assert error is None

    async def test_collects_error(self, mock_get_llm_client, tmp_path, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.return_value = mock_chat_response(error="Plan failed")

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            files_written, summary, error = await eng.run_and_collect(
                project_id="collect-error", prompt="test"
            )
            assert error is not None
            assert "Failed to generate plan" in error

    async def test_collects_files(self, mock_get_llm_client, tmp_path, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        tool_calls = [
            {
                "id": "call_f1",
                "type": "function",
                "function": {
                    "name": "write_patch",
                    "arguments": {"path": "file1.py", "content": "# code"},
                },
            },
        ]
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),
            mock_chat_response(content="", tool_calls=tool_calls),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            files_written, summary, error = await eng.run_and_collect(
                project_id="collect-files", prompt="Write file1.py"
            )
            assert files_written == ["file1.py"]
            assert "Generated 1 file(s)" in summary
            assert error is None

    async def test_non_recoverable_error_breaks(self, mock_get_llm_client, tmp_path, mock_chat_response):
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            # Force tool loop to yield a non-recoverable error
            async def error_loop(*a, **kw):
                yield ErrorEvent(message="Fatal", recoverable=False)

            with patch.object(eng, "_tool_loop", error_loop):
                files_written, summary, error = await eng.run_and_collect(
                    project_id="fatal", prompt="test"
                )
                assert error == "Fatal"


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    async def test_run_empty_prompt(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Run with empty prompt string."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            events: list[CodegenEvent] = []
            async for event in eng.run(project_id="empty-prompt", prompt=""):
                events.append(event)
            assert events[-1].type == "done"

    async def test_gemini_ollama_providers_run(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Non-hermes providers (gemini, ollama) also work in the pipeline."""
        for test_provider in ("gemini", "ollama"):
            mock_llm = mock_get_llm_client.return_value
            mock_llm.chat.side_effect = [
                mock_chat_response(content="Plan for " + test_provider),
                mock_chat_response(content="Done.", tool_calls=[]),
            ]

            with (
                patch("cortex.codegen.engine.get_starter", return_value=None),
                patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
                patch("cortex.codegen.engine.render_context_block", return_value=""),
            ):
                eng = CodegenEngine(provider=test_provider)
                eng._llm = mock_llm
                eng._config.project_root = tmp_path / "projects" / test_provider

                events: list[CodegenEvent] = []
                async for event in eng.run(project_id=f"run-{test_provider}", prompt="test"):
                    events.append(event)
                assert events[-1].type == "done"
                assert any(e.type == "plan" and test_provider in e.content for e in events)

    async def test_run_with_goal_parameter(self, mock_get_llm_client, tmp_path, mock_chat_response):
        """Goal parameter is passed through and used as fallback for prompt."""
        mock_llm = mock_get_llm_client.return_value
        mock_llm.chat.side_effect = [
            mock_chat_response(content="Plan."),
            mock_chat_response(content="Done.", tool_calls=[]),
        ]

        with (
            patch("cortex.codegen.engine.get_starter", return_value=None),
            patch("cortex.codegen.engine.get_prompt", return_value="prompt"),
            patch("cortex.codegen.engine.render_context_block", return_value=""),
        ):
            eng = CodegenEngine(provider="hermes")
            eng._config.project_root = tmp_path / "projects"

            events: list[CodegenEvent] = []
            async for event in eng.run(
                project_id="with-goal",
                prompt="Make a login page",
                goal="User authentication feature",
            ):
                events.append(event)
            assert events[-1].type == "done"
