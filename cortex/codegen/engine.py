"""
cortex/codegen/engine.py — CodegenEngine: two-stage code generation pipeline.

Pipeline:
  1. INIT → scaffold starter if project empty
  2. CONTEXT → load project files + tree
  3. PLAN → LLM produces implementation plan
  4. TOOL LOOP → LLM + tools iterate (read_file, write_patch, shell_exec)
  5. RESULT → return written files + summary

Supports all LLM providers (hermes, openai, gemini, ollama, etc.)
via the unified LLMClient interface.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .context import load_context, render_context_block
from .llm import LLMClient, LLMResult, get_llm_client
from .prompts import get_prompt
from .starter import get_starter
from .tools import (
    ALL_TOOLS,
    SHELL_EXEC_TOOL,
    ToolDef,
    execute_read_file,
    execute_shell_exec,
    execute_write_patch,
    tool_call_format_instruction,
    tool_schemas_for_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOOL_ITERATIONS = 20
DEFAULT_SHELL_OUTPUT_LIMIT = 8192

# ---------------------------------------------------------------------------
# Event types (streaming)
# ---------------------------------------------------------------------------


@dataclass
class StatusEvent:
    type: str = "status"
    stage: str = ""
    note: str = ""


@dataclass
class PlanEvent:
    type: str = "plan"
    content: str = ""


@dataclass
class ToolCallEvent:
    type: str = "tool_call"
    tool_name: str = ""
    tool_call_id: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEvent:
    type: str = "tool_result"
    tool_name: str = ""
    tool_call_id: str = ""
    output: str = ""
    approved: bool = True


@dataclass
class FileWriteEvent:
    type: str = "file_write"
    path: str = ""
    content: str = ""


@dataclass
class PermissionRequestEvent:
    type: str = "permission_request"
    tool_call_id: str = ""
    command: str = ""
    reason: str = ""


@dataclass
class ErrorEvent:
    type: str = "error"
    message: str = ""
    recoverable: bool = False


@dataclass
class DoneEvent:
    type: str = "done"
    files_written: List[str] = field(default_factory=list)
    summary: str = ""


CodegenEvent = (
    StatusEvent
    | PlanEvent
    | ToolCallEvent
    | ToolResultEvent
    | FileWriteEvent
    | PermissionRequestEvent
    | ErrorEvent
    | DoneEvent
)


# ---------------------------------------------------------------------------
# Engine config
# ---------------------------------------------------------------------------


@dataclass
class CodegenConfig:
    """Configuration for the CodegenEngine."""

    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS
    shell_output_limit: int = DEFAULT_SHELL_OUTPUT_LIMIT
    auto_approve_shell: bool = False
    temperature: float = 0.2
    max_tokens: int = 4096
    project_root: Optional[Path] = None


# ---------------------------------------------------------------------------
# Tool definitions for LLM (OpenAI-compatible format)
# ---------------------------------------------------------------------------


def _build_openai_tools() -> List[Dict[str, Any]]:
    """Build OpenAI-compatible tool definitions for the codegen LLM."""
    tools = []
    for tool in ALL_TOOLS:
        schema = tool.parameters
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        })
    return tools


_OPENAI_TOOLS = _build_openai_tools()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CodegenEngine:
    """Two-stage code generation engine.

    Usage:
        engine = CodegenEngine(provider="openai", model="gpt-4o", api_key="...")
        async for event in engine.run(project_id="my-app", prompt="Create a landing page"):
            if event.type == "file_write":
                print(f"  Wrote: {event.path}")
    """

    def __init__(
        self,
        provider: str = "hermes",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        profile: str = "swarm1",
        config: Optional[CodegenConfig] = None,
        storage: Any = None,
    ):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._profile = profile
        self._config = config or CodegenConfig()
        self._storage = storage

        # Create LLM client
        self._llm = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            profile=profile,
        )

    @property
    def provider(self) -> str:
        """The LLM provider name."""
        return self._provider

    @property
    def model(self) -> str:
        """The model name."""
        return self._model or self._llm.model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        project_id: str,
        prompt: str,
        starter_name: Optional[str] = "next",
        goal: Optional[str] = None,
    ) -> AsyncIterator[CodegenEvent]:
        """Run the full codegen pipeline.

        Args:
            project_id: Unique project identifier
            prompt: User's natural language request
            starter_name: Starter template name (None = no scaffold)
            goal: Optional goal description (used for context)

        Yields:
            CodegenEvent objects for streaming progress
        """
        goal = goal or prompt

        # Phase 1: Ensure project exists and scaffold if empty
        yield StatusEvent(stage="init", note="Preparing project")
        project_root = await self._ensure_project(project_id, starter_name)
        if project_root is None:
            yield ErrorEvent(message="Failed to initialize project directory")
            return

        # Phase 2: Load context
        yield StatusEvent(stage="context", note="Reading project files")
        context = self._load_project_context(project_id, prompt, starter_name)

        # Phase 3: Plan
        yield StatusEvent(stage="planning", note="Generating implementation plan")
        plan = await self._generate_plan(prompt, context, goal)
        if plan is None:
            yield ErrorEvent(message="Failed to generate plan", recoverable=False)
            return

        yield PlanEvent(content=plan)

        # Build messages for the tool loop
        messages = self._build_tool_messages(prompt, plan, context, goal)

        # Phase 4: Tool loop
        yield StatusEvent(stage="generating", note="Writing files")
        files_written: List[str] = []

        try:
            async for event in self._tool_loop(
                messages=messages,
                project_root=project_root,
                project_id=project_id,
            ):
                if event.type == "file_write":
                    files_written.append(event.path)
                yield event

                # Stop on non-recoverable error
                if event.type == "error" and not getattr(event, "recoverable", True):
                    return

        except Exception as exc:
            logger.exception("Tool loop failed")
            yield ErrorEvent(message=f"tool loop failed: {exc}")
            return

        # Phase 5: Done
        yield StatusEvent(stage="done", note="Code generation complete")
        yield DoneEvent(
            files_written=files_written,
            summary=f"Generated {len(files_written)} file(s) using {self._provider}/{self.model}",
        )

    # ------------------------------------------------------------------
    # Phase 1: Project setup
    # ------------------------------------------------------------------

    async def _ensure_project(
        self, project_id: str, starter_name: Optional[str]
    ) -> Optional[Path]:
        """Ensure project directory exists and scaffold if empty."""
        if self._storage is not None:
            try:
                from pathlib import Path as PPath
                project_dir = self._storage.project_dir(project_id)
            except Exception:
                # Fallback to config root
                root = self._config.project_root or Path.cwd() / "projects"
                project_dir = root / project_id
                project_dir.mkdir(parents=True, exist_ok=True)
        else:
            root = self._config.project_root or Path.cwd() / "projects"
            project_dir = root / project_id
            project_dir.mkdir(parents=True, exist_ok=True)

        # Scaffold if empty (no files or starter_name specified)
        if starter_name and self._is_project_empty(project_dir):
            starter = get_starter(starter_name)
            if starter:
                logger.info(
                    "Scaffolding project %s with starter '%s' (%d files)",
                    project_id, starter_name, len(starter),
                )
                for rel_path, content in starter.items():
                    target = project_dir / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    if self._storage is not None:
                        try:
                            self._storage.write_file(project_id, rel_path, content)
                        except Exception:
                            pass

        return project_dir

    def _is_project_empty(self, project_dir: Path) -> bool:
        """Check if a project directory has any non-hidden files."""
        if not project_dir.exists():
            return True
        for entry in project_dir.iterdir():
            if not entry.name.startswith("."):
                return False
        return True

    # ------------------------------------------------------------------
    # Phase 2: Context loading
    # ------------------------------------------------------------------

    def _load_project_context(
        self, project_id: str, prompt: str, starter_name: Optional[str]
    ) -> Any:
        """Load project context for the LLM."""
        from .context import load_context as _load_context

        if self._storage is not None:
            return _load_context(self._storage, project_id, prompt, starter_name)
        return None

    # ------------------------------------------------------------------
    # Phase 3: Plan generation
    # ------------------------------------------------------------------

    def _select_family(self) -> str:
        """Select the prompt family based on provider."""
        family_map = {
            "openai": "openai-chat",
            "hermes": "openai-chat",
            "gemini": "gemini",
            "ollama": "ollama",
            "openrouter": "openai-chat",
            "deepseek": "openai-chat",
        }
        return family_map.get(self._provider, "openai-chat")

    async def _generate_plan(
        self,
        prompt: str,
        context: Any,
        goal: str,
    ) -> Optional[str]:
        """Generate an implementation plan using the LLM."""
        family = self._select_family()
        planner_prompt = get_prompt(family, "planner")

        # Build context block
        context_block = ""
        if context is not None:
            context_block = render_context_block(context)

        user_content = (
            f"{context_block}\n\nUser request:\n{prompt or '(empty)'}"
        )

        messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": user_content},
        ]

        result = await self._llm.chat(
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        if result.error:
            logger.error("Plan generation failed: %s", result.error)
            return None

        plan_text = result.content.strip()
        if not plan_text:
            logger.error("Plan generation returned empty content")
            return None

        return plan_text

    # ------------------------------------------------------------------
    # Phase 4: Messages preparation
    # ------------------------------------------------------------------

    def _build_tool_messages(
        self,
        prompt: str,
        plan: str,
        context: Any,
        goal: str,
    ) -> List[Dict[str, Any]]:
        """Build the initial message list for the tool loop."""
        family = self._select_family()
        codegen_prompt = get_prompt(family, "codegen")

        context_block = ""
        if context is not None:
            context_block = render_context_block(context)

        # Append tool schemas to codegen prompt
        is_text_based = self._provider == "hermes"
        if is_text_based:
            codegen_prompt += "\n\n" + tool_schemas_for_prompt()
            codegen_prompt += "\n\n" + tool_call_format_instruction()

        user_content = (
            f"{context_block}\n\n"
            f"User request:\n{prompt or '(empty)'}\n\n"
            f"Plan:\n{plan or '(none)'}\n\n"
            "Use the available tools to implement the plan. "
            "Call read_file to inspect existing files before modifying them, "
            "write_patch to create or overwrite files, and shell_exec "
            "(only if needed) to run build or test commands. "
            "Proceed tool call by tool call until the task is complete, "
            "then respond with a summary."
        )

        return [
            {"role": "system", "content": codegen_prompt},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Phase 4: Tool loop
    # ------------------------------------------------------------------

    async def _tool_loop(
        self,
        messages: List[Dict[str, Any]],
        project_root: Path,
        project_id: str,
    ) -> AsyncIterator[CodegenEvent]:
        """Execute the tool-calling loop.

        Iterates: LLM call → parse tool calls → execute → append → repeat
        """
        is_text_based = self._provider == "hermes"
        tools_param = None if is_text_based else _OPENAI_TOOLS

        for iteration in range(self._config.max_tool_iterations + 1):
            # Call LLM
            try:
                result = await self._llm.chat(
                    messages=messages,
                    tools=tools_param,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                )
            except Exception as exc:
                logger.exception("LLM call failed at iteration %d", iteration)
                yield ErrorEvent(message=f"LLM call failed: {exc}")
                return

            if result.error:
                yield ErrorEvent(message=result.error)
                return

            # Check for tool calls
            tool_calls = result.tool_calls
            if not tool_calls:
                # LLM finished — content is the summary
                yield StatusEvent(stage="complete", note=result.content[:200])
                return

            # Check iteration limit
            if iteration >= self._config.max_tool_iterations:
                yield StatusEvent(stage="max_iterations_reached")
                yield StatusEvent(
                    stage="complete",
                    note="Reached max iterations with tools remaining.",
                )
                return

            # Add assistant response to messages
            if is_text_based:
                # For text-based, add the raw content (which may include TOOL_CALL markers)
                messages.append({"role": "assistant", "content": result.content})
            else:
                # For native function calling, add structured response
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": result.content or "",
                }
                # Convert tool calls to OpenAI format
                raw_calls = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    raw_calls.append({
                        "id": tc.get("id", f"call_{iteration}_{len(raw_calls)}"),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": json.dumps(fn.get("arguments", {})),
                        },
                    })
                if raw_calls:
                    assistant_msg["tool_calls"] = raw_calls
                messages.append(assistant_msg)

            # Execute each tool call
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                args = fn.get("arguments", {})
                tool_call_id = tc.get("id", f"call_{iteration}")

                yield ToolCallEvent(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    args=args,
                )

                # Execute
                if tool_name == "read_file":
                    output = execute_read_file(
                        args.get("path", ""), project_root
                    )
                    yield ToolResultEvent(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        output=output,
                    )

                elif tool_name == "write_patch":
                    result_msg, file_event = execute_write_patch(
                        args.get("path", ""),
                        args.get("content", ""),
                        project_root,
                        storage=self._storage,
                        project_id=project_id,
                    )
                    output = result_msg
                    if file_event is not None:
                        yield FileWriteEvent(
                            path=file_event["path"],
                            content=file_event.get("content", ""),
                        )
                    yield ToolResultEvent(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        output=result_msg,
                    )

                elif tool_name == "shell_exec":
                    command = args.get("command", "")
                    reason = args.get("reason", "")

                    # Check approval
                    if not self._config.auto_approve_shell:
                        yield PermissionRequestEvent(
                            tool_call_id=tool_call_id,
                            command=command,
                            reason=reason,
                        )
                        # For now, auto-deny (user would need to approve via API)
                        approved = False
                    else:
                        approved = True

                    if not approved:
                        output = '{"error": "shell command denied"}'
                    else:
                        output = execute_shell_exec(
                            command,
                            project_root,
                            self._config.shell_output_limit,
                        )

                    yield ToolResultEvent(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        output=output,
                        approved=approved,
                    )

                else:
                    output = f"error: unknown tool {tool_name!r}"
                    yield ToolResultEvent(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        output=output,
                    )

                # Append tool result to messages
                if is_text_based:
                    messages.append({
                        "role": "user",
                        "content": f"Tool result ({tool_name}):\n{output}",
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": output,
                    })

            # Continue loop (next iteration will call LLM with updated messages)

    # ------------------------------------------------------------------
    # Synchronous convenience wrapper
    # ------------------------------------------------------------------

    async def run_and_collect(
        self,
        project_id: str,
        prompt: str,
        starter_name: Optional[str] = "next",
    ) -> Tuple[List[str], str, Optional[str]]:
        """Run the full pipeline and collect results.

        Returns:
            (files_written, summary, error)
        """
        files_written: List[str] = []
        error: Optional[str] = None
        summary = ""

        async for event in self.run(
            project_id=project_id,
            prompt=prompt,
            starter_name=starter_name,
        ):
            if event.type == "file_write":
                files_written.append(event.path)
            elif event.type == "done":
                summary = event.summary
            elif event.type == "error":
                error = event.message
                if not event.recoverable:
                    break

        return files_written, summary, error


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_codegen_engine(
    provider: str = "hermes",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    profile: str = "swarm1",
    storage: Any = None,
    auto_approve_shell: bool = False,
    max_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> CodegenEngine:
    """Create a configured CodegenEngine instance.

    Args:
        provider: 'hermes', 'openai', 'gemini', 'ollama', etc.
        model: Model name (required for direct providers)
        api_key: API key (required for cloud providers)
        profile: Hermes profile (only for provider='hermes')
        storage: Optional storage backend
        auto_approve_shell: Auto-approve shell commands
        max_iterations: Max tool loop iterations

    Returns:
        Configured CodegenEngine
    """
    config = CodegenConfig(
        max_tool_iterations=max_iterations,
        auto_approve_shell=auto_approve_shell,
    )
    return CodegenEngine(
        provider=provider,
        model=model,
        api_key=api_key,
        profile=profile,
        config=config,
        storage=storage,
    )
