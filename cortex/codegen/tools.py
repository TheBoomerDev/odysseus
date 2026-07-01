"""
cortex/codegen/tools.py — Tool definitions for the LLM codegen tool loop.

Each tool has:
  - A Pydantic input schema (for LLM function calling bindings)
  - An execution function (called by the engine after LLM requests it)

Tools available to the codegen LLM:
  - read_file(path) — read a project file
  - write_patch(path, content) — create or overwrite a file
  - shell_exec(command, reason) — run a shell command (requires approval)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .patcher import (
    _ensure_use_client,
    _normalize_path,
    _path_is_safe,
    _truncate,
)

# ---------------------------------------------------------------------------
# Tool input schemas (Pydantic)
# ---------------------------------------------------------------------------


class ReadFileInput(BaseModel):
    """Read a file from the project directory."""

    path: str = Field(
        description="File path relative to the project root"
    )


class WritePatchInput(BaseModel):
    """Create or overwrite a file with full content."""

    path: str = Field(
        description="File path relative to the project root"
    )
    content: str = Field(
        description="Full content to write (creates or overwrites the file)"
    )


class ShellExecInput(BaseModel):
    """Run a shell command in the project directory."""

    command: str = Field(
        description="Shell command to execute in the project directory"
    )
    reason: str = Field(
        description="Why this command is needed (shown to the user for approval)"
    )


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@dataclass
class ToolDef:
    """Definition of a tool available to the codegen LLM."""

    name: str
    description: str
    input_schema: type[BaseModel]
    execute: Callable[..., Any]
    requires_approval: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build JSON schema from the input model."""
        self.parameters = self.input_schema.model_json_schema()


# ---------------------------------------------------------------------------
# Execution functions
# ---------------------------------------------------------------------------


def execute_read_file(path: str, project_root: Path) -> str:
    """Read a file relative to project root; return contents or error string."""
    rel = _normalize_path(path)
    if rel is None:
        return "error: empty path"
    if not _path_is_safe(rel):
        return f"error: path outside project root: {path!r}"
    try:
        return (project_root / rel).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"error: file not found: {path!r}"
    except OSError as exc:
        return f"error: {exc}"


def execute_write_patch(
    path: str,
    content: str,
    project_root: Path,
    storage: Any = None,
    project_id: Optional[str] = None,
) -> Tuple[str, Optional[dict]]:
    """Create or overwrite a file; return (result_message, file_event_or_none)."""
    rel = _normalize_path(path)
    if rel is None:
        return "error: empty path", None
    if not _path_is_safe(rel):
        return f"error: path outside project root: {path!r}", None

    final = _ensure_use_client(rel, content)
    final = _truncate(final)

    try:
        if storage is not None and project_id is not None:
            storage.write_file(project_id, rel, final)
        else:
            target = project_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(final, encoding="utf-8")
        return f"wrote {rel}", {"path": rel, "content": final}
    except (ValueError, OSError) as exc:
        return f"error writing file: {exc}", None


def execute_shell_exec(command: str, cwd: Path, output_limit: int = 8192) -> str:
    """Run a shell command; return combined stdout+stderr (truncated)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        if len(output) > output_limit:
            output = (
                output[:output_limit]
                + f"\n[truncated at {output_limit} bytes]"
            )
        return output
    except subprocess.TimeoutExpired:
        return "error: command timed out after 60 seconds"
    except OSError as exc:
        return f"error: {exc}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def _read_file_wrapper(args: dict, project_root: Path) -> str:
    return execute_read_file(args.get("path", ""), project_root)


def _write_patch_wrapper(
    args: dict,
    project_root: Path,
    storage: Optional[object] = None,
    project_id: Optional[str] = None,
) -> tuple[str, Optional[dict]]:
    return execute_write_patch(
        args.get("path", ""),
        args.get("content", ""),
        project_root,
        storage,
        project_id,
    )


def _shell_exec_wrapper(args: dict, project_root: Path) -> str:
    return execute_shell_exec(
        args.get("command", ""),
        project_root,
    )


# Tool definitions
READ_FILE_TOOL = ToolDef(
    name="read_file",
    description="Read the current contents of a project file.",
    input_schema=ReadFileInput,
    execute=_read_file_wrapper,
)

WRITE_PATCH_TOOL = ToolDef(
    name="write_patch",
    description="Create or overwrite a file with the given full content.",
    input_schema=WritePatchInput,
    execute=_write_patch_wrapper,
)

SHELL_EXEC_TOOL = ToolDef(
    name="shell_exec",
    description=(
        "Run a shell command in the project directory. "
        "Requires explicit user approval before execution."
    ),
    input_schema=ShellExecInput,
    execute=_shell_exec_wrapper,
    requires_approval=True,
)

ALL_TOOLS: List[ToolDef] = [READ_FILE_TOOL, WRITE_PATCH_TOOL, SHELL_EXEC_TOOL]


def get_tool_by_name(name: str) -> Optional[ToolDef]:
    """Find a tool definition by name."""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


def tool_schemas_for_prompt() -> str:
    """Generate a tool description string for embedding in LLM prompts.

    Used when the LLM provider doesn't support native function calling
    (e.g., Hermes Agent via text interface).
    """
    lines: list[str] = ["Available tools:"]
    for tool in ALL_TOOLS:
        lines.append(f"\n  - {tool.name}: {tool.description}")
        schema = tool.parameters
        if "properties" in schema:
            props = schema["properties"]
            lines.append("    Parameters:")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                lines.append(f"      {pname} ({ptype}): {pdesc}")
    return "\n".join(lines)


def tool_call_format_instruction() -> str:
    """Return instruction text telling the LLM how to call tools.

    Compatible with text-based LLM interfaces (Hermes Agent, etc.).
    """
    return """
When you need to use a tool, respond with EXACTLY:

TOOL_CALL: <tool_name>
<json arguments>
END_TOOL_CALL

For example:
TOOL_CALL: read_file
{"path": "app/page.tsx"}
END_TOOL_CALL

When the task is complete, respond with:
FINAL: <your summary of what was done>

Do NOT mix tool calls and final text in the same response.
""".strip()
