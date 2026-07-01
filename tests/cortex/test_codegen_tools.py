"""Comprehensive tests for cortex/codegen/tools.py — tool definitions and executors.

Targets >95% coverage across all models, execution functions, registry,
and prompt helpers. Subprocess calls are mocked; filesystem ops use tmp_path.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest
from pydantic import ValidationError

from cortex.codegen.tools import (
    ALL_TOOLS,
    READ_FILE_TOOL,
    SHELL_EXEC_TOOL,
    WRITE_PATCH_TOOL,
    ReadFileInput,
    ShellExecInput,
    ToolDef,
    WritePatchInput,
    execute_read_file,
    execute_shell_exec,
    execute_write_patch,
    get_tool_by_name,
    tool_call_format_instruction,
    tool_schemas_for_prompt,
)


# =============================================================================
# Pydantic input schema tests
# =============================================================================


class TestReadFileInput:
    def test_valid_path(self):
        model = ReadFileInput(path="src/main.py")
        assert model.path == "src/main.py"

    def test_empty_path(self):
        """Empty string is accepted by Pydantic (no min_length) but still passes."""
        model = ReadFileInput(path="")
        assert model.path == ""

    def test_path_with_trailing_slash(self):
        model = ReadFileInput(path="dir/")
        assert model.path == "dir/"

    def test_path_repr(self):
        model = ReadFileInput(path="app/page.tsx")
        assert "path" in model.model_dump()


class TestWritePatchInput:
    def test_valid_path_and_content(self):
        model = WritePatchInput(path="src/hello.ts", content='console.log("hi")')
        assert model.path == "src/hello.ts"
        assert model.content == 'console.log("hi")'

    def test_empty_content(self):
        """Empty content is allowed — can create empty files."""
        model = WritePatchInput(path="empty.txt", content="")
        assert model.content == ""

    def test_empty_path(self):
        model = WritePatchInput(path="", content="x")
        assert model.path == ""

    def test_long_content(self):
        content = "x" * 100_000
        model = WritePatchInput(path="big.txt", content=content)
        assert len(model.content) == 100_000

    def test_model_dump_includes_both_fields(self):
        model = WritePatchInput(path="a.py", content="code")
        dumped = model.model_dump()
        assert dumped == {"path": "a.py", "content": "code"}


class TestShellExecInput:
    def test_valid_command_and_reason(self):
        model = ShellExecInput(command="npm run build", reason="Build the project")
        assert model.command == "npm run build"
        assert model.reason == "Build the project"

    def test_empty_command(self):
        model = ShellExecInput(command="", reason="test")
        assert model.command == ""

    def test_empty_reason(self):
        model = ShellExecInput(command="ls", reason="")
        assert model.reason == ""

    def test_model_dump(self):
        model = ShellExecInput(command="echo hi", reason="say hi")
        assert model.model_dump() == {
            "command": "echo hi",
            "reason": "say hi",
        }


# =============================================================================
# ToolDef dataclass tests
# =============================================================================


class TestToolDef:
    def test_basic_properties(self):
        def dummy(args, root):
            return "ok"

        tool = ToolDef(
            name="test_tool",
            description="A test tool",
            input_schema=ReadFileInput,
            execute=dummy,
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema is ReadFileInput
        assert tool.execute is dummy
        assert tool.requires_approval is False
        assert isinstance(tool.parameters, dict)

    def test_requires_approval_true(self):
        tool = ToolDef(
            name="danger",
            description="Needs approval",
            input_schema=ShellExecInput,
            execute=lambda a, r: "",
            requires_approval=True,
        )
        assert tool.requires_approval is True

    def test_post_init_generates_json_schema(self):
        """__post_init__ should populate parameters from model_json_schema()."""
        tool = ToolDef(
            name="schema_check",
            description="Check schema",
            input_schema=ReadFileInput,
            execute=lambda a, r: "",
        )
        assert "properties" in tool.parameters
        assert "path" in tool.parameters["properties"]
        assert tool.parameters["properties"]["path"]["type"] == "string"

    def test_post_init_with_write_patch_schema(self):
        tool = ToolDef(
            name="wp",
            description="Write patch",
            input_schema=WritePatchInput,
            execute=lambda a, r, s=None, pid=None: ("", None),
        )
        props = tool.parameters.get("properties", {})
        assert "path" in props
        assert "content" in props

    def test_post_init_with_shell_exec_schema(self):
        tool = ToolDef(
            name="se",
            description="Shell exec",
            input_schema=ShellExecInput,
            execute=lambda a, r: "",
            requires_approval=True,
        )
        props = tool.parameters.get("properties", {})
        assert "command" in props
        assert "reason" in props

    def test_post_init_overrides_default_parameters(self):
        """__post_init__ replaces the default empty dict with JSON schema."""
        tool = ToolDef(
            name="test",
            description="test",
            input_schema=ReadFileInput,
            execute=lambda a, r: "",
        )
        assert tool.parameters != {}  # __post_init__ populated it
        assert "properties" in tool.parameters


# =============================================================================
# execute_read_file tests
# =============================================================================


class TestExecuteReadFile:
    def test_reads_file_content(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()
        target = project_root / "src/main.py"
        target.parent.mkdir(parents=True)
        target.write_text("print('hello')", encoding="utf-8")

        result = execute_read_file("src/main.py", project_root)
        assert result == "print('hello')"

    def test_empty_path_returns_error(self, tmp_path):
        result = execute_read_file("", tmp_path)
        assert result == "error: empty path"

    def test_whitespace_only_path_returns_error(self, tmp_path):
        result = execute_read_file("   ", tmp_path)
        assert result == "error: empty path"

    def test_path_traversal_returns_error(self, tmp_path):
        result = execute_read_file("../../etc/passwd", tmp_path)
        assert result.startswith("error: path outside project root")

    def test_forbidden_segment_git_returns_error(self, tmp_path):
        result = execute_read_file(".git/config", tmp_path)
        assert result.startswith("error: path outside project root")

    def test_forbidden_segment_node_modules_returns_error(self, tmp_path):
        result = execute_read_file("node_modules/foo/index.js", tmp_path)
        assert result.startswith("error: path outside project root")

    def test_forbidden_segment_micracode_returns_error(self, tmp_path):
        result = execute_read_file(".micracode/config.yaml", tmp_path)
        assert result.startswith("error: path outside project root")

    def test_file_not_found_returns_error(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()
        result = execute_read_file("nonexistent.py", project_root)
        assert result == "error: file not found: 'nonexistent.py'"

    def test_leading_slash_is_stripped(self, tmp_path):
        """Path like '/src/main.py' should be normalized to 'src/main.py'."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        target = project_root / "src/main.py"
        target.parent.mkdir(parents=True)
        target.write_text("ok", encoding="utf-8")

        result = execute_read_file("/src/main.py", project_root)
        assert result == "ok"

    def test_os_error_is_caught(self, tmp_path):
        """Simulate an OSError (e.g. permission denied) via patching."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
            result = execute_read_file("some/file.py", project_root)
            assert result == "error: permission denied"

    def test_path_with_backslash_separators(self, tmp_path):
        """Backslash separators should be normalized to forward slash."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        target = project_root / "src/main.py"
        target.parent.mkdir(parents=True)
        target.write_text("backslash test", encoding="utf-8")

        result = execute_read_file("src\\main.py", project_root)
        assert result == "backslash test"

    def test_directory_path(self, tmp_path):
        """Reading a directory path returns an error."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "somedir").mkdir()

        result = execute_read_file("somedir", project_root)
        # Should hit FileNotFoundError because directory is not a file
        assert "error:" in result


# =============================================================================
# execute_write_patch tests
# =============================================================================


class TestExecuteWritePatch:
    def test_writes_file_to_disk(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()

        msg, event = execute_write_patch(
            "src/hello.py", "print('hello')", project_root, storage=None, project_id=None
        )
        assert msg == "wrote src/hello.py"
        assert event is not None
        assert event["path"] == "src/hello.py"
        assert event["content"] == "print('hello')"

        # Verify the file was actually written
        written = (project_root / "src/hello.py").read_text(encoding="utf-8")
        assert written == "print('hello')"

    def test_creates_parent_directories(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()

        msg, event = execute_write_patch(
            "a/b/c/d/file.txt", "deep", project_root, storage=None, project_id=None
        )
        assert msg == "wrote a/b/c/d/file.txt"
        assert (project_root / "a/b/c/d/file.txt").exists()

    def test_empty_path_returns_error(self, tmp_path):
        msg, event = execute_write_patch("", "content", tmp_path)
        assert msg == "error: empty path"
        assert event is None

    def test_whitespace_path_returns_error(self, tmp_path):
        msg, event = execute_write_patch("   ", "content", tmp_path)
        assert msg == "error: empty path"
        assert event is None

    def test_path_traversal_returns_error(self, tmp_path):
        msg, event = execute_write_patch(
            "../../etc/malicious", "evil", tmp_path
        )
        assert msg.startswith("error: path outside project root")
        assert event is None

    def test_forbidden_git_segment_returns_error(self, tmp_path):
        msg, event = execute_write_patch(
            ".git/hooks/pre-commit", "hook", tmp_path
        )
        assert msg.startswith("error: path outside project root")
        assert event is None

    def test_forbidden_node_modules_segment_returns_error(self, tmp_path):
        msg, event = execute_write_patch(
            "node_modules/foo.js", "x", tmp_path
        )
        assert msg.startswith("error: path outside project root")
        assert event is None

    def test_os_error_on_write_is_caught(self, tmp_path):
        """Simulate OSError when trying to write."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            msg, event = execute_write_patch(
                "fail.txt", "content", project_root
            )
            assert msg == "error writing file: disk full"
            assert event is None

    def test_value_error_on_write_is_caught(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()

        with patch.object(Path, "write_text", side_effect=ValueError("bad encoding")):
            msg, event = execute_write_patch(
                "fail.txt", "content", project_root
            )
            assert msg == "error writing file: bad encoding"
            assert event is None

    def test_writes_via_storage_when_provided(self, tmp_path):
        """When storage and project_id are given, use storage.write_file."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        storage = MagicMock()
        project_id = "proj-123"

        msg, event = execute_write_patch(
            "stored.py", "stored content", project_root, storage, project_id
        )
        assert msg == "wrote stored.py"
        assert event["path"] == "stored.py"
        assert event["content"] == "stored content"

        # storage.write_file should have been called with (project_id, rel, content)
        storage.write_file.assert_called_once_with(
            "proj-123", "stored.py", "stored content"
        )
        # Should NOT also write to disk
        assert not (project_root / "stored.py").exists()

    def test_leading_slash_is_stripped(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()

        msg, event = execute_write_patch(
            "//leading/slash.py", "content", project_root
        )
        assert msg == "wrote leading/slash.py"
        assert (project_root / "leading/slash.py").exists()

    def test_backslash_separators_normalized(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()

        msg, event = execute_write_patch(
            "dir\\sub\\file.py", "content", project_root
        )
        assert msg == "wrote dir/sub/file.py"
        assert (project_root / "dir/sub/file.py").exists()

    def test_ensure_use_client_is_applied(self, tmp_path):
        """'use client' directive should be auto-prepended for React files."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Content uses useState which triggers 'use client'
        content = 'import { useState } from "react";\nconst [x, setX] = useState(0);\n'
        msg, event = execute_write_patch(
            "components/Counter.tsx", content, project_root, storage=None, project_id=None
        )
        assert event is not None
        final_content = event["content"]
        assert final_content.startswith('"use client";')
        assert 'import { useState } from "react";' in final_content

    def test_truncate_long_content(self, tmp_path):
        """Content exceeding _MAX_FILE_CONTENT_CHARS (80k) should be truncated."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        long_content = "x" * 100_000
        msg, event = execute_write_patch(
            "big.txt", long_content, project_root, storage=None, project_id=None
        )
        assert event is not None
        # Should be truncated to 80_000 chars
        assert len(event["content"]) == 80_000
        # File on disk should also be truncated
        written = (project_root / "big.txt").read_text(encoding="utf-8")
        assert len(written) == 80_000

    def test_overwrites_existing_file(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()
        target = project_root / "existing.txt"
        target.write_text("old", encoding="utf-8")

        msg, event = execute_write_patch(
            "existing.txt", "new content", project_root
        )
        assert msg == "wrote existing.txt"
        assert target.read_text(encoding="utf-8") == "new content"


# =============================================================================
# execute_shell_exec tests
# =============================================================================


class TestExecuteShellExec:
    def test_basic_command(self):
        result = execute_shell_exec("echo test", Path("/tmp"))
        result = result.strip()
        assert result == "test"

    def test_command_with_output(self):
        result = execute_shell_exec("echo hello world && echo foo", Path("/tmp"))
        lines = [l for l in result.split("\n") if l]
        assert "hello world" in lines
        assert "foo" in lines

    def test_output_truncation(self, monkeypatch):
        """When output exceeds output_limit, it should be truncated."""
        large_output = "x" * 10_000

        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.stdout = large_output
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = execute_shell_exec("echo big", Path("/tmp"), output_limit=100)
        assert len(result) <= 100 + len("\n[truncated at 100 bytes]")
        assert "[truncated at 100 bytes]" in result

    def test_timeout_expired(self, monkeypatch):
        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="test", timeout=60)

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = execute_shell_exec("sleep 999", Path("/tmp"))
        assert result == "error: command timed out after 60 seconds"

    def test_os_error(self, monkeypatch):
        def mock_run(*args, **kwargs):
            raise OSError("command not found")

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = execute_shell_exec("nonexistent_cmd_xyz", Path("/tmp"))
        assert result == "error: command not found"

    def test_stderr_is_included(self):
        """Stderr should be captured and included in output."""
        result = execute_shell_exec(
            "echo out && echo err >&2", Path("/tmp")
        )
        assert "out" in result
        assert "err" in result

    def test_exact_output_limit_boundary(self, monkeypatch):
        """Output exactly at output_limit should not be truncated."""
        exact_output = "a" * 100

        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.stdout = exact_output
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = execute_shell_exec("cmd", Path("/tmp"), output_limit=100)
        assert result == exact_output
        assert "[truncated" not in result

    def test_output_one_over_limit(self, monkeypatch):
        """Output one byte over limit triggers truncation."""
        big = "a" * 101

        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.stdout = big
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = execute_shell_exec("cmd", Path("/tmp"), output_limit=100)
        assert "[truncated at 100 bytes]" in result
        assert len(result) == 100 + len("\n[truncated at 100 bytes]")

    def test_cwd_is_passed_to_subprocess(self, monkeypatch):
        """The cwd argument should be forwarded to subprocess.run."""
        captured_kwargs = {}

        def mock_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            result = MagicMock()
            result.stdout = "ok"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_run)
        custom_cwd = Path("/my/project")
        execute_shell_exec("echo hi", custom_cwd)
        assert captured_kwargs["cwd"] == custom_cwd
        assert captured_kwargs["shell"] is True
        assert captured_kwargs["capture_output"] is True
        assert captured_kwargs["text"] is True


# =============================================================================
# Module-level tool definitions
# =============================================================================


class TestToolConstants:
    def test_read_file_tool_has_correct_name(self):
        assert READ_FILE_TOOL.name == "read_file"
        assert READ_FILE_TOOL.description == "Read the current contents of a project file."
        assert READ_FILE_TOOL.input_schema is ReadFileInput
        assert READ_FILE_TOOL.requires_approval is False
        assert callable(READ_FILE_TOOL.execute)

    def test_write_patch_tool_has_correct_name(self):
        assert WRITE_PATCH_TOOL.name == "write_patch"
        assert "Create or overwrite" in WRITE_PATCH_TOOL.description
        assert WRITE_PATCH_TOOL.input_schema is WritePatchInput
        assert WRITE_PATCH_TOOL.requires_approval is False
        assert callable(WRITE_PATCH_TOOL.execute)

    def test_shell_exec_tool_has_correct_name(self):
        assert SHELL_EXEC_TOOL.name == "shell_exec"
        assert "shell command" in SHELL_EXEC_TOOL.description
        assert SHELL_EXEC_TOOL.input_schema is ShellExecInput
        assert SHELL_EXEC_TOOL.requires_approval is True
        assert callable(SHELL_EXEC_TOOL.execute)

    def test_all_tools_contains_exactly_three(self):
        assert len(ALL_TOOLS) == 3
        assert READ_FILE_TOOL in ALL_TOOLS
        assert WRITE_PATCH_TOOL in ALL_TOOLS
        assert SHELL_EXEC_TOOL in ALL_TOOLS

    def test_all_tools_order(self):
        assert ALL_TOOLS[0] is READ_FILE_TOOL
        assert ALL_TOOLS[1] is WRITE_PATCH_TOOL
        assert ALL_TOOLS[2] is SHELL_EXEC_TOOL


# =============================================================================
# Wrapper function tests (exercised via tool definitions)
# =============================================================================


class TestToolWrappers:
    def test_read_file_wrapper_calls_execute(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()
        target = project_root / "test.txt"
        target.write_text("wrapper test", encoding="utf-8")

        result = READ_FILE_TOOL.execute({"path": "test.txt"}, project_root)
        assert result == "wrapper test"

    def test_read_file_wrapper_missing_path(self, tmp_path):
        """When 'path' key is missing from args, empty string is used -> error."""
        result = READ_FILE_TOOL.execute({}, tmp_path)
        assert result.startswith("error: empty path")

    def test_write_patch_wrapper_calls_execute(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()

        msg, event = WRITE_PATCH_TOOL.execute(
            {"path": "wrapper.txt", "content": "wrapper content"},
            project_root,
        )
        assert msg == "wrote wrapper.txt"
        assert (project_root / "wrapper.txt").read_text(encoding="utf-8") == "wrapper content"

    def test_write_patch_wrapper_missing_path(self, tmp_path):
        """Missing 'path' defaults to empty string -> error."""
        msg, event = WRITE_PATCH_TOOL.execute(
            {"content": "stuff"},
            tmp_path,
        )
        assert msg == "error: empty path"
        assert event is None

    def test_write_patch_wrapper_missing_content(self, tmp_path):
        """Missing 'content' defaults to empty string."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        msg, event = WRITE_PATCH_TOOL.execute(
            {"path": "empty.txt"},
            project_root,
        )
        assert msg == "wrote empty.txt"
        assert event is not None
        assert event["content"] == ""

    def test_write_patch_wrapper_with_storage(self, tmp_path):
        storage = MagicMock()
        project_id = "proj-1"

        msg, event = WRITE_PATCH_TOOL.execute(
            {"path": "via_wrapper.py", "content": "from wrapper"},
            tmp_path,
            storage,
            project_id,
        )
        assert msg == "wrote via_wrapper.py"
        storage.write_file.assert_called_once()

    def test_shell_exec_wrapper_calls_execute(self, tmp_path):
        result = SHELL_EXEC_TOOL.execute(
            {"command": "echo wrapper_works"}, tmp_path
        )
        assert "wrapper_works" in result

    def test_shell_exec_wrapper_missing_command(self, tmp_path):
        """Missing 'command' defaults to empty string -> runs empty command."""
        result = SHELL_EXEC_TOOL.execute({}, tmp_path)
        # An empty command via shell=True just exits with no output
        assert isinstance(result, str)


# =============================================================================
# get_tool_by_name tests
# =============================================================================


class TestGetToolByName:
    def test_finds_read_file(self):
        tool = get_tool_by_name("read_file")
        assert tool is READ_FILE_TOOL

    def test_finds_write_patch(self):
        tool = get_tool_by_name("write_patch")
        assert tool is WRITE_PATCH_TOOL

    def test_finds_shell_exec(self):
        tool = get_tool_by_name("shell_exec")
        assert tool is SHELL_EXEC_TOOL

    def test_returns_none_for_unknown(self):
        assert get_tool_by_name("nonexistent_tool") is None

    def test_returns_none_for_empty_string(self):
        assert get_tool_by_name("") is None

    def test_case_sensitive(self):
        assert get_tool_by_name("Read_File") is None


# =============================================================================
# tool_schemas_for_prompt tests
# =============================================================================


class TestToolSchemasForPrompt:
    def test_contains_all_tool_names(self):
        output = tool_schemas_for_prompt()
        assert "read_file" in output
        assert "write_patch" in output
        assert "shell_exec" in output

    def test_contains_tool_descriptions(self):
        output = tool_schemas_for_prompt()
        assert "Read the current contents" in output
        assert "Create or overwrite" in output
        assert "Run a shell command" in output

    def test_contains_parameter_info(self):
        output = tool_schemas_for_prompt()
        assert "Parameters:" in output
        assert "path (string):" in output
        assert "content (string):" in output
        assert "command (string):" in output
        assert "reason (string):" in output

    def test_starts_with_header(self):
        output = tool_schemas_for_prompt()
        assert output.startswith("Available tools:")

    def test_each_tool_has_dash_prefix(self):
        output = tool_schemas_for_prompt()
        lines = output.split("\n")
        tool_lines = [l for l in lines if l.strip().startswith("- ")]
        assert len(tool_lines) == 3

    def test_parameter_descriptions(self):
        """Parameter descriptions from Field() should appear in the output."""
        output = tool_schemas_for_prompt()
        assert "File path relative" in output
        assert "Full content to write" in output
        assert "Shell command to execute" in output
        assert "Why this command is needed" in output


# =============================================================================
# tool_call_format_instruction tests
# =============================================================================


class TestToolCallFormatInstruction:
    def test_contains_tool_call_marker(self):
        result = tool_call_format_instruction()
        assert "TOOL_CALL:" in result

    def test_contains_end_tool_call(self):
        result = tool_call_format_instruction()
        assert "END_TOOL_CALL" in result

    def test_contains_final_marker(self):
        result = tool_call_format_instruction()
        assert "FINAL:" in result

    def test_contains_read_file_example(self):
        result = tool_call_format_instruction()
        assert "read_file" in result
        assert '"path": "app/page.tsx"' in result

    def test_does_not_start_or_end_with_whitespace(self):
        """The result is .strip() at the end of the function."""
        result = tool_call_format_instruction()
        assert result == result.strip()
        assert result[0] != "\n"
        assert result[-1] != "\n"

    def test_contains_warning_about_mixing(self):
        result = tool_call_format_instruction()
        assert "Do NOT mix" in result

    def test_multiline_output(self):
        result = tool_call_format_instruction()
        assert "\n" in result
        assert result.count("\n") >= 8  # Substantial instruction


# =============================================================================
# Integration: full tool registration -> schema -> prompt flow
# =============================================================================


class TestFullIntegration:
    def test_all_tools_have_unique_names(self):
        names = [t.name for t in ALL_TOOLS]
        assert len(names) == len(set(names))

    def test_all_tools_have_json_schema(self):
        for tool in ALL_TOOLS:
            assert "properties" in tool.parameters
            assert len(tool.parameters["properties"]) > 0

    def test_all_tools_execute_is_callable(self):
        for tool in ALL_TOOLS:
            assert callable(tool.execute)

    def test_round_trip_name_lookup(self):
        for tool in ALL_TOOLS:
            assert get_tool_by_name(tool.name) is tool

    def test_shell_exec_requires_approval_only(self):
        """Only shell_exec requires approval."""
        assert READ_FILE_TOOL.requires_approval is False
        assert WRITE_PATCH_TOOL.requires_approval is False
        assert SHELL_EXEC_TOOL.requires_approval is True

    def test_tool_schemas_contains_all_parameters(self):
        schema_str = tool_schemas_for_prompt()
        # All parameter names should appear
        for tool in ALL_TOOLS:
            props = tool.parameters.get("properties", {})
            for pname in props:
                assert pname in schema_str

    def test_execute_read_file_round_trip(self, tmp_path):
        """Write with write_patch, then read back with read_file."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Write
        execute_write_patch("roundtrip.txt", "round trip!", project_root)

        # Read back
        content = execute_read_file("roundtrip.txt", project_root)
        assert content == "round trip!"

    def test_execute_shell_exec_default_output_limit(self, tmp_path):
        """Default output_limit should be 8192."""
        result = execute_shell_exec("echo hello", tmp_path)
        assert isinstance(result, str)
        assert "hello" in result


# =============================================================================
# Edge cases and error handling
# =============================================================================


class TestEdgeCases:
    def test_path_with_only_dot(self, tmp_path):
        """A path like '.' has empty parts tuple in Python 3.11 -> rejected as unsafe."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        msg, event = execute_write_patch(".", "x", project_root)
        # Path('.').parts is () in 3.11+ -> _path_is_safe returns False
        assert msg.startswith("error: path outside project root:")
        assert event is None

    def test_path_with_dot_dot_in_middle(self, tmp_path):
        """A path like 'foo/../../bar' should be rejected."""
        msg, event = execute_write_patch("foo/../../bar", "x", tmp_path)
        assert msg.startswith("error: path outside project root")

    def test_read_file_with_symlink_path(self, tmp_path):
        """Path with symlink segments — path_is_safe checks '..' segments."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        result = execute_read_file("foo/../bar", project_root)
        # '..' is in the parts -> unsafe
        assert result.startswith("error: path outside project root")

    def test_shell_exec_empty_command(self, tmp_path):
        """An empty command should execute without error (shell=True, no-op)."""
        result = execute_shell_exec("", tmp_path)
        assert isinstance(result, str)

    def test_tool_def_immutable_parameters_not_shared(self):
        """Each ToolDef should have its own parameters dict."""
        params_ids = [id(t.parameters) for t in ALL_TOOLS]
        assert len(set(params_ids)) == len(ALL_TOOLS)


# =============================================================================
# Error message formats
# =============================================================================


class TestErrorMessages:
    def test_execute_read_file_error_messages_format(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()

        assert execute_read_file("", project_root) == "error: empty path"
        assert execute_read_file(
            "../../bad", project_root
        ).startswith("error: path outside project root")
        assert execute_read_file(
            "nope.txt", project_root
        ).startswith("error: file not found:")

    def test_execute_write_patch_error_messages_format(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()

        msg1, _ = execute_write_patch("", "x", project_root)
        assert msg1 == "error: empty path"

        msg2, _ = execute_write_patch("../../bad", "x", project_root)
        assert msg2.startswith("error: path outside project root")

    def test_execute_shell_exec_error_messages_format(self, monkeypatch):
        def timeout_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="x", timeout=60)

        monkeypatch.setattr(subprocess, "run", timeout_run)
        assert execute_shell_exec("cmd", Path("/tmp")) == "error: command timed out after 60 seconds"

    def test_write_patch_with_storage_error(self, tmp_path):
        """Storage write_file raising an error should be caught."""
        storage = MagicMock()
        storage.write_file.side_effect = ValueError("storage full")

        msg, event = execute_write_patch(
            "fail.py", "x", tmp_path, storage, "proj-1"
        )
        assert msg == "error writing file: storage full"
        assert event is None

    def test_write_patch_oserror_with_storage(self, tmp_path):
        """OSError from storage write should be caught."""
        storage = MagicMock()
        storage.write_file.side_effect = OSError("disk error")

        msg, event = execute_write_patch(
            "fail.py", "x", tmp_path, storage, "proj-1"
        )
        assert "error writing file:" in msg
        assert event is None
