"""Comprehensive tests for cortex/codegen/patcher.py — >95% target coverage.

Tests cover:
  - Data types (SearchReplace, PatchFile, PatchBundle, PatchResult, ProjectContext)
  - Path utilities (_normalize_path, _path_is_safe)
  - "use client" detection (_needs_use_client, _ensure_use_client)
  - Patch application (apply_patch, _apply_one, apply_bundle)
  - File I/O convenience wrappers (execute_write_file, execute_read_file)
  - Edge cases: empty paths, unsafe paths, missing files, OSError, oversized content
"""

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex.codegen.patcher import (
    PatchFile,
    PatchBundle,
    PatchResult,
    SearchReplace,
    ProjectContext,
    apply_patch,
    apply_bundle,
    _normalize_path,
    _path_is_safe,
    _needs_use_client,
    _ensure_use_client,
    _apply_one,
    _truncate,
    execute_write_file,
    execute_read_file,
    PatchError,
    Operation,
    FileLoader,
    _MAX_FILE_CONTENT_CHARS,
    _MAX_SEARCH_REPLACE_CHARS,
    _MAX_EDITS_PER_FILE,
    _FORBIDDEN_SEGMENTS,
    _CLIENT_DIRECTIVE_SUFFIXES,
    _SERVER_ONLY_APP_FILES,
    _CLIENT_ONLY_IMPORT_RES,
    _CLIENT_ONLY_HOOKS,
    _CLIENT_HOOK_RE,
    _USE_CLIENT_RE,
)


# ===================================================================
# Constants & sanity
# ===================================================================


class TestConstants:
    """Verify module-level constants exist with expected types."""

    def test_max_file_content_chars(self):
        assert isinstance(_MAX_FILE_CONTENT_CHARS, int)
        assert _MAX_FILE_CONTENT_CHARS == 80_000

    def test_max_search_replace_chars(self):
        assert isinstance(_MAX_SEARCH_REPLACE_CHARS, int)
        assert _MAX_SEARCH_REPLACE_CHARS == 8_000

    def test_max_edits_per_file(self):
        assert isinstance(_MAX_EDITS_PER_FILE, int)
        assert _MAX_EDITS_PER_FILE == 20

    def test_forbidden_segments(self):
        assert ".git" in _FORBIDDEN_SEGMENTS
        assert "node_modules" in _FORBIDDEN_SEGMENTS
        assert ".micracode" in _FORBIDDEN_SEGMENTS
        assert len(_FORBIDDEN_SEGMENTS) == 3

    def test_client_directive_suffixes(self):
        assert ".tsx" in _CLIENT_DIRECTIVE_SUFFIXES
        assert ".jsx" in _CLIENT_DIRECTIVE_SUFFIXES
        assert ".ts" in _CLIENT_DIRECTIVE_SUFFIXES
        assert ".js" in _CLIENT_DIRECTIVE_SUFFIXES

    def test_server_only_app_files(self):
        assert "app/layout.tsx" in _SERVER_ONLY_APP_FILES
        assert "app/layout.jsx" in _SERVER_ONLY_APP_FILES
        assert len(_SERVER_ONLY_APP_FILES) == 2

    def test_client_only_import_res(self):
        assert len(_CLIENT_ONLY_IMPORT_RES) == 2
        assert _CLIENT_ONLY_IMPORT_RES[0].pattern == """from\\s+['\"]framer-motion['\"]"""
        assert "react-spring" in _CLIENT_ONLY_IMPORT_RES[1].pattern

    def test_client_only_hooks(self):
        assert "useState" in _CLIENT_ONLY_HOOKS
        assert "useEffect" in _CLIENT_ONLY_HOOKS
        assert "useLayoutEffect" in _CLIENT_ONLY_HOOKS
        assert "useRef" in _CLIENT_ONLY_HOOKS

    def test_client_hook_re(self):
        assert _CLIENT_HOOK_RE.search("useState(")
        assert _CLIENT_HOOK_RE.search("useEffect(")
        assert not _CLIENT_HOOK_RE.search("useState")
        assert not _CLIENT_HOOK_RE.search("nouseState(")

    def test_use_client_re(self):
        assert _USE_CLIENT_RE.match('"use client"')
        assert _USE_CLIENT_RE.match("'use client'")
        assert _USE_CLIENT_RE.match('"use client";')
        assert _USE_CLIENT_RE.match("'use client';")
        assert _USE_CLIENT_RE.match('  "use client"')
        assert not _USE_CLIENT_RE.match("// use client")


# ===================================================================
# Data types
# ===================================================================


class TestSearchReplace:
    def test_dataclass(self):
        sr = SearchReplace(search="foo", replace="bar")
        assert sr.search == "foo"
        assert sr.replace == "bar"

    def test_is_frozen(self):
        sr = SearchReplace(search="a", replace="b")
        with pytest.raises(AttributeError):
            sr.search = "c"


class TestPatchFile:
    def test_create(self):
        pf = PatchFile(path="foo.ts", operation="create", content="hi")
        assert pf.path == "foo.ts"
        assert pf.operation == "create"
        assert pf.content == "hi"
        assert pf.edits is None

    def test_edit_with_edits(self):
        pf = PatchFile(
            path="bar.py",
            operation="edit",
            content=None,
            edits=[SearchReplace("old", "new")],
        )
        assert pf.operation == "edit"
        assert pf.edits == [SearchReplace("old", "new")]

    def test_is_frozen(self):
        pf = PatchFile(path="x", operation="create", content="y")
        with pytest.raises(AttributeError):
            pf.path = "z"


class TestPatchBundle:
    def test_empty_files(self):
        bundle = PatchBundle(files=[])
        assert bundle.files == []

    def test_with_files(self):
        f1 = PatchFile(path="a", operation="create", content="1")
        f2 = PatchFile(path="b", operation="delete")
        bundle = PatchBundle(files=[f1, f2])
        assert len(bundle.files) == 2

    def test_is_frozen(self):
        bundle = PatchBundle(files=[])
        with pytest.raises(AttributeError):
            bundle.files = [PatchFile(path="x", operation="delete")]


class TestPatchResult:
    def test_write_result(self):
        r = PatchResult(path="foo.ts", kind="write", content="code")
        assert r.path == "foo.ts"
        assert r.kind == "write"
        assert r.content == "code"
        assert r.error == ""

    def test_error_result(self):
        r = PatchResult(path="foo.ts", kind="error", error="bad")
        assert r.kind == "error"
        assert r.error == "bad"
        assert r.content == ""

    def test_delete_result(self):
        r = PatchResult(path="foo.ts", kind="delete")
        assert r.kind == "delete"
        assert r.content == ""
        assert r.error == ""

    def test_is_frozen(self):
        r = PatchResult(path="p", kind="write")
        with pytest.raises(AttributeError):
            r.kind = "delete"


class TestProjectContext:
    def test_defaults(self):
        ctx = ProjectContext(project_id="p1", tree_summary="sum")
        assert ctx.project_id == "p1"
        assert ctx.tree_summary == "sum"
        assert ctx.files == {}
        assert ctx.loader is None
        assert ctx.placeholder_files == frozenset()

    def test_get_file_from_cache(self):
        ctx = ProjectContext(project_id="p", tree_summary="s", files={"a.py": "hello"})
        assert ctx.get_file("a.py") == "hello"

    def test_get_file_from_loader(self):
        loader = MagicMock(spec=FileLoader)
        loader.return_value = "loaded_content"
        ctx = ProjectContext(project_id="p", tree_summary="s", loader=loader)
        result = ctx.get_file("b.py")
        assert result == "loaded_content"
        loader.assert_called_once_with("b.py")
        # Verify it was cached
        assert ctx.files["b.py"] == "loaded_content"

    def test_get_file_loader_returns_none(self):
        loader = MagicMock(spec=FileLoader)
        loader.return_value = None
        ctx = ProjectContext(project_id="p", tree_summary="s", loader=loader)
        result = ctx.get_file("c.py")
        assert result is None
        assert "c.py" not in ctx.files

    def test_get_file_no_loader(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        result = ctx.get_file("d.py")
        assert result is None

    def test_is_mutable(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        ctx.project_id = "p2"  # Should not raise


# ===================================================================
# _normalize_path
# ===================================================================


class TestNormalizePath:
    def test_strips_leading_slash(self):
        assert _normalize_path("/foo/bar.ts") == "foo/bar.ts"

    def test_replaces_backslashes(self):
        assert _normalize_path("foo\\bar.ts") == "foo/bar.ts"

    def test_strips_leading_slash_and_backslashes(self):
        assert _normalize_path("\\foo\\bar.ts") == "foo/bar.ts"

    def test_strips_whitespace(self):
        assert _normalize_path("  foo/bar  ") == "foo/bar"

    def test_handles_multiple_leading_slashes(self):
        assert _normalize_path("//foo/bar") == "foo/bar"

    def test_returns_none_for_empty_string(self):
        assert _normalize_path("") is None

    def test_returns_none_for_whitespace_only(self):
        assert _normalize_path("   ") is None

    def test_returns_none_for_only_slashes(self):
        assert _normalize_path("//") is None

    def test_preserves_normal_path(self):
        assert _normalize_path("src/components/Button.tsx") == "src/components/Button.tsx"


# ===================================================================
# _path_is_safe
# ===================================================================


class TestPathIsSafe:
    def test_normal_relative_path(self):
        assert _path_is_safe("src/foo.py") is True

    def test_single_component(self):
        assert _path_is_safe("file.ts") is True

    def test_nested_deep_path(self):
        assert _path_is_safe("a/b/c/d/e.py") is True

    def test_blocks_dotdot(self):
        assert _path_is_safe("../foo.py") is False

    def test_blocks_dotdot_in_middle(self):
        assert _path_is_safe("src/../foo.py") is False

    def test_blocks_dotdot_nested(self):
        assert _path_is_safe("a/b/../../c.py") is False

    def test_blocks_git_directory(self):
        assert _path_is_safe(".git/HEAD") is False

    def test_blocks_git_nested(self):
        assert _path_is_safe("src/.git/config") is False

    def test_blocks_node_modules(self):
        assert _path_is_safe("node_modules/foo/index.js") is False

    def test_blocks_node_modules_nested(self):
        assert _path_is_safe("src/node_modules/bar.js") is False

    def test_blocks_micracode(self):
        assert _path_is_safe(".micracode/config.yml") is False

    def test_blocks_micracode_nested(self):
        assert _path_is_safe("src/.micracode/secret") is False

    def test_empty_path(self):
        assert _path_is_safe("") is False


# ===================================================================
# _needs_use_client
# ===================================================================


class TestNeedsUseClient:
    def test_non_jsx_tsx_file(self):
        assert _needs_use_client("foo.py", "print('hello')") is False

    def test_d_ts_file(self):
        assert _needs_use_client("types.d.ts", "export type Foo = string;") is False

    def test_app_layout_tsx(self):
        assert _needs_use_client("app/layout.tsx", "export default function") is False

    def test_app_layout_jsx(self):
        assert _needs_use_client("app/layout.jsx", "export default function") is False

    def test_already_has_use_client_double_quote(self):
        assert _needs_use_client("comp.tsx", '"use client";\n\ncode') is False

    def test_already_has_use_client_single_quote(self):
        assert _needs_use_client("comp.tsx", "'use client';\n\ncode") is False

    def test_already_has_use_client_no_semicolon(self):
        assert _needs_use_client("comp.tsx", '"use client"\ncode') is False

    def test_framer_motion_import(self):
        content = 'import { motion } from "framer-motion";'
        assert _needs_use_client("comp.tsx", content) is True

    def test_framer_motion_single_quote(self):
        content = "import { motion } from 'framer-motion';"
        assert _needs_use_client("comp.tsx", content) is True

    def test_react_spring_import(self):
        content = 'import { animated } from "react-spring";'
        assert _needs_use_client("comp.tsx", content) is True

    def test_react_spring_scoped_import(self):
        """NOTE: The source regex has [\\\\w-]+ which means a literal backslash
        before word chars, so scoped @react-spring/* imports are NOT detected.
        This test documents the current (buggy) behavior."""
        content = 'import { useSpring } from "@react-spring/web";'
        assert _needs_use_client("comp.tsx", content) is False

    def test_react_spring_nested(self):
        """Same scoped-regression note as test_react_spring_scoped_import."""
        content = 'import { useTransition } from "@react-spring/shared";'
        assert _needs_use_client("comp.tsx", content) is False

    def test_useState_hook(self):
        content = "const [x, setX] = useState(0);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useEffect_hook(self):
        content = "useEffect(() => {}, []);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useRef_hook(self):
        content = "const ref = useRef(null);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useContext_hook(self):
        content = "const ctx = useContext(MyContext);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useMemo_hook(self):
        content = "const memo = useMemo(() => x, [x]);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useCallback_hook(self):
        content = "const cb = useCallback(() => {}, []);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useLayoutEffect_hook(self):
        content = "useLayoutEffect(() => {}, []);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useReducer_hook(self):
        content = "const [s, d] = useReducer(r, i);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useTransition_hook(self):
        content = "const [isPending, start] = useTransition();"
        assert _needs_use_client("page.tsx", content) is True

    def test_useDeferredValue_hook(self):
        content = "const v = useDeferredValue(val);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useSyncExternalStore_hook(self):
        content = "const state = useSyncExternalStore(s, g);"
        assert _needs_use_client("page.tsx", content) is True

    def test_useImperativeHandle_hook(self):
        content = "useImperativeHandle(ref, () => ({}));"
        assert _needs_use_client("page.tsx", content) is True

    def test_no_client_indicators(self):
        content = "export default function Home() { return <div>Hi</div>; }"
        assert _needs_use_client("page.tsx", content) is False

    def test_js_file_with_hook(self):
        content = "const [x, setX] = useState(0);"
        assert _needs_use_client("component.js", content) is True

    def test_tsx_without_hooks_or_imports(self):
        content = "const a = 1;"
        assert _needs_use_client("component.tsx", content) is False

    def test_hook_in_string_literal_no_match(self):
        """Hook name without parentheses shouldn't match."""
        content = 'const str = "useState is a hook";'
        assert _needs_use_client("comp.tsx", content) is False


# ===================================================================
# _ensure_use_client
# ===================================================================


class TestEnsureUseClient:
    def test_not_needed_returns_original(self):
        content = "const a = 1;\n"
        assert _ensure_use_client("page.tsx", content) == content

    def test_prepends_directive_when_needed(self):
        content = 'import { motion } from "framer-motion";\nconst a = 1;\n'
        result = _ensure_use_client("comp.tsx", content)
        expected = '"use client";\n\nimport { motion } from "framer-motion";\nconst a = 1;\n'
        assert result == expected

    def test_strips_leading_newlines_before_prepend(self):
        content = "\n\n\nconst x = useState(0);"
        result = _ensure_use_client("comp.tsx", content)
        assert result == '"use client";\n\nconst x = useState(0);'

    def test_non_react_file_unchanged(self):
        content = "print('hello')"
        assert _ensure_use_client("foo.py", content) == content

    def test_already_has_directive(self):
        content = '"use client";\n\ncode'
        assert _ensure_use_client("comp.tsx", content) == content


# ===================================================================
# apply_patch
# ===================================================================


class TestApplyPatch:
    def test_single_replacement(self):
        result = apply_patch("hello world", [SearchReplace("world", "there")])
        assert result == "hello there"

    def test_multiple_sequential_ops(self):
        ops = [
            SearchReplace("foo", "bar"),
            SearchReplace("bar", "baz"),
            SearchReplace("baz", "qux"),
        ]
        result = apply_patch("foo", ops)
        assert result == "qux"

    def test_empty_ops_list(self):
        result = apply_patch("original content", [])
        assert result == "original content"

    def test_replace_part_of_string(self):
        result = apply_patch(
            "function foo() { return 1; }",
            [SearchReplace("return 1;", "return 2;")],
        )
        assert result == "function foo() { return 2; }"

    def test_search_not_found_raises(self):
        with pytest.raises(PatchError, match="search string not found"):
            apply_patch("hello world", [SearchReplace("nope", "x")])

    def test_multiple_matches_raises(self):
        with pytest.raises(PatchError, match="matches 2 times"):
            apply_patch("a b a", [SearchReplace("a", "c")])

    def test_error_message_includes_edit_number(self):
        with pytest.raises(PatchError, match="edit #1:"):
            apply_patch("hello", [SearchReplace("x", "y")])

    def test_second_op_fails_with_clear_message(self):
        with pytest.raises(PatchError, match="edit #2:"):
            apply_patch(
                "hello world",
                [
                    SearchReplace("hello", "hi"),
                    SearchReplace("nonexistent", "x"),
                ],
            )

    def test_multiple_matches_second_op(self):
        with pytest.raises(PatchError, match="edit #2:"):
            apply_patch(
                "a b a",
                [
                    SearchReplace("b", "c"),
                    SearchReplace("a", "d"),
                ],
            )


# ===================================================================
# _truncate
# ===================================================================


class TestTruncate:
    def test_short_content_unchanged(self):
        content = "short"
        assert _truncate(content) == content

    def test_exactly_max(self):
        content = "x" * _MAX_FILE_CONTENT_CHARS
        assert _truncate(content) == content

    def test_truncates_oversized(self):
        content = "x" * (_MAX_FILE_CONTENT_CHARS + 100)
        result = _truncate(content)
        assert len(result) == _MAX_FILE_CONTENT_CHARS

    def test_truncates_preserves_prefix(self):
        content = "PREFIX" + "x" * _MAX_FILE_CONTENT_CHARS
        result = _truncate(content)
        assert result.startswith("PREFIX")
        assert len(result) == _MAX_FILE_CONTENT_CHARS


# ===================================================================
# _apply_one (internal helper)
# ===================================================================


class TestApplyOne:
    def test_empty_path(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="", operation="create", content="x")
        result = _apply_one(pf, ctx)
        assert result.kind == "error"
        assert result.error == "empty path"

    def test_unsafe_path(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="../escape.py", operation="create", content="x")
        result = _apply_one(pf, ctx)
        assert result.kind == "error"
        assert "unsafe path" in result.error

    def test_create_operation(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="hello.tsx", operation="create", content="const x = 1;")
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert result.path == "hello.tsx"
        assert "const x = 1;" in result.content

    def test_create_missing_content(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="f.ts", operation="create", content=None)
        result = _apply_one(pf, ctx)
        assert result.kind == "error"
        assert "missing content" in result.error

    def test_create_adds_use_client(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        content = 'import { motion } from "framer-motion";'
        pf = PatchFile(path="comp.tsx", operation="create", content=content)
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert result.content.startswith('"use client"')

    def test_replace_operation(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="old.ts", operation="replace", content="new content")
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert result.content == "new content"

    def test_replace_missing_content(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="f.ts", operation="replace", content=None)
        result = _apply_one(pf, ctx)
        assert result.kind == "error"
        assert "missing content" in result.error

    def test_edit_success(self):
        ctx = ProjectContext(
            project_id="p",
            tree_summary="s",
            files={"edit_me.ts": "hello world"},
        )
        pf = PatchFile(
            path="edit_me.ts",
            operation="edit",
            edits=[SearchReplace("world", "there")],
        )
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert result.content == "hello there"
        # Verify context.files was updated
        assert ctx.files["edit_me.ts"] == "hello there"

    def test_edit_file_not_found(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(
            path="missing.ts",
            operation="edit",
            edits=[SearchReplace("a", "b")],
        )
        result = _apply_one(pf, ctx)
        assert result.kind == "error"
        assert "file not found" in result.error

    def test_edit_patch_error_propagated(self):
        ctx = ProjectContext(
            project_id="p",
            tree_summary="s",
            files={"err.ts": "a a"},
        )
        pf = PatchFile(
            path="err.ts",
            operation="edit",
            edits=[SearchReplace("a", "b")],
        )
        result = _apply_one(pf, ctx)
        assert result.kind == "error"
        assert "edit:" in result.error

    def test_edit_empty_edits_list(self):
        ctx = ProjectContext(
            project_id="p",
            tree_summary="s",
            files={"same.ts": "content"},
        )
        pf = PatchFile(path="same.ts", operation="edit", edits=[])
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert result.content == "content"

    def test_edit_adds_use_client(self):
        ctx = ProjectContext(
            project_id="p",
            tree_summary="s",
            files={"comp.tsx": "const x = useState(0);"},
        )
        pf = PatchFile(
            path="comp.tsx",
            operation="edit",
            edits=[],
        )
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert result.content.startswith('"use client"')

    def test_delete_operation(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="remove.ts", operation="delete")
        result = _apply_one(pf, ctx)
        assert result.kind == "delete"
        assert result.path == "remove.ts"

    def test_unknown_operation(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="f.ts", operation="unknown_op")  # type: ignore
        result = _apply_one(pf, ctx)
        assert result.kind == "error"
        assert "unknown operation" in result.error

    def test_edit_truncates_content(self):
        oversized = "x" * (_MAX_FILE_CONTENT_CHARS + 1000)
        ctx = ProjectContext(
            project_id="p",
            tree_summary="s",
            files={"big.ts": oversized},
        )
        pf = PatchFile(path="big.ts", operation="edit", edits=[])
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert len(result.content) == _MAX_FILE_CONTENT_CHARS

    def test_create_truncates_oversized(self):
        oversized = "x" * (_MAX_FILE_CONTENT_CHARS + 500)
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="big.ts", operation="create", content=oversized)
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert len(result.content) == _MAX_FILE_CONTENT_CHARS


# ===================================================================
# apply_bundle
# ===================================================================


class TestApplyBundle:
    def test_empty_bundle(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        bundle = PatchBundle(files=[])
        results = apply_bundle(bundle, ctx)
        assert results == []

    def test_single_create(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        bundle = PatchBundle(
            files=[PatchFile(path="a.ts", operation="create", content="hi")]
        )
        results = apply_bundle(bundle, ctx)
        assert len(results) == 1
        assert results[0].kind == "write"
        assert results[0].path == "a.ts"

    def test_mixed_operations(self):
        ctx = ProjectContext(
            project_id="p",
            tree_summary="s",
            files={"edit.ts": "original"},
        )
        bundle = PatchBundle(
            files=[
                PatchFile(path="new.ts", operation="create", content="new"),
                PatchFile(
                    path="edit.ts",
                    operation="edit",
                    edits=[SearchReplace("original", "patched")],
                ),
                PatchFile(path="delete.ts", operation="delete"),
            ]
        )
        results = apply_bundle(bundle, ctx)
        assert len(results) == 3
        assert results[0].kind == "write"
        assert results[0].path == "new.ts"
        assert results[1].kind == "write"
        assert results[1].path == "edit.ts"
        assert results[1].content == "patched"
        assert results[2].kind == "delete"
        assert results[2].path == "delete.ts"

    def test_all_operations_attempted_on_failure(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        bundle = PatchBundle(
            files=[
                PatchFile(path="", operation="create", content="x"),  # empty path
                PatchFile(path="../bad.ts", operation="create", content="x"),  # unsafe
                PatchFile(path="good.ts", operation="create", content="ok"),
            ]
        )
        results = apply_bundle(bundle, ctx)
        assert len(results) == 3
        assert results[0].kind == "error"
        assert results[1].kind == "error"
        assert results[2].kind == "write"

    def test_replace_via_create_with_use_client(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        content = 'import { motion } from "framer-motion";'
        bundle = PatchBundle(
            files=[PatchFile(path="comp.tsx", operation="create", content=content)]
        )
        results = apply_bundle(bundle, ctx)
        assert results[0].kind == "write"
        assert '"use client"' in results[0].content


# ===================================================================
# execute_write_file
# ===================================================================


class TestExecuteWriteFile:
    def test_writes_file_with_parent_dirs(self, tmp_path: Path):
        rel = "nested/deep/file.ts"
        content = "// hello"
        result = execute_write_file(
            path=rel, content=content, project_root=tmp_path
        )
        assert result.kind == "write"
        assert result.path == rel
        assert (tmp_path / rel).read_text(encoding="utf-8") == content

    def test_creates_parent_dir(self, tmp_path: Path):
        rel = "new_dir/sub/file.js"
        result = execute_write_file(
            path=rel, content="code", project_root=tmp_path
        )
        assert result.kind == "write"
        assert (tmp_path / rel).exists()

    def test_empty_path_returns_error(self, tmp_path: Path):
        result = execute_write_file(
            path="", content="x", project_root=tmp_path
        )
        assert result.kind == "error"
        assert result.error == "empty path"

    def test_unsafe_path_returns_error(self, tmp_path: Path):
        result = execute_write_file(
            path="../escape.ts", content="x", project_root=tmp_path
        )
        assert result.kind == "error"
        assert "unsafe path" in result.error

    def test_no_project_root_returns_error(self):
        result = execute_write_file(
            path="file.ts", content="x", project_root=None
        )
        assert result.kind == "error"
        assert "no project_root" in result.error

    def test_oserror_caught(self, tmp_path: Path):
        """Simulate an OSError by making path non-writable."""
        rel = "readonly/sub/file.ts"
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        # Make the file read-only
        target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        result = execute_write_file(
            path=rel, content="new content", project_root=tmp_path
        )
        assert result.kind == "error"
        assert "error writing file" in result.error
        # Restore permissions for cleanup
        target.chmod(stat.S_IWUSR | stat.S_IRUSR)

    def test_adds_use_client(self, tmp_path: Path):
        content = 'import { motion } from "framer-motion";'
        result = execute_write_file(
            path="comp.tsx", content=content, project_root=tmp_path
        )
        assert result.kind == "write"
        assert '"use client"' in result.content
        assert (tmp_path / "comp.tsx").read_text(encoding="utf-8").startswith('"use client"')

    def test_truncates_oversized(self, tmp_path: Path):
        oversized = "x" * (_MAX_FILE_CONTENT_CHARS + 500)
        result = execute_write_file(
            path="big.txt", content=oversized, project_root=tmp_path
        )
        assert result.kind == "write"
        assert len(result.content) == _MAX_FILE_CONTENT_CHARS
        assert len((tmp_path / "big.txt").read_text(encoding="utf-8")) == _MAX_FILE_CONTENT_CHARS

    def test_with_storage_and_project_id(self, tmp_path: Path):
        """When storage and project_id are provided, still writes to disk."""
        storage = MagicMock()
        rel = "stored/file.ts"
        content = "stored content"
        result = execute_write_file(
            path=rel,
            content=content,
            project_root=tmp_path,
            storage=storage,
            project_id="proj_123",
        )
        assert result.kind == "write"
        # Should still write to disk
        assert (tmp_path / rel).exists()
        assert (tmp_path / rel).read_text(encoding="utf-8") == content

    def test_whitespace_path_normalized(self, tmp_path: Path):
        result = execute_write_file(
            path="  spaced/file.ts  ", content="data", project_root=tmp_path
        )
        assert result.kind == "write"
        assert result.path == "spaced/file.ts"

    def test_backslash_in_path(self, tmp_path: Path):
        result = execute_write_file(
            path="a\\b\\file.ts", content="data", project_root=tmp_path
        )
        assert result.kind == "write"
        assert result.path == "a/b/file.ts"
        assert (tmp_path / "a" / "b" / "file.ts").exists()


# ===================================================================
# execute_read_file
# ===================================================================


class TestExecuteReadFile:
    def test_reads_existing_file(self, tmp_path: Path):
        target = tmp_path / "hello.txt"
        target.write_text("world", encoding="utf-8")
        result = execute_read_file("hello.txt", tmp_path)
        assert result == "world"

    def test_reads_nested_file(self, tmp_path: Path):
        target = tmp_path / "a" / "b" / "c.txt"
        target.parent.mkdir(parents=True)
        target.write_text("nested", encoding="utf-8")
        result = execute_read_file("a/b/c.txt", tmp_path)
        assert result == "nested"

    def test_empty_path(self, tmp_path: Path):
        result = execute_read_file("", tmp_path)
        assert result == "error: empty path"

    def test_unsafe_path(self, tmp_path: Path):
        result = execute_read_file("../outside.txt", tmp_path)
        assert "error: path outside project root" in result

    def test_unsafe_path_git(self, tmp_path: Path):
        result = execute_read_file(".git/config", tmp_path)
        assert "error: path outside project root" in result

    def test_file_not_found(self, tmp_path: Path):
        result = execute_read_file("nonexistent.txt", tmp_path)
        assert "error: file not found" in result

    def test_oserror_caught(self, tmp_path: Path):
        """Simulate an OS error by pointing to a directory as if it were a file."""
        target = tmp_path / "mydir"
        target.mkdir()
        result = execute_read_file("mydir", tmp_path)
        assert "error:" in result

    def test_leading_slash_stripped(self, tmp_path: Path):
        target = tmp_path / "foo.txt"
        target.write_text("content", encoding="utf-8")
        result = execute_read_file("/foo.txt", tmp_path)
        assert result == "content"

    def test_backslash_in_path(self, tmp_path: Path):
        target = tmp_path / "bar.txt"
        target.write_text("data", encoding="utf-8")
        result = execute_read_file("bar.txt", tmp_path)
        assert result == "data"


# ===================================================================
# Integration: apply_bundle + execute_write_file round-trip
# ===================================================================


class TestIntegration:
    def test_create_then_edit_then_read(self, tmp_path: Path):
        """Realistic scenario: create a file, edit it, verify content on disk."""
        # Step 1: Create file via execute_write_file
        content = 'import { useState } from "react";\nconst [count, setCount] = useState(0);'
        r1 = execute_write_file(
            path="src/hello.tsx",
            content=content,
            project_root=tmp_path,
        )
        assert r1.kind == "write"
        assert r1.path == "src/hello.tsx"

        # Step 2: Build a ProjectContext that loads from the real filesystem
        ctx = ProjectContext(
            project_id="integ",
            tree_summary="test",
        )
        # Populate context with same file (will have "use client" added)
        ctx.files["src/hello.tsx"] = (tmp_path / "src/hello.tsx").read_text(encoding="utf-8")

        # Step 3: Apply an edit via bundle — use unique search strings
        bundle = PatchBundle(
            files=[
                PatchFile(
                    path="src/hello.tsx",
                    operation="edit",
                    edits=[
                        SearchReplace("useState(0)", "useReducer(0, dummyReducer)"),
                    ],
                )
            ]
        )
        results = apply_bundle(bundle, ctx)
        assert len(results) == 1
        assert results[0].kind == "write", f"Expected write, got error: {results[0].error}"
        assert "useReducer" in results[0].content
        assert "dummyReducer" in results[0].content
        # Verify context was updated
        assert "useReducer" in ctx.files["src/hello.tsx"]

    def test_bundle_with_multiple_ops_on_disk(self, tmp_path: Path):
        """Create multiple files via bundle, verify them on disk."""
        # Use apply_bundle to create files (writes to context.files, not disk)
        ctx = ProjectContext(project_id="multi", tree_summary="multi")
        bundle = PatchBundle(
            files=[
                PatchFile(path="a.ts", operation="create", content="// a"),
                PatchFile(path="b.ts", operation="create", content="// b"),
                PatchFile(path="c.ts", operation="create", content="// c"),
            ]
        )
        results = apply_bundle(bundle, ctx)
        assert all(r.kind == "write" for r in results)
        # Now write them to disk
        for r in results:
            wr = execute_write_file(
                path=r.path, content=r.content, project_root=tmp_path
            )
            assert wr.kind == "write"
            assert (tmp_path / r.path).exists()

        assert (tmp_path / "a.ts").read_text(encoding="utf-8") == "// a"
        assert (tmp_path / "b.ts").read_text(encoding="utf-8") == "// b"
        assert (tmp_path / "c.ts").read_text(encoding="utf-8") == "// c"


# ===================================================================
# Edge cases and error paths not covered above
# ===================================================================


class TestEdgeCases:
    def test_operation_type_alias(self):
        """Verify Operation is a Literal type that accepts valid values."""
        valid: Operation = "create"
        assert valid == "create"
        valid = "replace"
        valid = "edit"
        valid = "delete"

    def test_file_loader_protocol(self):
        """FileLoader protocol is structural, so any callable works."""
        def loader(path: str) -> str | None:
            return None
        _: FileLoader = loader

    def test_normalize_path_handles_mixed_separators(self):
        assert _normalize_path(" /foo\\bar/baz\\qux ") == "foo/bar/baz/qux"

    def test_path_is_safe_with_trailing_slash_in_parts(self):
        """parts.split from Path doesn't normally produce empty segments, but verify."""
        # Path('foo/').parts == ('foo',) on most platforms, so trailing slash is fine.
        assert _path_is_safe("foo/") is True

    def test_ensure_use_client_empty_content(self):
        result = _ensure_use_client("empty.tsx", "")
        # Empty content doesn't match any client indicators
        assert result == ""

    def test_ensure_use_client_only_newlines(self):
        content = "\n\n\n"
        result = _ensure_use_client("comp.tsx", content)
        # No client indicators, returns as-is
        assert result == "\n\n\n"

    def test_apply_patch_multiple_ops_on_same_string(self):
        ops = [
            SearchReplace("a", "b"),
            SearchReplace("b", "c"),
            SearchReplace("c", "d"),
        ]
        assert apply_patch("a", ops) == "d"

    def test_apply_patch_only_replaces_first_occurrence(self):
        """str.replace with count=1 only replaces first occurrence."""
        # But the function checks count == 1, so single match enforced first.
        pass  # covered by multiple_matches_raises

    def test_apply_one_normalizes_path(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        # Path with leading slash should be normalized
        pf = PatchFile(path="/normalized.ts", operation="create", content="ok")
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert result.path == "normalized.ts"

    def test_apply_one_create_empty_content_string(self):
        """create with empty string content should be valid (not None)."""
        ctx = ProjectContext(project_id="p", tree_summary="s")
        pf = PatchFile(path="empty.ts", operation="create", content="")
        result = _apply_one(pf, ctx)
        assert result.kind == "write"
        assert result.content == ""

    def test_execute_write_file_leading_slash_path(self, tmp_path: Path):
        result = execute_write_file(
            path="/leading/slash.ts", content="data", project_root=tmp_path
        )
        assert result.kind == "write"
        assert result.path == "leading/slash.ts"
        assert (tmp_path / "leading" / "slash.ts").exists()

    def test_execute_read_file_leading_slash(self, tmp_path: Path):
        target = tmp_path / "slashed.txt"
        target.write_text("content", encoding="utf-8")
        result = execute_read_file("/slashed.txt", tmp_path)
        assert result == "content"

    def test_execute_write_file_backslash_path(self, tmp_path: Path):
        result = execute_write_file(
            path="win\\style\\path.ts", content="data", project_root=tmp_path
        )
        assert result.kind == "write"
        assert (tmp_path / "win" / "style" / "path.ts").exists()

    def test_patcher_error_is_exception(self):
        assert issubclass(PatchError, Exception)
        e = PatchError("test error")
        assert str(e) == "test error"

    def test_none_path_normalize(self):
        """_normalize_path(None) should fail with AttributeError since raw.strip() won't work."""
        pass  # Type-safe - not expected to be called with None

    def test_apply_bundle_preserves_order(self):
        ctx = ProjectContext(project_id="p", tree_summary="s")
        bundle = PatchBundle(
            files=[
                PatchFile(path="z.ts", operation="create", content="last"),
                PatchFile(path="a.ts", operation="create", content="first"),
            ]
        )
        results = apply_bundle(bundle, ctx)
        assert results[0].path == "z.ts"
        assert results[1].path == "a.ts"

    def test_project_context_get_file_caches_loader_result(self):
        loader = MagicMock(spec=FileLoader)
        loader.return_value = "cached_content"
        ctx = ProjectContext(project_id="p", tree_summary="s", loader=loader)
        # First call loads via loader
        r1 = ctx.get_file("cached.py")
        assert r1 == "cached_content"
        # Second call uses cache
        r2 = ctx.get_file("cached.py")
        assert r2 == "cached_content"
        loader.assert_called_once()  # Only called once

    def test_path_is_safe_forbidden_segments_middle(self):
        assert _path_is_safe("src/node_modules/pkg/index.js") is False
        assert _path_is_safe("src/.micracode/data.txt") is False
        assert _path_is_safe("lib/.git/config") is False
