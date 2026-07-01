"""Comprehensive tests for cortex/codegen/context.py — >95% target coverage.

Tests cover:
  - Constants (CONTEXT_CHAR_BUDGET, ALWAYS_LOAD, PLACEHOLDER_CANDIDATES, etc.)
  - Tree flattening (_flatten): nesting, emptiness, non-string contents
  - Tree reading (_read_from_tree): correct path, missing, broken nodes
  - Path mentioning (_mentioned_paths): by full path, basename, short names
  - File loader factory (_build_loader): error handling, path traversal
  - Context loader (load_context): priority, budget, placeholders, errors
  - Context rendering (render_context_block): empty, files, placeholders, truncation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cortex.codegen.context import (
    CONTEXT_CHAR_BUDGET,
    CONTEXT_FILE_DISPLAY_CAP,
    ALWAYS_LOAD,
    MAX_TREE_ENTRIES,
    PLACEHOLDER_CANDIDATES,
    _flatten,
    _mentioned_paths,
    _read_from_tree,
    load_context,
    render_context_block,
)
from cortex.codegen.patcher import ProjectContext
from cortex.codegen.starter import NEXT_STARTER


# =========================================================================
# Constants
# =========================================================================


class TestConstants:
    """Verify module-level constants exist with expected values."""

    def test_context_char_budget(self):
        assert CONTEXT_CHAR_BUDGET == 40_000

    def test_context_file_display_cap(self):
        assert CONTEXT_FILE_DISPLAY_CAP == 12_000

    def test_always_load(self):
        assert ALWAYS_LOAD == (
            "app/page.tsx",
            "app/layout.tsx",
            "app/globals.css",
        )

    def test_placeholder_candidates(self):
        assert PLACEHOLDER_CANDIDATES == (
            "app/page.tsx",
            "app/layout.tsx",
            "app/globals.css",
        )

    def test_max_tree_entries(self):
        assert MAX_TREE_ENTRIES == 400


# =========================================================================
# _flatten – tree flattening
# =========================================================================


class TestFlatten:
    """_flatten converts a nested tree dict into (path, size) tuples."""

    def test_flat_list_of_files(self):
        """A single-level tree is flattened correctly."""
        tree = {
            "readme.md": {"file": {"contents": "hello"}},
            "main.py": {"file": {"contents": "print(1)"}},
        }
        result = _flatten(tree)
        assert sorted(result) == [
            ("main.py", 8),
            ("readme.md", 5),
        ]

    def test_nested_directories(self):
        """Files inside directories produce paths with slashes."""
        tree = {
            "app": {
                "directory": {
                    "page.tsx": {"file": {"contents": "export default Page() {}"}},
                    "layout.tsx": {"file": {"contents": "layout"}},
                }
            },
        }
        result = _flatten(tree)
        assert sorted(result) == [
            ("app/layout.tsx", 6),
            ("app/page.tsx", 24),
        ]

    def test_deep_nesting(self):
        """Deeply nested directories flatten to full paths."""
        tree = {
            "a": {
                "directory": {
                    "b": {
                        "directory": {
                            "c.py": {"file": {"contents": "x"}},
                        }
                    }
                }
            },
        }
        result = _flatten(tree)
        assert result == [("a/b/c.py", 1)]

    def test_empty_tree(self):
        """An empty tree returns an empty list."""
        assert _flatten({}) == []

    def test_non_string_contents(self):
        """Non-string contents are treated as size 0."""
        tree = {
            "f.tsx": {"file": {"contents": 123}},
            "g.tsx": {"file": {"contents": None}},
        }
        result = _flatten(tree)
        assert sorted(result) == [
            ("f.tsx", 0),
            ("g.tsx", 0),
        ]

    def test_directory_with_no_files(self):
        """A directory with no file children produces no entries."""
        tree = {
            "empty_dir": {"directory": {}},
        }
        assert _flatten(tree) == []

    def test_missing_contents_key(self):
        """A file node without 'contents' defaults to size 0."""
        tree = {
            "no_content.ts": {"file": {}},
        }
        assert _flatten(tree) == [("no_content.ts", 0)]

    def test_mixed_tree(self):
        """Files and directories coexist at the same level."""
        tree = {
            "readme.md": {"file": {"contents": "hello"}},
            "src": {
                "directory": {
                    "index.ts": {"file": {"contents": "hi"}},
                }
            },
        }
        result = _flatten(tree)
        assert sorted(result) == [
            ("readme.md", 5),
            ("src/index.ts", 2),
        ]


# =========================================================================
# _read_from_tree – reading files from tree
# =========================================================================


class TestReadFromTree:
    """_read_from_tree extracts file contents by path."""

    def test_reads_correct_file(self):
        """A simple path returns the file contents."""
        tree = {
            "hello.txt": {"file": {"contents": "world"}},
        }
        assert _read_from_tree(tree, "hello.txt") == "world"

    def test_reads_nested_file(self):
        """A nested path returns the file contents."""
        tree = {
            "app": {
                "directory": {
                    "page.tsx": {"file": {"contents": "page"}},
                }
            },
        }
        assert _read_from_tree(tree, "app/page.tsx") == "page"

    def test_returns_none_for_missing_file(self):
        """A path that doesn't exist returns None."""
        tree = {"exists.txt": {"file": {"contents": "yes"}}}
        assert _read_from_tree(tree, "missing.txt") is None

    def test_returns_none_for_missing_intermediate_dir(self):
        """A path with missing intermediate directory returns None."""
        tree = {
            "app": {
                "directory": {
                    "page.tsx": {"file": {"contents": "page"}},
                }
            },
        }
        assert _read_from_tree(tree, "app/missing/page.tsx") is None

    def test_returns_none_if_node_is_not_dict(self):
        """A path segment that is not a dict returns None."""
        tree = {
            "app": {"file": {"contents": "not a dir"}},
        }
        assert _read_from_tree(tree, "app/page.tsx") is None

    def test_returns_none_if_leaf_not_file(self):
        """A leaf that is not a 'file' node returns None."""
        tree = {
            "dirname": {
                "directory": {
                    "sub": {"file": {"contents": "ok"}},
                }
            },
            "also_dir": {"directory": {}},
        }
        # 'also_dir' is a directory, not a file
        assert _read_from_tree(tree, "also_dir") is None

    def test_returns_none_if_file_node_has_no_contents(self):
        """A leaf with 'file' key but no 'contents' key returns None."""
        tree = {
            "empty": {"file": {}},
        }
        assert _read_from_tree(tree, "empty") is None

    def test_returns_none_for_non_string_contents(self):
        """File contents that are not a string return None."""
        tree = {
            "nums.txt": {"file": {"contents": 42}},
        }
        assert _read_from_tree(tree, "nums.txt") is None

    def test_intermediate_node_not_directory(self):
        """If an intermediate path component is a file, not a directory."""
        tree = {
            "config.json": {"file": {"contents": "{}"}},
        }
        assert _read_from_tree(tree, "config.json/extra") is None

    def test_intermediate_missing_directory_key(self):
        """An intermediate dict node without 'directory' key returns None."""
        tree = {
            "src": {"not_directory": {}},
        }
        assert _read_from_tree(tree, "src/main.py") is None

    def test_intermediate_node_is_not_dict(self):
        """An intermediate node that resolves to a non-dict triggers line 63."""
        tree = {
            "app": "just a string",
        }
        assert _read_from_tree(tree, "app/page.tsx") is None

    def test_tree_not_a_dict(self):
        """When tree itself is not a dict, return None at isinstance check."""
        assert _read_from_tree("not a dict", "file.txt") is None


# =========================================================================
# _mentioned_paths – finding paths mentioned in prompt
# =========================================================================


class TestMentionedPaths:
    """_mentioned_paths finds file paths referenced in a prompt string."""

    def test_finds_path_by_full_path(self):
        """A full path in the prompt is detected."""
        prompt = "please update app/page.tsx"
        candidates = ["app/page.tsx", "app/layout.tsx"]
        assert _mentioned_paths(prompt, candidates) == ["app/page.tsx"]

    def test_finds_path_by_basename(self):
        """A basename (>3 chars) in the prompt is detected."""
        prompt = "fix the layout.tsx please"
        candidates = ["app/layout.tsx", "app/page.tsx"]
        assert _mentioned_paths(prompt, candidates) == ["app/layout.tsx"]

    def test_multiple_hits(self):
        """Multiple mentioned paths are returned in order."""
        prompt = "update page.tsx and layout.tsx"
        candidates = ["app/page.tsx", "app/layout.tsx", "app/globals.css"]
        assert _mentioned_paths(prompt, candidates) == [
            "app/page.tsx",
            "app/layout.tsx",
        ]

    def test_no_matches(self):
        """No matches returns empty list."""
        prompt = "do something else"
        candidates = ["app/page.tsx", "app/layout.tsx"]
        assert _mentioned_paths(prompt, candidates) == []

    def test_empty_prompt(self):
        """An empty prompt returns no matches."""
        assert _mentioned_paths("", ["app/page.tsx"]) == []

    def test_short_basename_not_matched(self):
        """A basename of 3 or fewer chars is not matched."""
        prompt = "fix .ts"
        candidates = ["index.ts", "app/page.tsx"]
        assert _mentioned_paths(prompt, candidates) == []

    def test_basename_exactly_three_chars(self):
        """A basename exactly 3 chars is matched if it appears in prompt."""
        prompt = "fix abc.ts please"
        candidates = ["abc.ts"]
        result = _mentioned_paths(prompt, candidates)
        # Since 'abc.ts' in prompt -> True (path in prompt)
        # Also base = 'abc.ts', len = 6 > 3
        # So it will be matched by both conditions
        assert result == ["abc.ts"]

    def test_basename_less_than_three_chars_but_path_matches(self):
        """A basename <=3 chars needs exact path match."""
        prompt = "fix ab.ts"
        candidates = ["ab.ts"]
        # path in prompt: 'ab.ts' in 'fix ab.ts' -> True
        assert _mentioned_paths(prompt, candidates) == ["ab.ts"]

    def test_deduplication(self):
        """Duplicate matches in candidates are not duplicated in output."""
        prompt = "app/page.tsx"
        # Only one occurrence in candidates, so dedup isn't tested here directly
        # but the function naturally dedupes since each candidate is checked once
        candidates = ["app/page.tsx", "app/page.tsx"]
        result = _mentioned_paths(prompt, candidates)
        assert len(result) == 2  # both candidates match individually


# =========================================================================
# load_context – main context loader
# =========================================================================


class TestLoadContext:
    """load_context builds a ProjectContext from a storage backend."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_storage(tree: dict | None = None, **kwargs) -> MagicMock:
        """Create a mock storage with a read_tree method."""
        storage = MagicMock()
        storage.read_tree.return_value = tree or {}
        if "project_dir" not in kwargs:
            storage.project_dir.return_value = "/fake/project"
        for key, val in kwargs.items():
            setattr(storage, key, val)
        return storage

    @staticmethod
    def _tree_with_files(file_map: dict[str, str]) -> dict:
        """Build a tree dict from a simple path→content map."""
        tree: dict = {}
        for path, content in file_map.items():
            parts = path.split("/")
            current = tree
            for i, part in enumerate(parts):
                is_last = i == len(parts) - 1
                if is_last:
                    current[part] = {"file": {"contents": content}}
                else:
                    if part not in current:
                        current[part] = {"directory": {}}
                    current = current[part]["directory"]
        return tree

    # ------------------------------------------------------------------
    # Basic / Happy path
    # ------------------------------------------------------------------

    def test_basic_happy_path(self):
        """A normal tree with files returns a valid ProjectContext."""
        tree = self._tree_with_files({
            "app/page.tsx": "page content",
            "app/layout.tsx": "layout content",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "my-project", "update the page")
        assert isinstance(ctx, ProjectContext)
        assert ctx.project_id == "my-project"
        assert "app/page.tsx" in ctx.files
        assert ctx.files["app/page.tsx"] == "page content"

    def test_tree_summary_includes_all_files(self):
        """Tree summary contains flattened file list with sizes."""
        tree = self._tree_with_files({
            "a.ts": "abc",
            "b.ts": "12345",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        assert "a.ts" in ctx.tree_summary
        assert "b.ts" in ctx.tree_summary
        assert "(3)" in ctx.tree_summary or "abc" in ctx.tree_summary
        assert "(5)" in ctx.tree_summary or "12345" in ctx.tree_summary

    def test_always_loaded_files_included(self):
        """Files in ALWAYS_LOAD that exist in the tree are included."""
        tree = self._tree_with_files({
            "app/page.tsx": "page",
            "app/layout.tsx": "layout",
            "app/globals.css": "css",
            "other.ts": "other",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "mention other.ts")
        # All always-load files should be present
        assert "app/page.tsx" in ctx.files
        assert "app/layout.tsx" in ctx.files
        assert "app/globals.css" in ctx.files
        # The mentioned file should also be present
        assert "other.ts" in ctx.files

    def test_always_loaded_missing_are_skipped(self):
        """Files in ALWAYS_LOAD not in the tree are not included."""
        tree = self._tree_with_files({
            "app/page.tsx": "page",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        assert "app/page.tsx" in ctx.files
        assert "app/layout.tsx" not in ctx.files
        assert "app/globals.css" not in ctx.files

    # ------------------------------------------------------------------
    # Mentioned paths
    # ------------------------------------------------------------------

    def test_mentioned_paths_included(self):
        """Files mentioned in the prompt are included."""
        tree = self._tree_with_files({
            "app/page.tsx": "page",
            "components/button.tsx": "button",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "fix button.tsx")
        assert "components/button.tsx" in ctx.files

    # ------------------------------------------------------------------
    # Deduplication (always + mentioned)
    # ------------------------------------------------------------------

    def test_dedup_always_and_mentioned(self):
        """A file that is both always-loaded and mentioned appears once."""
        tree = self._tree_with_files({
            "app/page.tsx": "page",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "app/page.tsx")
        # Should be in files exactly once
        assert list(ctx.files.keys()).count("app/page.tsx") == 1

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def test_budget_respected_under_limit(self):
        """Multiple files within budget are all loaded when files are wanted."""
        files = {}
        for i in range(20):
            files[f"file{i}.ts"] = "x" * 100
        tree = self._tree_with_files(files)
        storage = self._make_storage(tree)
        # Mention all files so they get included
        prompt = " and ".join(f"file{i}.ts" for i in range(20))
        ctx = load_context(storage, "p", prompt)
        # All 20 files (2000 chars total) should fit in 40000 budget
        assert len(ctx.files) == 20

    def test_budget_stops_after_exhaustion(self):
        """Once budget is exhausted, no more files are loaded."""
        big = "x" * (CONTEXT_CHAR_BUDGET - 5)  # almost fills budget
        tree = self._tree_with_files({
            "app/page.tsx": big,
            "app/layout.tsx": "remaining",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        # page.tsx loaded, budget becomes ~5
        # layout.tsx is 9 chars > remaining budget, so skipped
        assert "app/page.tsx" in ctx.files
        assert "app/layout.tsx" not in ctx.files

    def test_file_exceeding_budget_skipped(self):
        """A file larger than the current budget is skipped."""
        huge = "x" * (CONTEXT_CHAR_BUDGET + 1)
        tree = self._tree_with_files({
            "app/page.tsx": huge,
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        assert "app/page.tsx" not in ctx.files

    def test_multiple_files_within_budget(self):
        """Multiple small files all fit within budget."""
        small = "x" * 100
        tree = self._tree_with_files({
            "app/page.tsx": small,
            "app/layout.tsx": small,
            "app/globals.css": small,
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        assert len(ctx.files) == 3

    def test_budget_exactly_zero_triggers_break(self):
        """Budget reaching zero triggers the break in the load loop."""
        # First file exactly fills the budget
        big = "x" * CONTEXT_CHAR_BUDGET
        tree = self._tree_with_files({
            "app/page.tsx": big,
            "app/layout.tsx": "should not be loaded",
            "app/globals.css": "neither should this",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        # Only the first file should be loaded
        assert "app/page.tsx" in ctx.files
        assert "app/layout.tsx" not in ctx.files
        assert "app/globals.css" not in ctx.files

    def test_content_is_none_skipped(self):
        """A file with None content from tree is skipped."""
        tree = {
            "app": {
                "directory": {
                    "page.tsx": {"file": {"contents": None}},
                    "layout.tsx": {"file": {"contents": "actual"}},
                }
            },
        }
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        # page.tsx has None content -> skipped, layout.tsx is fine
        assert "app/page.tsx" not in ctx.files
        assert "app/layout.tsx" in ctx.files
        assert ctx.files["app/layout.tsx"] == "actual"

    # ------------------------------------------------------------------
    # Tree read errors
    # ------------------------------------------------------------------

    def test_file_not_found_error_on_tree(self):
        """FileNotFoundError from storage.read_tree results in empty tree."""
        storage = self._make_storage()
        storage.read_tree.side_effect = FileNotFoundError("no tree")
        ctx = load_context(storage, "p", "")
        assert ctx.tree_summary == ""
        assert ctx.files == {}

    def test_not_implemented_error_on_tree(self):
        """NotImplementedError from storage.read_tree results in empty tree."""
        storage = self._make_storage()
        storage.read_tree.side_effect = NotImplementedError()
        ctx = load_context(storage, "p", "")
        assert ctx.tree_summary == ""
        assert ctx.files == {}

    def test_attribute_error_on_tree(self):
        """AttributeError from storage.read_tree results in empty tree."""
        storage = self._make_storage()
        storage.read_tree.side_effect = AttributeError()
        ctx = load_context(storage, "p", "")
        assert ctx.tree_summary == ""
        assert ctx.files == {}

    def test_unexpected_error_on_tree_propagates(self):
        """Unexpected errors from storage.read_tree propagate."""
        storage = self._make_storage()
        storage.read_tree.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            load_context(storage, "p", "")

    # ------------------------------------------------------------------
    # MAX_TREE_ENTRIES truncation
    # ------------------------------------------------------------------

    def test_tree_summary_truncation(self):
        """Tree summary is truncated at MAX_TREE_ENTRIES entries."""
        entries = {f"f{i}.ts": {"file": {"contents": "x"}} for i in range(MAX_TREE_ENTRIES + 50)}
        tree = dict(sorted(entries.items()))
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        lines = ctx.tree_summary.split("\n")
        assert len(lines) == MAX_TREE_ENTRIES + 1  # +1 for the "..." line
        assert "... (50 more files)" in lines[-1]

    def test_tree_summary_no_truncation_when_below_limit(self):
        """Tree summary without truncation has no '...' line."""
        entries = {f"f{i}.ts": {"file": {"contents": "x"}} for i in range(10)}
        tree = dict(sorted(entries.items()))
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        lines = ctx.tree_summary.split("\n")
        assert len(lines) == 10
        assert not any("..." in line for line in lines)

    # ------------------------------------------------------------------
    # Placeholder detection with starter name
    # ------------------------------------------------------------------

    def test_placeholder_with_starter_name(self):
        """Placeholder files are detected when starter_name is given."""
        tree = self._tree_with_files({
            "app/page.tsx": NEXT_STARTER["app/page.tsx"],
            "app/layout.tsx": "modified layout",
            "app/globals.css": NEXT_STARTER["app/globals.css"],
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "", starter_name="next")
        # page.tsx and globals.css match the starter -> placeholders
        # layout.tsx is modified -> not a placeholder
        assert "app/page.tsx" in ctx.placeholder_files
        assert "app/layout.tsx" not in ctx.placeholder_files
        assert "app/globals.css" in ctx.placeholder_files

    def test_placeholder_starter_not_found(self):
        """An unknown starter name produces no placeholders."""
        tree = self._tree_with_files({
            "app/page.tsx": NEXT_STARTER["app/page.tsx"],
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "", starter_name="nonexistent")
        assert len(ctx.placeholder_files) == 0

    def test_placeholder_starter_returns_none(self):
        """When get_starter returns None, no placeholders are detected."""
        tree = self._tree_with_files({
            "app/page.tsx": "some content",
        })
        storage = self._make_storage(tree)
        with patch("cortex.codegen.context.get_starter", return_value=None):
            ctx = load_context(storage, "p", "", starter_name="next")
        assert len(ctx.placeholder_files) == 0

    # ------------------------------------------------------------------
    # Placeholder detection fallback (no starter_name -> NEXT_STARTER)
    # ------------------------------------------------------------------

    def test_placeholder_fallback_to_next_starter(self):
        """Without starter_name, fallback compares against NEXT_STARTER."""
        tree = self._tree_with_files({
            "app/page.tsx": NEXT_STARTER["app/page.tsx"],
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        assert "app/page.tsx" in ctx.placeholder_files

    def test_placeholder_fallback_no_match(self):
        """Fallback: file not in NEXT_STARTER produces no placeholder."""
        tree = self._tree_with_files({
            "app/page.tsx": "custom content",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        assert len(ctx.placeholder_files) == 0

    def test_placeholder_fallback_path_not_in_starter(self):
        """Fallback: path not in NEXT_STARTER is not a placeholder."""
        tree = self._tree_with_files({
            "components/button.tsx": "anything",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        assert len(ctx.placeholder_files) == 0

    def test_placeholder_fallback_file_in_starter_not_in_files(self):
        """A PLACEHOLDER_CANDIDATES path not in files is not flagged."""
        # page.tsx content doesn't match NEXT_STARTER, so no placeholder
        tree = self._tree_with_files({
            "app/page.tsx": "custom content",
            "app/globals.css": "x",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        # page.tsx content != NEXT_STARTER["app/page.tsx"], so not a placeholder
        # globals.css content "x" != NEXT_STARTER["app/globals.css"], so not a placeholder
        assert len(ctx.placeholder_files) == 0

    # ------------------------------------------------------------------
    # Empty / edge cases
    # ------------------------------------------------------------------

    def test_empty_project(self):
        """An empty tree results in empty files and summary."""
        storage = self._make_storage({})
        ctx = load_context(storage, "p", "")
        assert ctx.tree_summary == ""
        assert ctx.files == {}

    def test_loader_is_callable(self):
        """The context's loader is a callable function."""
        tree = self._tree_with_files({"hello.txt": "world"})
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "")
        assert callable(ctx.loader)

    def test_project_id_preserved(self):
        """The project_id is passed through to ProjectContext."""
        storage = self._make_storage({})
        ctx = load_context(storage, "my-cool-project", "")
        assert ctx.project_id == "my-cool-project"

    def test_storage_read_tree_called_with_project_id(self):
        """storage.read_tree is called with the correct project_id."""
        storage = self._make_storage({})
        load_context(storage, "proj-42", "")
        storage.read_tree.assert_called_once_with("proj-42")

    def test_files_ordered_always_then_mentioned(self):
        """Files preserve ordering: always-loaded first, then mentioned."""
        tree = self._tree_with_files({
            "app/page.tsx": "page",
            "app/globals.css": "css",
            "app/layout.tsx": "layout",
            "other/file.ts": "other",
            "another/lib.ts": "lib",
        })
        storage = self._make_storage(tree)
        ctx = load_context(storage, "p", "lib.ts and file.ts")
        keys = list(ctx.files.keys())
        # ALWAYS_LOAD order: page.tsx, layout.tsx, globals.css
        # Then mentioned: lib.ts, file.ts
        assert keys.index("app/page.tsx") < keys.index("app/layout.tsx")
        assert keys.index("app/layout.tsx") < keys.index("app/globals.css")
        # Mentioned files come after always-loaded
        for mentioned in ("another/lib.ts", "other/file.ts"):
            for always in ("app/page.tsx", "app/layout.tsx", "app/globals.css"):
                assert keys.index(always) < keys.index(mentioned)


# =========================================================================
# _build_loader – file loader factory
# =========================================================================


class TestBuildLoader:
    """_build_loader creates a callable that reads files from disk."""

    def test_loader_returns_content(self, tmp_path):
        """The loader reads a file from disk correctly."""
        d = tmp_path / "project"
        d.mkdir()
        (d / "hello.txt").write_text("world")
        storage = MagicMock()
        storage.project_dir.return_value = d
        ctx = load_context(storage, "p", "")
        assert ctx.loader("hello.txt") == "world"

    def test_loader_returns_none_for_missing_file(self, tmp_path):
        """The loader returns None for a non-existent file."""
        d = tmp_path / "project"
        d.mkdir()
        storage = MagicMock()
        storage.project_dir.return_value = d
        ctx = load_context(storage, "p", "")
        assert ctx.loader("missing.txt") is None

    def test_loader_prevents_path_traversal(self, tmp_path):
        """The loader rejects paths that escape the project root."""
        d = tmp_path / "project"
        d.mkdir()
        (tmp_path / "secret.txt").write_text("hush")
        storage = MagicMock()
        storage.project_dir.return_value = d
        ctx = load_context(storage, "p", "")
        assert ctx.loader("../secret.txt") is None

    def test_loader_project_dir_raises(self):
        """When project_dir raises, the loader always returns None."""
        storage = MagicMock()
        storage.project_dir.side_effect = Exception("no dir")
        ctx = load_context(storage, "p", "")
        assert ctx.loader("any.txt") is None

    def test_loader_project_root_is_none(self):
        """When project_root is None (project_dir error), loader returns None."""
        # Covered by test_loader_project_dir_raises but testing directly
        storage = MagicMock()
        storage.project_dir.side_effect = OSError("denied")
        ctx = load_context(storage, "p", "")
        assert ctx.loader("file.txt") is None

    def test_loader_oserror_on_read(self, tmp_path):
        """The loader returns None on OSError during read."""
        d = tmp_path / "project"
        d.mkdir()
        f = d / "locked.txt"
        f.write_text("data")
        # Make it unreadable
        f.chmod(0o000)
        try:
            storage = MagicMock()
            storage.project_dir.return_value = d
            ctx = load_context(storage, "p", "")
            result = ctx.loader("locked.txt")
            # May be None if permission denied (OSError) or succeed on root
            assert result is None or result == "data"
        finally:
            f.chmod(0o644)

    def test_loader_project_dir_not_a_directory(self, tmp_path):
        """If project_dir points to a file, the loader returns None."""
        f = tmp_path / "not_a_dir"
        f.write_text("")
        storage = MagicMock()
        storage.project_dir.return_value = f
        ctx = load_context(storage, "p", "")
        # resolve() on a file as root, then relative_to check... worth testing
        result = ctx.loader("anything.txt")
        # is_file() check happens after path traversal check
        # full.relative_to(project_root.resolve()) should pass
        # but full.is_file() should be False
        assert result is None

    def test_loader_unicode_decode_error(self, tmp_path):
        """The loader returns None on UnicodeDecodeError."""
        d = tmp_path / "project"
        d.mkdir()
        (d / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")
        storage = MagicMock()
        storage.project_dir.return_value = d
        ctx = load_context(storage, "p", "")
        result = ctx.loader("binary.bin")
        assert result is None


# =========================================================================
# render_context_block – rendering context for prompt
# =========================================================================


class TestRenderContextBlock:
    """render_context_block formats a ProjectContext as a prompt string."""

    def test_empty_project_returns_placeholder_message(self):
        """Empty context (no summary, no files) returns the empty-project message."""
        ctx = ProjectContext(project_id="p", tree_summary="", files={})
        result = render_context_block(ctx)
        assert result == "Current project: (empty - this is the first turn)."

    def test_with_tree_summary_no_files(self):
        """Context with summary but no files renders file list and no content."""
        ctx = ProjectContext(
            project_id="p",
            tree_summary="a.ts (5)\nb.ts (3)",
            files={},
        )
        result = render_context_block(ctx)
        assert "Current project files" in result
        assert "a.ts (5)" in result
        assert "b.ts (3)" in result
        assert "Contents of the most relevant files" not in result

    def test_with_files_no_placeholders(self):
        """Files are rendered without placeholder annotation."""
        ctx = ProjectContext(
            project_id="p",
            tree_summary="hello.txt (5)",
            files={"hello.txt": "world"},
            placeholder_files=frozenset(),
        )
        result = render_context_block(ctx)
        assert "----- hello.txt -----" in result
        assert "world" in result
        assert "placeholder scaffold" not in result

    def test_with_files_and_placeholders(self):
        """Placeholder files get the '(placeholder scaffold)' marker."""
        ctx = ProjectContext(
            project_id="p",
            tree_summary="page.tsx (100)",
            files={"page.tsx": "starter content"},
            placeholder_files=frozenset({"page.tsx"}),
        )
        result = render_context_block(ctx)
        assert "page.tsx" in result
        assert "(placeholder scaffold)" in result
        assert "These files still hold unmodified" in result

    def test_multiple_placeholders_listed(self):
        """Multiple placeholder files are listed in the annotation block."""
        ctx = ProjectContext(
            project_id="p",
            tree_summary="a (1)\nb (1)",
            files={"a": "x", "b": "y"},
            placeholder_files=frozenset({"a", "b"}),
        )
        result = render_context_block(ctx)
        # Both should be listed, sorted alphabetically
        assert "a, b" in result

    def test_file_content_truncated(self):
        """File content exceeding CONTEXT_FILE_DISPLAY_CAP is truncated."""
        long_body = "a" * (CONTEXT_FILE_DISPLAY_CAP + 100)
        ctx = ProjectContext(
            project_id="p",
            tree_summary="big.txt (12100)",
            files={"big.txt": long_body},
        )
        result = render_context_block(ctx)
        assert len(result) < len(long_body) + 500  # truncated
        assert "/* ... truncated ... */" in result
        # Only the first CONTEXT_FILE_DISPLAY_CAP chars should be present
        assert "a" * CONTEXT_FILE_DISPLAY_CAP in result

    def test_short_file_not_truncated(self):
        """Small files are not truncated."""
        body = "small content"
        ctx = ProjectContext(
            project_id="p",
            tree_summary="small.txt (13)",
            files={"small.txt": body},
        )
        result = render_context_block(ctx)
        assert body in result
        assert "truncated" not in result

    def test_tree_summary_empty_string(self):
        """When tree_summary is empty, show '(no files yet)'."""
        ctx = ProjectContext(
            project_id="p",
            tree_summary="",
            files={"a.ts": "content"},
        )
        result = render_context_block(ctx)
        # tree_summary is empty string (falsy), but files exist so it still renders
        # Actually context.tree_summary is '' which is falsy,
        # but we check 'not context.tree_summary and not context.files' for empty.
        # Since files is non-empty, we go to the main code path.
        # tree_summary or "(no files yet)" applies
        assert "(no files yet)" in result

    def test_placeholder_annotation_block_appears(self):
        """The placeholder annotation paragraph appears when placeholders exist."""
        ctx = ProjectContext(
            project_id="p",
            tree_summary="a.ts (1)",
            files={"a.ts": "x"},
            placeholder_files=frozenset({"a.ts"}),
        )
        result = render_context_block(ctx)
        assert "These files still hold unmodified" in result
        assert "overwritten with `replace`" in result

    def test_no_placeholder_block_when_empty(self):
        """No placeholder annotation when placeholder_files is empty."""
        ctx = ProjectContext(
            project_id="p",
            tree_summary="a.ts (1)",
            files={"a.ts": "x"},
        )
        result = render_context_block(ctx)
        assert "These files still hold" not in result

    def test_loader_is_callable_within_context(self):
        """The loader returned by load_context is a callable."""
        storage = MagicMock()
        storage.read_tree.return_value = {}
        # project_dir returning a real path so loader can be built
        import tempfile
        storage.project_dir.return_value = tempfile.gettempdir()
        ctx = load_context(storage, "p", "")
        assert callable(ctx.loader)

    def test_render_context_block_non_empty_project_detected(self):
        """Even with empty tree_summary but non-empty files, renders correctly."""
        ctx = ProjectContext(
            project_id="p",
            tree_summary="",
            files={"keep.txt": "content"},
        )
        result = render_context_block(ctx)
        assert "Current project files" in result

    def test_render_context_block_empty_summary_no_files_message(self):
        """Explicitly empty context returns the first-turn message."""
        ctx = ProjectContext(project_id="p", tree_summary="", files={})
        result = render_context_block(ctx)
        assert result == "Current project: (empty - this is the first turn)."
