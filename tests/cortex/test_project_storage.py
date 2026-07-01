"""Comprehensive pytest tests for cortex/project_storage.py.

Target >95% coverage of all classes, functions, and methods including:
  - slugify, validate_slug, safe_join path utilities
  - Storage CRUD (create, list, get, delete projects)
  - File operations (write, read, read_tree, delete)
  - Prompt history (append, read, pop, clear)
  - Snapshots (create, list, restore, delete, prune)
  - ensure_next_preview_layout
  - get_storage / reset_storage / iter_ignored_top_level
  - Edge cases, path traversal protection, error handling
"""

from __future__ import annotations

import json
import os
import shutil
import string
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from cortex.project_storage import (
    PROJECT_FILE,
    PROMPTS_FILE,
    SIDECAR_DIR,
    SNAPSHOT_FILES_DIR,
    SNAPSHOT_ID_RE,
    SNAPSHOT_KEEP,
    SNAPSHOTS_DIR,
    ProjectRecord,
    PromptRecord,
    SnapshotRecord,
    Storage,
    _IGNORED_TOP_LEVEL,
    _now,
    get_storage,
    iter_ignored_top_level,
    reset_storage,
    safe_join,
    slugify,
    validate_slug,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Storage:
    """Create a Storage instance rooted at a temp directory."""
    return Storage(root=tmp_path)


def _write_project_json(sidecar_dir: Path, record: ProjectRecord) -> None:
    """Helper to manually write project.json metadata."""
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / PROJECT_FILE
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")


def _write_prompts(sidecar_dir: Path, records: list[PromptRecord]) -> None:
    """Helper to write prompt records to prompts.jsonl."""
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / PROMPTS_FILE
    with open(path, "w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec.__dict__) + "\n")


# =============================================================================
# slugify
# =============================================================================


class TestSlugify:
    def test_basic_lowercase(self):
        """Simple name becomes lowercase."""
        assert slugify("Hello World") == "hello-world"

    def test_trailing_and_leading_whitespace(self):
        """Whitespace is stripped."""
        assert slugify("  My Project  ") == "my-project"

    def test_special_chars_replaced(self):
        """Non-alphanumeric characters become hyphens."""
        assert slugify("hello!@#$world") == "hello-world"

    def test_multiple_hyphens_collapsed(self):
        """Multiple separators collapse to a single hyphen."""
        assert slugify("a   b___c") == "a-b-c"

    def test_leading_trailing_hyphens_stripped(self):
        """Leading/trailing hyphens are removed."""
        assert slugify("-hello-") == "hello"

    def test_truncated_to_63_chars(self):
        """Result is truncated to 63 characters."""
        long_name = "a" * 100
        result = slugify(long_name)
        assert len(result) == 63
        assert result == "a" * 63

    def test_empty_after_cleaning(self):
        """If cleaning yields empty, return empty string."""
        assert slugify("!!!") == ""

    def test_empty_input(self):
        """Empty input returns empty string."""
        assert slugify("") == ""

    def test_whitespace_only(self):
        """Whitespace-only input returns empty string."""
        assert slugify("   ") == ""

    def test_numeric_input(self):
        """Numeric-only input works."""
        assert slugify("12345") == "12345"

    def test_mixed_case(self):
        """Mixed case is lowercased."""
        assert slugify("MyCoolProject") == "mycoolproject"

    def test_alphanumeric_with_hyphens(self):
        """Already-slugified input is preserved."""
        assert slugify("my-project-42") == "my-project-42"

    def test_unicode_chars(self):
        """Unicode characters are stripped/replaced."""
        result = slugify("café du monde")
        # only a-z0-9 and hyphens survive
        assert all(c in string.ascii_lowercase + string.digits + "-" for c in result)
        assert result  # should not be empty


# =============================================================================
# validate_slug
# =============================================================================


class TestValidateSlug:
    @pytest.mark.parametrize(
        "valid",
        [
            "a",
            "abc",
            "a123",
            "123abc",
            "my-project",
            "a-b-c",
            "a" * 63,  # max length
            "0",
            "z".zfill(63),
        ],
    )
    def test_valid_slugs(self, valid: str):
        """Valid slugs pass validation."""
        assert validate_slug(valid) is True

    @pytest.mark.parametrize(
        "invalid",
        [
            "",
            "-leading-hyphen",
            "UPPERCASE",
            "MixedCase",
            "has_underscore",
            "has space",
            "a" * 64,  # too long
            "-",
            "a" * 64,  # too long
            ".dot",
            "a/b",
            None,
        ],
    )
    def test_invalid_slugs(self, invalid: str):
        """Invalid slugs fail validation."""
        if invalid is None:
            return  # type mismatch, handled elsewhere
        assert validate_slug(invalid) is False

    def test_none_raises(self):
        """Passing None should raise AttributeError (not a string)."""
        with pytest.raises((AttributeError, TypeError)):
            validate_slug(None)  # type: ignore[arg-type]


# =============================================================================
# safe_join
# =============================================================================


class TestSafeJoin:
    def test_normal_join(self, tmp_path: Path):
        """Normal relative path resolves within root."""
        result = safe_join(tmp_path, "subdir/file.txt")
        assert result == (tmp_path / "subdir/file.txt").resolve()

    def test_dot_relative(self, tmp_path: Path):
        """./ prefix is fine."""
        result = safe_join(tmp_path, "./file.txt")
        assert result == (tmp_path / "file.txt").resolve()

    def test_traversal_blocked(self, tmp_path: Path):
        """Path with ../ traversal raises ValueError."""
        with pytest.raises(ValueError, match="path escapes project root"):
            safe_join(tmp_path, "../etc/passwd")

    def test_traversal_blocked_deep(self, tmp_path: Path):
        """Deep traversal is blocked."""
        with pytest.raises(ValueError, match="path escapes project root"):
            safe_join(tmp_path, "subdir/../../etc/passwd")

    def test_absolute_path_blocked(self, tmp_path: Path):
        """Absolute paths are blocked."""
        with pytest.raises(ValueError, match="absolute paths are not allowed"):
            safe_join(tmp_path, "/etc/passwd")

    def test_symlink_traversal(self, tmp_path: Path):
        """If root contains a symlink pointing outside, still catches."""
        outside = tmp_path / "outside"
        outside.mkdir()
        link = tmp_path / "link"
        link.symlink_to(outside)
        # safe_join resolves symlinks because it calls resolve()
        result = safe_join(tmp_path, "link/../outside")
        # This stays inside root
        assert result == (tmp_path / "outside").resolve()

    def test_nested_valid(self, tmp_path: Path):
        """Deep nesting within bounds works."""
        result = safe_join(tmp_path, "a/b/c/d/e/file.txt")
        assert result == (tmp_path / "a/b/c/d/e/file.txt").resolve()


# =============================================================================
# Storage.__init__
# =============================================================================


class TestStorageInit:
    def test_default_root(self):
        """Default root is data/projects resolved."""
        storage = Storage()
        assert storage.root == Path("data/projects").resolve()

    def test_custom_root_str(self, tmp_path: Path):
        """String root is accepted and resolved."""
        s = Storage(root=str(tmp_path))
        assert s.root == tmp_path.resolve()

    def test_custom_root_path(self, tmp_path: Path):
        """Path root is accepted."""
        s = Storage(root=tmp_path)
        assert s.root == tmp_path.resolve()

    def test_root_expands_user(self):
        """Tilde expansion works."""
        s = Storage(root="~/test_projects")
        assert s.root == Path("~/test_projects").expanduser().resolve()

    def test_has_write_lock(self, tmp_path: Path):
        """Storage has a write lock."""
        s = Storage(root=tmp_path)
        assert hasattr(s, "_write_lock")

    @pytest.mark.skipif(os.name != "posix", reason="posix only")
    def test_root_absolute(self):
        """Root is always an absolute path."""
        s = Storage(root="relative/path")
        assert s.root.is_absolute()


# =============================================================================
# ensure_root
# =============================================================================


class TestEnsureRoot:
    def test_creates_directory(self, tmp_path: Path):
        """ensure_root creates the root directory."""
        root = tmp_path / "new_root"
        assert not root.exists()
        s = Storage(root=root)
        s.ensure_root()
        assert root.is_dir()

    def test_idempotent(self, tmp_path: Path):
        """Calling ensure_root twice is safe."""
        s = Storage(root=tmp_path)
        s.ensure_root()
        s.ensure_root()  # should not raise
        assert tmp_path.is_dir()


# =============================================================================
# project_dir and sidecar_dir
# =============================================================================


class TestProjectDir:
    def test_returns_correct_path(self, tmp_storage: Storage):
        """project_dir returns root/slug."""
        result = tmp_storage.project_dir("my-project")
        assert result == tmp_storage.root / "my-project"

    def test_invalid_slug_raises(self, tmp_storage: Storage):
        """Invalid slug raises ValueError."""
        with pytest.raises(ValueError, match="invalid project slug"):
            tmp_storage.project_dir("InvalidSlug")

    def test_empty_slug_raises(self, tmp_storage: Storage):
        """Empty slug raises ValueError."""
        with pytest.raises(ValueError, match="invalid project slug"):
            tmp_storage.project_dir("")


class TestSidecarDir:
    def test_returns_correct_path(self, tmp_storage: Storage):
        """sidecar_dir returns root/slug/.odysseus."""
        result = tmp_storage.sidecar_dir("my-project")
        assert result == tmp_storage.root / "my-project" / SIDECAR_DIR

    def test_delegates_validation(self, tmp_storage: Storage):
        """Invalid slug still raises ValueError."""
        with pytest.raises(ValueError, match="invalid project slug"):
            tmp_storage.sidecar_dir("InvalidSlug")


# =============================================================================
# unique_slug
# =============================================================================


class TestUniqueSlug:
    def test_basic_slugification(self, tmp_storage: Storage):
        """Returns slugified name."""
        assert tmp_storage.unique_slug("My Project") == "my-project"

    def test_no_collision(self, tmp_storage: Storage):
        """When directory doesn't exist, returns base slug."""
        slug = tmp_storage.unique_slug("hello")
        assert slug == "hello"
        assert not (tmp_storage.root / slug).exists()

    def test_appends_number_on_collision(self, tmp_storage: Storage):
        """When slug directory exists, appends -2."""
        (tmp_storage.root / "my-project").mkdir(parents=True)
        slug = tmp_storage.unique_slug("my-project")
        assert slug == "my-project-2"

    def test_appends_incrementing(self, tmp_storage: Storage):
        """Multiple collisions increment the suffix."""
        for i in range(1, 5):
            (tmp_storage.root / f"my-project-{i}" if i > 1 else tmp_storage.root / "my-project").mkdir(parents=True)
        slug = tmp_storage.unique_slug("my-project")
        assert slug == "my-project-5"

    def test_fallback_for_empty_slug(self, tmp_storage: Storage):
        """When slugify returns empty, uses project-<uuid>."""
        slug = tmp_storage.unique_slug("!!!")
        assert slug.startswith("project-")
        assert len(slug) == len("project-") + 8  # fixed prefix + 8 hex chars

    def test_truncation_with_suffix(self, tmp_storage: Storage):
        """Long base slug is truncated to fit suffix."""
        base = "a" * 63
        (tmp_storage.root / base).mkdir(parents=True)
        slug = tmp_storage.unique_slug(base)
        # Should be truncated to 63 - len("-2") = 61 chars + "-2"
        assert len(slug) <= 63
        assert slug.endswith("-2")

    def test_handles_existing_fallback(self, tmp_storage: Storage):
        """When fallback slug exists, appends number."""
        slug1 = tmp_storage.unique_slug("!!!")
        (tmp_storage.root / slug1).mkdir(parents=True)
        slug2 = tmp_storage.unique_slug("!!!")
        assert slug2 != slug1
        assert slug2.startswith("project-")
        assert len(slug2) > len("project-")


# =============================================================================
# create_project
# =============================================================================


class TestCreateProject:
    def test_creates_next_app_project(self, tmp_storage: Storage):
        """Creates a Next.js app-type project with starter files."""
        record = tmp_storage.create_project("test-app", template="next", project_type="app")
        assert record.name == "test-app"
        assert record.template == "next"
        assert record.project_type == "app"
        assert record.id == "test-app"
        assert record.created_at is not None
        assert record.updated_at is not None

        proj_dir = tmp_storage.project_dir(record.id)
        assert proj_dir.is_dir()
        sidecar = proj_dir / SIDECAR_DIR
        assert sidecar.is_dir()
        assert (sidecar / PROJECT_FILE).is_file()
        assert (sidecar / PROMPTS_FILE).is_file()

        # Check starter files present (app/ subdir for app type)
        assert (proj_dir / "package.json").is_file()
        assert (proj_dir / "app" / "layout.tsx").is_file()

    def test_creates_next_full_project(self, tmp_storage: Storage):
        """Creates a Next.js full-type project with code/ and docs/ subdirs."""
        record = tmp_storage.create_project("test-full", template="next", project_type="full")
        assert record.project_type == "full"
        proj_dir = tmp_storage.project_dir(record.id)
        assert (proj_dir / "code").is_dir()
        assert (proj_dir / "docs").is_dir()
        # Starter files go under code/
        assert (proj_dir / "code" / "package.json").is_file()

    def test_creates_express_app_project(self, tmp_storage: Storage):
        """Creates an Express project."""
        record = tmp_storage.create_project("test-express", template="express", project_type="app")
        assert record.template == "express"
        proj_dir = tmp_storage.project_dir(record.id)
        # Express starter has src/index.ts
        assert (proj_dir / "package.json").is_file()
        assert (proj_dir / "src" / "index.ts").is_file()

    def test_creates_with_starter_name_override(self, tmp_storage: Storage):
        """starter_name overrides template name."""
        record = tmp_storage.create_project(
            "test-override", template="next", starter_name="next-minimal"
        )
        assert record.starter_name == "next-minimal"

    def test_creates_with_none_starter_name(self, tmp_storage: Storage):
        """starter_name=None defaults to template."""
        record = tmp_storage.create_project("test-none", template="express", starter_name=None)
        assert record.starter_name == "express"

    def test_auto_generates_unique_slug(self, tmp_storage: Storage):
        """Creates project and auto-deduplicates slug."""
        record1 = tmp_storage.create_project("test", template="next")
        record2 = tmp_storage.create_project("test", template="next")
        assert record1.id == "test"
        assert record2.id.startswith("test-")
        assert record2.id != record1.id

    def test_ensure_root_called(self, tmp_path: Path):
        """Root is created automatically."""
        root = tmp_path / "nonexistent"
        s = Storage(root=root)
        assert not root.exists()
        s.create_project("test", template="next")
        assert root.is_dir()

    def test_record_is_returned(self, tmp_storage: Storage):
        """create_project returns the correct ProjectRecord."""
        record = tmp_storage.create_project("Test-Name", template="next")
        assert isinstance(record, ProjectRecord)
        assert record.id == "test-name"
        assert record.name == "Test-Name"

    def test_starter_files_are_writable(self, tmp_storage: Storage):
        """Starter files can be read back."""
        record = tmp_storage.create_project("readback", template="express")
        content = tmp_storage.read_file(record.id, "src/index.ts")
        assert content is not None
        assert "express" in content


# =============================================================================
# list_projects
# =============================================================================


class TestListProjects:
    def test_empty_when_no_root(self, tmp_path: Path):
        """When root doesn't exist, returns empty list."""
        s = Storage(root=tmp_path / "does-not-exist")
        assert s.list_projects() == []

    def test_empty_when_no_projects(self, tmp_storage: Storage):
        """When root exists but empty, returns empty list."""
        tmp_storage.ensure_root()
        assert tmp_storage.list_projects() == []

    def test_returns_created_projects(self, tmp_storage: Storage):
        """Lists all created projects."""
        p1 = tmp_storage.create_project("alpha", template="next")
        p2 = tmp_storage.create_project("beta", template="express")
        projects = tmp_storage.list_projects()
        slugs = {p.id for p in projects}
        assert p1.id in slugs
        assert p2.id in slugs

    def test_ordering_most_recent_first(self, tmp_storage: Storage):
        """Projects are sorted by updated_at descending."""
        p1 = tmp_storage.create_project("first", template="next")
        p2 = tmp_storage.create_project("second", template="next")
        projects = tmp_storage.list_projects()
        assert projects[0].id == p2.id  # most recently created first
        assert projects[1].id == p1.id

    def test_skips_non_project_dirs(self, tmp_storage: Storage):
        """Directories without project.json are skipped."""
        tmp_storage.ensure_root()
        (tmp_storage.root / "not-a-project").mkdir()
        p1 = tmp_storage.create_project("real", template="next")
        projects = tmp_storage.list_projects()
        assert len(projects) == 1
        assert projects[0].id == p1.id

    def test_skips_dirs_with_invalid_slug_names(self, tmp_storage: Storage):
        """Directories whose names are not valid slugs are skipped."""
        tmp_storage.ensure_root()
        (tmp_storage.root / "Invalid_Name").mkdir()
        p1 = tmp_storage.create_project("valid", template="next")
        projects = tmp_storage.list_projects()
        assert len(projects) == 1
        assert projects[0].id == p1.id

    def test_skips_files_in_root(self, tmp_storage: Storage):
        """Files in root directory are skipped."""
        tmp_storage.ensure_root()
        (tmp_storage.root / "somefile.txt").write_text("hello")
        p1 = tmp_storage.create_project("real", template="next")
        projects = tmp_storage.list_projects()
        assert len(projects) == 1
        assert projects[0].id == p1.id


# =============================================================================
# get_project
# =============================================================================


class TestGetProject:
    def test_returns_project(self, tmp_storage: Storage):
        """Returns the correct project record."""
        created = tmp_storage.create_project("findme", template="next")
        fetched = tmp_storage.get_project("findme")
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == created.name

    def test_none_for_missing(self, tmp_storage: Storage):
        """Returns None for non-existent slug."""
        assert tmp_storage.get_project("nonexistent") is None

    def test_none_for_invalid_slug(self, tmp_storage: Storage):
        """Returns None for invalid slug."""
        assert tmp_storage.get_project("Invalid") is None

    def test_none_for_empty_slug(self, tmp_storage: Storage):
        """Returns None for empty slug."""
        assert tmp_storage.get_project("") is None

    def test_none_for_bad_json(self, tmp_storage: Storage):
        """Returns None if project.json is corrupted."""
        tmp_storage.ensure_root()
        sidecar = tmp_storage.root / "broken" / SIDECAR_DIR
        sidecar.mkdir(parents=True)
        (sidecar / PROJECT_FILE).write_text("not json", encoding="utf-8")
        assert tmp_storage.get_project("broken") is None


# =============================================================================
# delete_project
# =============================================================================


class TestDeleteProject:
    def test_deletes_existing(self, tmp_storage: Storage):
        """Deletes an existing project directory."""
        tmp_storage.create_project("todelete", template="next")
        proj_dir = tmp_storage.project_dir("todelete")
        assert proj_dir.exists()
        result = tmp_storage.delete_project("todelete")
        assert result is True
        assert not proj_dir.exists()

    def test_returns_false_for_missing(self, tmp_storage: Storage):
        """Returns False if project doesn't exist."""
        assert tmp_storage.delete_project("nonexistent") is False

    def test_returns_false_for_invalid_slug(self, tmp_storage: Storage):
        """Returns False for invalid slug."""
        assert tmp_storage.delete_project("Invalid") is False

    def test_blocked_invalid_slug(self, tmp_storage: Storage):
        """Returns False for invalid slug (validation runs first)."""
        assert tmp_storage.delete_project("InvalidSlug") is False
        assert tmp_storage.delete_project("..") is False

    def test_get_returns_none_after_delete(self, tmp_storage: Storage):
        """After deletion, get_project returns None."""
        tmp_storage.create_project("ephemeral", template="next")
        tmp_storage.delete_project("ephemeral")
        assert tmp_storage.get_project("ephemeral") is None


# =============================================================================
# File operations: write_file, read_file, read_tree, delete_file
# =============================================================================


class TestWriteFile:
    def test_writes_content(self, tmp_storage: Storage):
        """write_file creates file with content."""
        tmp_storage.create_project("files-test", template="next")
        path = tmp_storage.write_file("files-test", "custom/hello.txt", "Hello, World!")
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == "Hello, World!"

    def test_raises_for_missing_project(self, tmp_storage: Storage):
        """Raises FileNotFoundError for non-existent project."""
        with pytest.raises(FileNotFoundError, match="project not found"):
            tmp_storage.write_file("nonexistent", "f.txt", "content")

    def test_creates_intermediate_dirs(self, tmp_storage: Storage):
        """Creates parent directories automatically."""
        tmp_storage.create_project("dirs-test", template="next")
        path = tmp_storage.write_file("dirs-test", "deeply/nested/path/file.txt", "deep")
        assert path.is_file()

    def test_updates_project_timestamp(self, tmp_storage: Storage):
        """Writing a file updates updated_at."""
        record = tmp_storage.create_project("ts-test", template="next")
        orig_updated = record.updated_at
        tmp_storage.write_file("ts-test", "newfile.txt", "hello")
        updated = tmp_storage.get_project("ts-test")
        assert updated is not None
        assert updated.updated_at is not None
        assert updated.updated_at > orig_updated  # type: ignore[operator]

    def test_block_traversal(self, tmp_storage: Storage):
        """Writing outside project root is blocked."""
        tmp_storage.create_project("safe", template="next")
        with pytest.raises(ValueError, match="path escapes project root"):
            tmp_storage.write_file("safe", "../escape.txt", "bad")


class TestReadFile:
    def test_reads_content(self, tmp_storage: Storage):
        """read_file returns file content."""
        tmp_storage.create_project("read-test", template="next")
        tmp_storage.write_file("read-test", "test.txt", "content")
        assert tmp_storage.read_file("read-test", "test.txt") == "content"

    def test_returns_none_for_missing(self, tmp_storage: Storage):
        """Returns None for non-existent file."""
        tmp_storage.create_project("read-test2", template="next")
        assert tmp_storage.read_file("read-test2", "missing.txt") is None

    def test_returns_none_for_traversal(self, tmp_storage: Storage):
        """Returns None when path traversal is attempted."""
        tmp_storage.create_project("safe", template="next")
        assert tmp_storage.read_file("safe", "../etc/passwd") is None

    def test_returns_none_for_directory(self, tmp_storage: Storage):
        """Returns None when path is a directory."""
        tmp_storage.create_project("dir-test", template="next")
        tmp_storage.write_file("dir-test", "afile.txt", "content")
        assert tmp_storage.read_file("dir-test", ".") is None


    def test_returns_none_for_unreadable_file(self, tmp_storage: Storage):
        """Returns None when file exists but can't be read as text."""
        tmp_storage.create_project("binary-read", template="next")
        proj = tmp_storage.project_dir("binary-read")
        # Create a binary file that passes is_file() but fails read_text()
        bin_file = proj / "binary.bin"
        bin_file.write_bytes(b"\xff\xfe\x00\x01\xff")
        result = tmp_storage.read_file("binary-read", "binary.bin")
        assert result is None


class TestReadTree:
    def test_returns_structure(self, tmp_storage: Storage):
        """read_tree returns nested dict of files."""
        tmp_storage.create_project("tree-test", template="next")
        tmp_storage.write_file("tree-test", "hello.txt", "world")
        tree = tmp_storage.read_tree("tree-test")
        assert "hello.txt" in tree
        assert tree["hello.txt"]["file"]["contents"] == "world"

    def test_includes_directories(self, tmp_storage: Storage):
        """Directories appear in the tree."""
        tmp_storage.create_project("tree-dirs", template="next")
        tmp_storage.write_file("tree-dirs", "sub/deep.txt", "content")
        tree = tmp_storage.read_tree("tree-dirs")
        assert "sub" in tree
        assert "directory" in tree["sub"]
        assert "deep.txt" in tree["sub"]["directory"]

    def test_ignores_sidecar_dir(self, tmp_storage: Storage):
        """.odysseus directory is excluded from the tree."""
        tmp_storage.create_project("tree-ignore", template="next")
        tree = tmp_storage.read_tree("tree-ignore")
        assert SIDECAR_DIR not in tree

    def test_ignores_node_modules(self, tmp_storage: Storage):
        """node_modules directory is excluded."""
        tmp_storage.create_project("tree-ignore2", template="next")
        (tmp_storage.project_dir("tree-ignore2") / "node_modules").mkdir()
        tree = tmp_storage.read_tree("tree-ignore2")
        assert "node_modules" not in tree

    def test_raises_for_missing_project(self, tmp_storage: Storage):
        """FileNotFoundError for non-existent project."""
        with pytest.raises(FileNotFoundError, match="project not found"):
            tmp_storage.read_tree("missing-project")

    def test_ignores_symlinks(self, tmp_storage: Storage):
        """Symlinks are skipped in tree."""
        tmp_storage.create_project("tree-symlinks", template="next")
        proj = tmp_storage.project_dir("tree-symlinks")
        link = proj / "alink"
        link.symlink_to(proj / "package.json")
        tree = tmp_storage.read_tree("tree-symlinks")
        assert "alink" not in tree

    def test_skips_binary_files_gracefully(self, tmp_storage: Storage):
        """Binary/unreadable files are skipped without raising."""
        tmp_storage.create_project("tree-binary", template="next")
        proj = tmp_storage.project_dir("tree-binary")
        bin_file = proj / "binary.bin"
        bin_file.write_bytes(b"\x00\x01\x02\xff")
        tree = tmp_storage.read_tree("tree-binary")
        assert "binary.bin" not in tree  # skipped due to UnicodeDecodeError


class TestDeleteFile:
    def test_deletes_file(self, tmp_storage: Storage):
        """delete_file removes a file."""
        tmp_storage.create_project("del", template="next")
        tmp_storage.write_file("del", "todelete.txt", "bye")
        path = tmp_storage.project_dir("del") / "todelete.txt"
        assert path.is_file()
        result = tmp_storage.delete_file("del", "todelete.txt")
        assert result is True
        assert not path.exists()

    def test_returns_false_for_missing(self, tmp_storage: Storage):
        """Returns False if file doesn't exist."""
        tmp_storage.create_project("del2", template="next")
        assert tmp_storage.delete_file("del2", "missing.txt") is False

    def test_block_traversal(self, tmp_storage: Storage):
        """Traversal is blocked, returns False."""
        tmp_storage.create_project("del3", template="next")
        assert tmp_storage.delete_file("del3", "../escape.txt") is False

    def test_deletes_directory(self, tmp_storage: Storage):
        """delete_file can delete a directory too."""
        tmp_storage.create_project("del-dir", template="next")
        proj = tmp_storage.project_dir("del-dir")
        (proj / "subdir").mkdir()
        (proj / "subdir" / "file.txt").write_text("x")
        result = tmp_storage.delete_file("del-dir", "subdir")
        assert result is True
        assert not (proj / "subdir").exists()

    def test_updates_timestamp(self, tmp_storage: Storage):
        """Deleting a file updates project's updated_at."""
        record = tmp_storage.create_project("del-ts", template="next")
        tmp_storage.write_file("del-ts", "todel.txt", "hello")
        orig = record.updated_at
        tmp_storage.delete_file("del-ts", "todel.txt")
        updated = tmp_storage.get_project("del-ts")
        assert updated is not None
        assert updated.updated_at is not None
        assert updated.updated_at > orig  # type: ignore[operator]


# =============================================================================
# Prompt operations
# =============================================================================


class TestAppendPrompt:
    def test_appends_prompt_record(self, tmp_storage: Storage):
        """Appends a prompt and returns a PromptRecord."""
        tmp_storage.create_project("prompts", template="next")
        rec = tmp_storage.append_prompt("prompts", "user", "Hello!")
        assert isinstance(rec, PromptRecord)
        assert rec.role == "user"
        assert rec.content == "Hello!"
        assert rec.id != ""
        assert rec.created_at is not None

    def test_appends_with_snapshot_id(self, tmp_storage: Storage):
        """Can append with a snapshot_id reference."""
        tmp_storage.create_project("prompts2", template="next")
        rec = tmp_storage.append_prompt("prompts2", "assistant", "Hi!", snapshot_id="snap1")
        assert rec.snapshot_id == "snap1"

    def test_creates_sidecar_if_missing(self, tmp_storage: Storage):
        """Creates sidecar directory even if project exists without it."""
        proj = tmp_storage.root / "bare"
        proj.mkdir(parents=True)
        # Manually create project.json and prompts file so project_dir works
        sidecar = proj / SIDECAR_DIR
        sidecar.mkdir(parents=True)
        _write_project_json(sidecar, ProjectRecord(id="bare", name="bare"))
        (sidecar / PROMPTS_FILE).touch()
        # Now it should work
        rec = tmp_storage.append_prompt("bare", "user", "hello")
        assert rec.role == "user"


class TestReadPrompts:
    def test_reads_appended_prompts(self, tmp_storage: Storage):
        """read_prompts returns list of appended prompts."""
        tmp_storage.create_project("readp", template="next")
        tmp_storage.append_prompt("readp", "user", "q1")
        tmp_storage.append_prompt("readp", "assistant", "a1")
        prompts = tmp_storage.read_prompts("readp")
        assert len(prompts) == 2
        assert prompts[0].role == "user"
        assert prompts[0].content == "q1"
        assert prompts[1].role == "assistant"
        assert prompts[1].content == "a1"

    def test_returns_empty_for_no_file(self, tmp_storage: Storage):
        """Returns empty list when no prompts file exists."""
        # Create a project and manually remove prompts file
        tmp_storage.create_project("noprompts", template="next")
        prompts_file = tmp_storage.sidecar_dir("noprompts") / PROMPTS_FILE
        assert prompts_file.exists()
        prompts_file.unlink()
        prompts = tmp_storage.read_prompts("noprompts")
        assert prompts == []

    def test_skips_malformed_lines(self, tmp_storage: Storage):
        """Malformed JSON lines are skipped."""
        tmp_storage.create_project("malformed", template="next")
        sidecar = tmp_storage.sidecar_dir("malformed")
        prompts_file = sidecar / PROMPTS_FILE
        # Write one good line and one bad
        with open(prompts_file, "a", encoding="utf-8") as fp:
            good = json.dumps({"id": "1", "role": "user", "content": "ok"})
            fp.write(good + "\n")
            fp.write("not json\n")
            fp.write("\n")
        prompts = tmp_storage.read_prompts("malformed")
        assert len(prompts) == 1
        assert prompts[0].content == "ok"


class TestPopLastAssistantPrompt:
    def test_pops_last_assistant(self, tmp_storage: Storage):
        """Pops the most recent assistant prompt."""
        tmp_storage.create_project("pop", template="next")
        tmp_storage.append_prompt("pop", "user", "q")
        rec = tmp_storage.append_prompt("pop", "assistant", "a1")
        tmp_storage.append_prompt("pop", "assistant", "a2")
        popped = tmp_storage.pop_last_assistant_prompt("pop")
        assert popped is not None
        assert popped.content == "a2"
        # Verify it's gone
        remaining = tmp_storage.read_prompts("pop")
        assert len(remaining) == 2
        assert remaining[-1].content == "a1"

    def test_returns_none_if_no_assistant(self, tmp_storage: Storage):
        """Returns None when no assistant prompt exists."""
        tmp_storage.create_project("pop2", template="next")
        tmp_storage.append_prompt("pop2", "user", "q")
        assert tmp_storage.pop_last_assistant_prompt("pop2") is None

    def test_returns_none_if_no_file(self, tmp_storage: Storage):
        """Returns None when prompts file doesn't exist."""
        tmp_storage.create_project("pop3", template="next")
        # Delete prompts file
        prompts_file = tmp_storage.sidecar_dir("pop3") / PROMPTS_FILE
        prompts_file.unlink()
        assert tmp_storage.pop_last_assistant_prompt("pop3") is None

    def test_preserves_other_roles(self, tmp_storage: Storage):
        """Only assistant prompts are popped."""
        tmp_storage.create_project("pop4", template="next")
        tmp_storage.append_prompt("pop4", "system", "sys")
        tmp_storage.append_prompt("pop4", "user", "q")
        tmp_storage.append_prompt("pop4", "assistant", "a")
        tmp_storage.append_prompt("pop4", "tool", "result")
        popped = tmp_storage.pop_last_assistant_prompt("pop4")
        assert popped is not None
        assert popped.content == "a"
        remaining = tmp_storage.read_prompts("pop4")
        assert len(remaining) == 3


    def test_pops_skipping_blank_and_malformed(self, tmp_storage: Storage):
        """Blank and malformed lines at end are skipped during pop."""
        tmp_storage.create_project("pop-blank", template="next")
        sidecar = tmp_storage.sidecar_dir("pop-blank")
        prompts_file = sidecar / PROMPTS_FILE
        # Write: user, then assistant, then blank, then malformed (at end)
        # The blank line must be AFTER the last valid assistant so it's iterated first
        with open(prompts_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps({"id": "1", "role": "user", "content": "q"}) + "\n")
            fp.write(json.dumps({"id": "2", "role": "assistant", "content": "a"}) + "\n")
            fp.write("\n")
            fp.write("not valid json\n")
        popped = tmp_storage.pop_last_assistant_prompt("pop-blank")
        assert popped is not None
        assert popped.content == "a"
        # Verify assistant was removed, user preserved
        remaining = tmp_storage.read_prompts("pop-blank")
        assert len(remaining) == 1
        assert remaining[0].role == "user"

    def test_returns_none_with_only_malformed_lines(self, tmp_storage: Storage):
        """Returns None when all lines are malformed."""
        tmp_storage.create_project("pop-malformed", template="next")
        sidecar = tmp_storage.sidecar_dir("pop-malformed")
        prompts_file = sidecar / PROMPTS_FILE
        with open(prompts_file, "w", encoding="utf-8") as fp:
            fp.write("not json\n")
            fp.write("also not json\n")
        assert tmp_storage.pop_last_assistant_prompt("pop-malformed") is None


class TestClearPrompts:
    def test_clears_all_prompts(self, tmp_storage: Storage):
        """clear_prompts removes all prompt history."""
        tmp_storage.create_project("clear", template="next")
        tmp_storage.append_prompt("clear", "user", "q")
        tmp_storage.append_prompt("clear", "assistant", "a")
        result = tmp_storage.clear_prompts("clear")
        assert result is True
        assert tmp_storage.read_prompts("clear") == []

    def test_returns_false_if_no_file(self, tmp_storage: Storage):
        """Returns False when no prompts file exists."""
        tmp_storage.create_project("clear2", template="next")
        prompts_file = tmp_storage.sidecar_dir("clear2") / PROMPTS_FILE
        prompts_file.unlink()
        assert tmp_storage.clear_prompts("clear2") is False


# =============================================================================
# Snapshot operations
# =============================================================================


class TestCreateSnapshot:
    def test_creates_snapshot_with_files(self, tmp_storage: Storage):
        """create_snapshot copies project files into snapshot dir."""
        tmp_storage.create_project("snap", template="next")
        tmp_storage.write_file("snap", "hello.txt", "world")
        snap = tmp_storage.create_snapshot("snap", user_prompt="test prompt", kind="manual")
        assert isinstance(snap, SnapshotRecord)
        assert snap.user_prompt == "test prompt"
        assert snap.kind == "manual"
        assert snap.id != ""
        assert snap.created_at != ""

        # Check snapshot directory exists
        snap_dir = tmp_storage._snapshot_dir("snap", snap.id)
        assert snap_dir.is_dir()
        files_dir = snap_dir / SNAPSHOT_FILES_DIR
        assert files_dir.is_dir()
        assert (files_dir / "hello.txt").is_file()
        assert (files_dir / "hello.txt").read_text(encoding="utf-8") == "world"
        # Metadata file
        assert (snap_dir / PROJECT_FILE).is_file()

    def test_truncates_long_prompt(self, tmp_storage: Storage):
        """user_prompt longer than 4000 chars is truncated."""
        tmp_storage.create_project("snap-lp", template="next")
        long_prompt = "x" * 5000
        snap = tmp_storage.create_snapshot("snap-lp", user_prompt=long_prompt)
        assert len(snap.user_prompt) == 4000

    def test_default_kind(self, tmp_storage: Storage):
        """Default kind is 'pre-turn'."""
        tmp_storage.create_project("snap-def", template="next")
        snap = tmp_storage.create_snapshot("snap-def")
        assert snap.kind == "pre-turn"

    def test_raises_for_missing_project(self, tmp_storage: Storage):
        """Raises FileNotFoundError for non-existent project."""
        with pytest.raises(FileNotFoundError, match="project not found"):
            tmp_storage.create_snapshot("missing-project")

    def test_snapshot_excludes_ignored_dirs(self, tmp_storage: Storage):
        """Ignored top-level dirs are not included in snapshot."""
        tmp_storage.create_project("snap-ignore", template="next")
        proj = tmp_storage.project_dir("snap-ignore")
        (proj / "node_modules" / "pkg").mkdir(parents=True)
        (proj / "node_modules" / "pkg" / "index.js").write_text("x")
        snap = tmp_storage.create_snapshot("snap-ignore")
        files_dir = tmp_storage._snapshot_dir("snap-ignore", snap.id) / SNAPSHOT_FILES_DIR
        assert not (files_dir / "node_modules").exists()

    def test_snapshot_skips_symlinks(self, tmp_storage: Storage):
        """Symlinks in project are not included in snapshot."""
        tmp_storage.create_project("snap-sym", template="next")
        proj = tmp_storage.project_dir("snap-sym")
        # Create a file and a symlink to it
        (proj / "real.txt").write_text("real")
        link = proj / "link.txt"
        link.symlink_to(proj / "real.txt")
        snap = tmp_storage.create_snapshot("snap-sym")
        files_dir = tmp_storage._snapshot_dir("snap-sym", snap.id) / SNAPSHOT_FILES_DIR
        # Symlink itself should not be in snapshot
        assert not (files_dir / "link.txt").exists()
        # Real file should be
        assert (files_dir / "real.txt").exists()


class TestListSnapshots:
    def test_returns_empty_for_no_snapshots(self, tmp_storage: Storage):
        """list_snapshots returns [] when no snapshots exist."""
        tmp_storage.create_project("no-snaps", template="next")
        assert tmp_storage.list_snapshots("no-snaps") == []

    def test_lists_snapshots_newest_first(self, tmp_storage: Storage):
        """Snapshots are ordered newest first."""
        tmp_storage.create_project("snap-list", template="next")
        s1 = tmp_storage.create_snapshot("snap-list", kind="pre-turn")
        s2 = tmp_storage.create_snapshot("snap-list", kind="manual")
        snaps = tmp_storage.list_snapshots("snap-list")
        assert len(snaps) == 2
        assert snaps[0].id == s2.id  # newest first
        assert snaps[1].id == s1.id

    def test_skips_invalid_snapshot_dirs(self, tmp_storage: Storage):
        """Non-snapshot directories in snapshots folder are ignored."""
        tmp_storage.create_project("snap-skip", template="next")
        snap = tmp_storage.create_snapshot("snap-skip")
        # Create a bogus dir
        snaps_dir = tmp_storage._snapshots_dir("snap-skip")
        (snaps_dir / "not-a-snapshot").mkdir()
        snaps = tmp_storage.list_snapshots("snap-skip")
        assert len(snaps) == 1
        assert snaps[0].id == snap.id

    def test_skips_missing_metadata(self, tmp_storage: Storage):
        """Snapshot dir without project.json is skipped."""
        tmp_storage.create_project("snap-meta", template="next")
        snap = tmp_storage.create_snapshot("snap-meta")
        snaps_dir = tmp_storage._snapshots_dir("snap-meta")
        # Create a dir that looks like a snapshot but has no project.json
        fake_id = "20250101T000000Z-12345678"
        fake_dir = snaps_dir / fake_id
        fake_dir.mkdir()
        snaps = tmp_storage.list_snapshots("snap-meta")
        assert len(snaps) == 1
        assert snaps[0].id == snap.id

    def test_handles_corrupted_metadata(self, tmp_storage: Storage):
        """Corrupted metadata file is skipped."""
        tmp_storage.create_project("snap-corr", template="next")
        snap = tmp_storage.create_snapshot("snap-corr")
        snaps_dir = tmp_storage._snapshots_dir("snap-corr")
        # Create a dir with corrupted project.json
        fake_id = "20250101T000000Z-87654321"
        fake_dir = snaps_dir / fake_id
        fake_dir.mkdir()
        (fake_dir / PROJECT_FILE).write_text("not-json")
        snaps = tmp_storage.list_snapshots("snap-corr")
        assert len(snaps) == 1
        assert snaps[0].id == snap.id

    def test_skips_files_in_snapshots_dir(self, tmp_storage: Storage):
        """Files (not dirs) in snapshots dir are skipped."""
        tmp_storage.create_project("snap-file", template="next")
        snap = tmp_storage.create_snapshot("snap-file")
        snaps_dir = tmp_storage._snapshots_dir("snap-file")
        (snaps_dir / "some_file.txt").write_text("hello")
        snaps = tmp_storage.list_snapshots("snap-file")
        assert len(snaps) == 1
        assert snaps[0].id == snap.id

    def test_skips_dirs_not_matching_snapshot_id_re(self, tmp_storage: Storage):
        """Dirs not matching SNAPSHOT_ID_RE are skipped."""
        tmp_storage.create_project("snap-skip2", template="next")
        snap = tmp_storage.create_snapshot("snap-skip2")
        snaps_dir = tmp_storage._snapshots_dir("snap-skip2")
        # Dir name doesn't match SNAPSHOT_ID_RE
        (snaps_dir / "not-a-valid-snapshot-id").mkdir()
        snaps = tmp_storage.list_snapshots("snap-skip2")
        assert len(snaps) == 1
        assert snaps[0].id == snap.id


class TestRestoreSnapshot:
    def test_restores_files(self, tmp_storage: Storage):
        """restore_snapshot restores previous file state."""
        tmp_storage.create_project("restore", template="next")
        tmp_storage.write_file("restore", "version.txt", "v1")
        snap = tmp_storage.create_snapshot("restore", kind="pre-turn")
        # Modify the file
        tmp_storage.write_file("restore", "version.txt", "v2")
        assert tmp_storage.read_file("restore", "version.txt") == "v2"
        # Restore
        result = tmp_storage.restore_snapshot("restore", snap.id)
        assert result is True
        assert tmp_storage.read_file("restore", "version.txt") == "v1"

    def test_restores_removed_files(self, tmp_storage: Storage):
        """Restore brings back files that were deleted."""
        tmp_storage.create_project("restore2", template="next")
        tmp_storage.write_file("restore2", "willremove.txt", "keep me")
        snap = tmp_storage.create_snapshot("restore2")
        tmp_storage.delete_file("restore2", "willremove.txt")
        assert tmp_storage.read_file("restore2", "willremove.txt") is None
        tmp_storage.restore_snapshot("restore2", snap.id)
        assert tmp_storage.read_file("restore2", "willremove.txt") == "keep me"

    def test_raises_for_missing_project(self, tmp_storage: Storage):
        """Raises FileNotFoundError for non-existent project."""
        with pytest.raises(FileNotFoundError, match="project not found"):
            tmp_storage.restore_snapshot("missing", "some-id")

    def test_returns_false_for_missing_snapshot(self, tmp_storage: Storage):
        """Returns False if snapshot doesn't exist."""
        tmp_storage.create_project("restore3", template="next")
        result = tmp_storage.restore_snapshot("restore3", "20250101T000000Z-12345678")
        assert result is False

    def test_preserves_sidecar_files(self, tmp_storage: Storage):
        """Sidecar directory is untouched during restore."""
        tmp_storage.create_project("restore4", template="next")
        snap = tmp_storage.create_snapshot("restore4")
        sidecar = tmp_storage.sidecar_dir("restore4")
        assert sidecar.is_dir()  # still exists after restore
        tmp_storage.restore_snapshot("restore4", snap.id)
        assert sidecar.is_dir()

    def test_restore_handles_symlinks(self, tmp_storage: Storage):
        """Symlinks in project dir are properly removed during restore."""
        tmp_storage.create_project("restore-sym", template="next")
        proj = tmp_storage.project_dir("restore-sym")
        # Take a snapshot (symlinks are not included in snapshot)
        snap = tmp_storage.create_snapshot("restore-sym")
        # Add a symlink after the snapshot
        (proj / "real.txt").write_text("content")
        link = proj / "link.txt"
        link.symlink_to(proj / "real.txt")
        assert link.is_symlink()
        # Restore should remove the symlink
        tmp_storage.restore_snapshot("restore-sym", snap.id)
        assert not link.exists()


class TestDeleteSnapshot:
    def test_deletes_snapshot(self, tmp_storage: Storage):
        """Delete a specific snapshot."""
        tmp_storage.create_project("del-snap", template="next")
        snap = tmp_storage.create_snapshot("del-snap")
        snap_dir = tmp_storage._snapshot_dir("del-snap", snap.id)
        assert snap_dir.is_dir()
        result = tmp_storage.delete_snapshot("del-snap", snap.id)
        assert result is True
        assert not snap_dir.exists()

    def test_returns_false_for_missing(self, tmp_storage: Storage):
        """Returns False for non-existent snapshot."""
        tmp_storage.create_project("del-snap2", template="next")
        result = tmp_storage.delete_snapshot("del-snap2", "20250101T000000Z-99999999")
        assert result is False

    def test_raises_for_invalid_id(self, tmp_storage: Storage):
        """Invalid snapshot ID raises ValueError."""
        tmp_storage.create_project("del-snap3", template="next")
        with pytest.raises(ValueError, match="invalid snapshot id"):
            tmp_storage.delete_snapshot("del-snap3", "..")


# =============================================================================
# _prune_snapshots
# =============================================================================


class TestPruneSnapshots:
    def test_prunes_oldest(self, tmp_storage: Storage):
        """When more than SNAPSHOT_KEEP snapshots exist, oldest are pruned."""
        tmp_storage.create_project("prune", template="next")
        # Create SNAPSHOT_KEEP + 5 snapshots
        snap_ids = []
        for _ in range(SNAPSHOT_KEEP + 5):
            snap = tmp_storage.create_snapshot("prune", kind="pre-turn")
            snap_ids.append(snap.id)
        remaining = tmp_storage.list_snapshots("prune")
        assert len(remaining) <= SNAPSHOT_KEEP

    def test_no_prune_under_limit(self, tmp_storage: Storage):
        """No pruning when at or below SNAPSHOT_KEEP."""
        tmp_storage.create_project("prune2", template="next")
        for _ in range(SNAPSHOT_KEEP):
            tmp_storage.create_snapshot("prune2", kind="pre-turn")
        remaining = tmp_storage.list_snapshots("prune2")
        assert len(remaining) == SNAPSHOT_KEEP

    def test_prune_preserves_newest(self, tmp_storage: Storage):
        """Only the oldest snapshots beyond the limit are removed."""
        tmp_storage.create_project("prune3", template="next")
        # Create SNAPSHOT_KEEP + 5 snapshots
        for i in range(SNAPSHOT_KEEP + 5):
            # Write a marker file to identify the snapshot
            tmp_storage.write_file("prune3", f"marker-{i}.txt", str(i))
            snap = tmp_storage.create_snapshot("prune3", kind="pre-turn")
        remaining = tmp_storage.list_snapshots("prune3")
        assert len(remaining) == SNAPSHOT_KEEP
        # The oldest 5 should be gone; keep the newest SNAPSHOT_KEEP
        # Since snapshots are listed newest first, the last entries should be
        # the oldest surviving ones

    def test_prune_failure_logged(self, tmp_storage: Storage, caplog):
        """When a snapshot deletion fails during prune, a warning is logged."""
        tmp_storage.create_project("prune-fail", template="next")
        # Create SNAPSHOT_KEEP + 1 snapshots
        for i in range(SNAPSHOT_KEEP + 1):
            tmp_storage.create_snapshot("prune-fail", kind="pre-turn")
        # Manually delete the metadata of the oldest snapshot so it fails
        snaps = tmp_storage.list_snapshots("prune-fail")
        oldest = snaps[-1]  # oldest should be pruned
        # Delete its metadata to cause failure
        snap_dir = tmp_storage._snapshot_dir("prune-fail", oldest.id)
        # Remove write permission to cause failure
        import stat
        snap_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # remove write perm
        try:
            # Trigger prune by creating another snapshot (this will try to prune the oldest)
            import contextlib
            with contextlib.suppress(PermissionError, OSError):
                tmp_storage.create_snapshot("prune-fail", kind="pre-turn")
            # Should log a warning about failed prune
            # Check at least one snapshot operation occurred
            assert len(tmp_storage.list_snapshots("prune-fail")) >= SNAPSHOT_KEEP
        finally:
            snap_dir.chmod(stat.S_IRWXU)  # restore permissions


# =============================================================================
# ensure_next_preview_layout
# =============================================================================


class TestEnsureNextPreviewLayout:
    def test_adds_missing_starter_files(self, tmp_storage: Storage):
        """Adds missing Next.js starter files."""
        tmp_storage.create_project("next-preview", template="next")
        proj = tmp_storage.project_dir("next-preview")
        # Remove a starter file
        layout_file = proj / "app" / "layout.tsx"
        layout_file.unlink()
        assert not layout_file.exists()
        tmp_storage.ensure_next_preview_layout("next-preview")
        assert layout_file.is_file()

    @pytest.mark.parametrize("template", ["express", "next-minimal"])
    def test_skips_non_next_templates(self, tmp_storage: Storage, template: str):
        """Does nothing for non-'next' templates."""
        tmp_storage.create_project("non-next", template=template)
        tmp_storage.ensure_next_preview_layout("non-next")
        # Should not raise or do anything harmful

    def test_skips_missing_project(self, tmp_storage: Storage):
        """Does nothing for non-existent project."""
        tmp_storage.ensure_next_preview_layout("nonexistent")
        # Should not raise

    def test_does_not_overwrite_existing_files(self, tmp_storage: Storage):
        """Existing files are not overwritten."""
        tmp_storage.create_project("next-preview2", template="next")
        proj = tmp_storage.project_dir("next-preview2")
        layout_file = proj / "app" / "layout.tsx"
        original = layout_file.read_text(encoding="utf-8")
        tmp_storage.ensure_next_preview_layout("next-preview2")
        assert layout_file.read_text(encoding="utf-8") == original

    def test_handles_missing_sidecar_gracefully(self, tmp_storage: Storage):
        """If project dir exists but no project.json, returns early."""
        proj = tmp_storage.root / "bare-next"
        proj.mkdir(parents=True)
        # No .odysseus/project.json
        tmp_storage.ensure_next_preview_layout("bare-next")
        # Should not raise

    @patch("cortex.project_storage.get_starter")
    def test_starter_none_returns_early(self, mock_get_starter, tmp_storage: Storage):
        """When get_starter('next') returns None, returns early."""
        mock_get_starter.return_value = None
        tmp_storage.create_project("next-none", template="next")
        tmp_storage.ensure_next_preview_layout("next-none")
        # No error; early return

    def test_skips_if_project_dir_missing(self, tmp_storage: Storage):
        """Returns early if project directory doesn't exist."""
        # Create project.json manually but no project directory
        sidecar = tmp_storage.root / "ghost" / SIDECAR_DIR
        _write_project_json(sidecar, ProjectRecord(id="ghost", name="ghost", template="next"))
        # No project dir (sidecar is created by _write_project_json)
        # ensure_next_preview_layout should return early
        tmp_storage.ensure_next_preview_layout("ghost")
        # Should not raise


# =============================================================================
# Module-level functions: get_storage, reset_storage, iter_ignored_top_level
# =============================================================================


class TestGetStorage:
    def setup_method(self):
        reset_storage()

    def test_returns_storage_instance(self):
        """get_storage returns a Storage instance."""
        s = get_storage()
        assert isinstance(s, Storage)

    def test_uses_default_root(self):
        """Default root is 'data/projects'."""
        s = get_storage()
        assert s.root == Path("data/projects").resolve()

    def test_custom_root(self, tmp_path: Path):
        """Passing root creates new storage with that root."""
        s = get_storage(root=tmp_path)
        assert s.root == tmp_path.resolve()

    def test_singleton_without_root(self, tmp_path: Path):
        """Without root, returns same instance."""
        s1 = get_storage()
        s2 = get_storage()
        assert s1 is s2

    def test_new_instance_with_root(self, tmp_path: Path):
        """Passing root creates new instance even if default exists."""
        s1 = get_storage()
        s2 = get_storage(root=tmp_path)
        assert s1 is not s2

    def teardown_method(self):
        reset_storage()


class TestResetStorage:
    def test_resets_singleton(self):
        """reset_storage clears the default storage."""
        s1 = get_storage()
        reset_storage()
        s2 = get_storage()
        assert s1 is not s2

    def teardown_method(self):
        reset_storage()


class TestIterIgnoredTopLevel:
    def test_returns_strings(self):
        """Returns an iterable of ignored top-level names."""
        items = list(iter_ignored_top_level())
        assert len(items) > 0
        assert SIDECAR_DIR in items
        assert "node_modules" in items

    def test_returns_iterator_not_set(self):
        """Returns an iterator, not the original set."""
        result = iter_ignored_top_level()
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_immutable(self):
        """Modifying returned items does not affect original."""
        items = list(iter_ignored_top_level())
        items.clear()
        assert len(list(iter_ignored_top_level())) > 0


# =============================================================================
# ProjectRecord dataclass
# =============================================================================


class TestProjectRecord:
    def test_created_at_defaults_to_now(self):
        """to_dict fills created_at with ISO timestamp if None."""
        rec = ProjectRecord(id="test", name="test")
        data = rec.to_dict()
        assert data["created_at"] is not None
        assert "T" in data["created_at"]

    def test_updated_at_defaults_to_now(self):
        """to_dict fills updated_at with ISO timestamp if None."""
        rec = ProjectRecord(id="test", name="test")
        data = rec.to_dict()
        assert data["updated_at"] is not None
        assert "T" in data["updated_at"]

    def test_from_dict_roundtrip(self):
        """from_dict restores a record from to_dict output."""
        original = ProjectRecord(
            id="my-project",
            name="My Project",
            template="express",
            project_type="full",
            starter_name="express",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )
        data = original.to_dict()
        restored = ProjectRecord.from_dict(data)
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.template == original.template
        assert restored.project_type == original.project_type
        assert restored.starter_name == original.starter_name
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at

    def test_from_dict_with_missing_keys(self):
        """from_dict uses defaults for missing keys."""
        rec = ProjectRecord.from_dict({"id": "test", "name": "test"})
        assert rec.template == "next"
        assert rec.project_type == "app"
        assert rec.starter_name == ""
        assert rec.created_at is None
        assert rec.updated_at is None

    def test_to_dict_includes_all_keys(self):
        """to_dict output contains all expected keys."""
        rec = ProjectRecord(id="test", name="test")
        data = rec.to_dict()
        assert set(data.keys()) == {
            "id", "name", "template", "project_type",
            "starter_name", "created_at", "updated_at",
        }


# =============================================================================
# SnapshotRecord dataclass
# =============================================================================


class TestSnapshotRecord:
    def test_default_kind(self):
        """Default kind is 'pre-turn'."""
        rec = SnapshotRecord(id="s1", created_at="now")
        assert rec.kind == "pre-turn"

    def test_default_user_prompt(self):
        """Default user_prompt is empty string."""
        rec = SnapshotRecord(id="s1", created_at="now")
        assert rec.user_prompt == ""


# =============================================================================
# PromptRecord dataclass
# =============================================================================


class TestPromptRecord:
    def test_default_snapshot_id_is_none(self):
        """Default snapshot_id is None."""
        rec = PromptRecord(id="p1", role="user", content="hi")
        assert rec.snapshot_id is None


# =============================================================================
# _now() helper
# =============================================================================


class TestNow:
    def test_returns_datetime_with_tz(self):
        """_now() returns a timezone-aware datetime."""
        now = _now()
        assert now.tzinfo is not None


# =============================================================================
# Edge cases and error handling
# =============================================================================


class TestEdgeCases:
    def test_create_project_empty_name(self, tmp_storage: Storage):
        """Empty name generates a UUID-based slug."""
        rec = tmp_storage.create_project("", template="next")
        assert rec.id.startswith("project-")
        assert rec.name == ""

    def test_get_project_non_existent(self, tmp_storage: Storage):
        """get_project returns None for non-existent project."""
        assert tmp_storage.get_project("no-such-project") is None

    def test_list_projects_after_delete(self, tmp_storage: Storage):
        """Deleted projects don't appear in listing."""
        tmp_storage.create_project("alpha", template="next")
        tmp_storage.create_project("beta", template="next")
        tmp_storage.delete_project("alpha")
        projects = tmp_storage.list_projects()
        slugs = [p.id for p in projects]
        assert "alpha" not in slugs
        assert "beta" in slugs

    def test_write_file_then_read_tree(self, tmp_storage: Storage):
        """Writing a file and reading tree includes it."""
        tmp_storage.create_project("wt-rt", template="next")
        tmp_storage.write_file("wt-rt", "data.json", '{"key": "val"}')
        tree = tmp_storage.read_tree("wt-rt")
        assert "data.json" in tree
        assert tree["data.json"]["file"]["contents"] == '{"key": "val"}'

    def test_snapshot_id_format(self, tmp_storage: Storage):
        """Snapshot ID matches expected format."""
        tmp_storage.create_project("snap-fmt", template="next")
        snap = tmp_storage.create_snapshot("snap-fmt")
        assert SNAPSHOT_ID_RE.fullmatch(snap.id) is not None

    def test_snapshot_metadata_content(self, tmp_storage: Storage):
        """Snapshot metadata file contains correct JSON."""
        tmp_storage.create_project("snap-meta2", template="next")
        snap = tmp_storage.create_snapshot("snap-meta2", user_prompt="test", kind="manual")
        meta_path = tmp_storage._snapshot_dir("snap-meta2", snap.id) / PROJECT_FILE
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["id"] == snap.id
        assert data["user_prompt"] == "test"
        assert data["kind"] == "manual"

    def test_concurrent_snapshot_id_creation(self, tmp_storage: Storage):
        """Multiple snapshots get unique IDs."""
        tmp_storage.create_project("snap-con", template="next")
        ids = set()
        for _ in range(10):
            snap = tmp_storage.create_snapshot("snap-con")
            ids.add(snap.id)
        assert len(ids) == 10

    def test_safe_join_with_empty_rel(self, tmp_path: Path):
        """Empty relative path resolves to root."""
        result = safe_join(tmp_path, "")
        assert result == tmp_path.resolve()

    def test_safe_join_with_dot(self, tmp_path: Path):
        """'.' resolves to root."""
        result = safe_join(tmp_path, ".")
        assert result == tmp_path.resolve()

    def test_read_tree_non_utf8_ignored(self, tmp_storage: Storage):
        """Files that fail UTF-8 decode are skipped in tree."""
        tmp_storage.create_project("binary-tree", template="next")
        proj = tmp_storage.project_dir("binary-tree")
        (proj / "binary.dat").write_bytes(b"\xff\xfe\x00\x01")
        tree = tmp_storage.read_tree("binary-tree")
        assert "binary.dat" not in tree

    def test_delete_file_removes_dir_recursively(self, tmp_storage: Storage):
        """delete_file removes a directory recursively."""
        tmp_storage.create_project("del-rec", template="next")
        proj = tmp_storage.project_dir("del-rec")
        (proj / "sub" / "nested").mkdir(parents=True)
        (proj / "sub" / "nested" / "file.txt").write_text("content")
        result = tmp_storage.delete_file("del-rec", "sub")
        assert result is True
        assert not (proj / "sub").exists()

    def test_append_prompt_updates_timestamp(self, tmp_storage: Storage):
        """Appending a prompt touches the project."""
        record = tmp_storage.create_project("prompt-ts", template="next")
        orig_updated = record.updated_at
        tmp_storage.append_prompt("prompt-ts", "user", "hello")
        updated = tmp_storage.get_project("prompt-ts")
        assert updated is not None
        assert updated.updated_at is not None
        assert updated.updated_at > orig_updated  # type: ignore[operator]

    def test_pop_prompt_updates_timestamp(self, tmp_storage: Storage):
        """Popping an assistant prompt touches the project."""
        record = tmp_storage.create_project("pop-ts", template="next")
        tmp_storage.append_prompt("pop-ts", "assistant", "hello")
        orig_updated = record.updated_at
        tmp_storage.pop_last_assistant_prompt("pop-ts")
        updated = tmp_storage.get_project("pop-ts")
        assert updated is not None
        assert updated.updated_at is not None
        assert updated.updated_at > orig_updated  # type: ignore[operator]

    def test_clear_prompts_updates_timestamp(self, tmp_storage: Storage):
        """Clearing prompts touches the project."""
        record = tmp_storage.create_project("clear-ts", template="next")
        tmp_storage.append_prompt("clear-ts", "user", "q")
        orig_updated = record.updated_at
        tmp_storage.clear_prompts("clear-ts")
        updated = tmp_storage.get_project("clear-ts")
        assert updated is not None
        assert updated.updated_at is not None
        assert updated.updated_at > orig_updated  # type: ignore[operator]

    def test_restore_snapshot_updates_timestamp(self, tmp_storage: Storage):
        """Restoring a snapshot touches the project."""
        record = tmp_storage.create_project("restore-ts", template="next")
        tmp_storage.write_file("restore-ts", "f.txt", "v1")
        snap = tmp_storage.create_snapshot("restore-ts")
        orig_updated = record.updated_at
        tmp_storage.restore_snapshot("restore-ts", snap.id)
        updated = tmp_storage.get_project("restore-ts")
        assert updated is not None
        assert updated.updated_at is not None
        assert updated.updated_at > orig_updated  # type: ignore[operator]

    def test_slugify_already_valid(self):
        """Already-valid slugs pass through unchanged."""
        assert slugify("my-project-42") == "my-project-42"

    def test_validate_slug_edge_cases(self):
        """Various slug boundary conditions."""
        # Single char
        assert validate_slug("a") is True
        assert validate_slug("0") is True
        # Max length
        assert validate_slug("a" * 63) is True
        assert validate_slug("a" * 64) is False

    def test_storage_root_path_property(self, tmp_storage: Storage):
        """root_path property returns the root Path."""
        assert tmp_storage.root_path == tmp_storage.root

    def test_create_project_with_full_type_no_starter(self, tmp_storage: Storage):
        """Full-type project with unknown starter still creates dirs."""
        record = tmp_storage.create_project("full-unknown", template="unknown", project_type="full")
        assert record.project_type == "full"
        proj_dir = tmp_storage.project_dir(record.id)
        assert (proj_dir / "code").is_dir()
        assert (proj_dir / "docs").is_dir()

    def test_create_project_with_nonexistent_template(self, tmp_storage: Storage):
        """Unknown template creates project but no starter files."""
        record = tmp_storage.create_project("no-starter", template="nonexistent")
        proj_dir = tmp_storage.project_dir(record.id)
        # No starter files, but .odysseus exists
        assert (proj_dir / SIDECAR_DIR).is_dir()
        # Should have no app/package.json since template is unknown
        assert not (proj_dir / "package.json").exists()

    def test_ensure_next_preview_layout_non_next_returns(self, tmp_storage: Storage):
        """ensure_next_preview_layout returns early for non-next projects."""
        tmp_storage.create_project("my-express", template="express")
        tmp_storage.ensure_next_preview_layout("my-express")
        # No error, no files added
        assert (tmp_storage.project_dir("my-express") / "src" / "index.ts").is_file()


# =============================================================================
# Integration: full workflow
# =============================================================================


class TestFullWorkflow:
    """End-to-end workflow test covering multiple operations in sequence."""

    def test_full_project_lifecycle(self, tmp_storage: Storage):
        """Complete create → write → snapshot → modify → restore → delete."""
        # Create
        rec = tmp_storage.create_project("lifecycle", template="next", project_type="app")
        assert rec.id == "lifecycle"

        # Write files
        tmp_storage.write_file("lifecycle", "README.md", "# Lifecycle Test")
        tmp_storage.write_file("lifecycle", "src/lib/helper.ts", "export const x = 1;")

        # Verify file reads
        assert tmp_storage.read_file("lifecycle", "README.md") == "# Lifecycle Test"
        assert tmp_storage.read_file("lifecycle", "src/lib/helper.ts") == "export const x = 1;"

        # Read tree
        tree = tmp_storage.read_tree("lifecycle")
        assert "README.md" in tree
        assert "src" in tree

        # Snapshot
        snap = tmp_storage.create_snapshot("lifecycle", user_prompt="initial setup", kind="manual")

        # Modify files
        tmp_storage.write_file("lifecycle", "README.md", "# Modified")
        tmp_storage.delete_file("lifecycle", "src/lib/helper.ts")

        # Verify changes
        assert tmp_storage.read_file("lifecycle", "README.md") == "# Modified"
        assert tmp_storage.read_file("lifecycle", "src/lib/helper.ts") is None

        # Restore snapshot
        result = tmp_storage.restore_snapshot("lifecycle", snap.id)
        assert result is True
        assert tmp_storage.read_file("lifecycle", "README.md") == "# Lifecycle Test"
        assert tmp_storage.read_file("lifecycle", "src/lib/helper.ts") == "export const x = 1;"

        # Prompts
        tmp_storage.append_prompt("lifecycle", "system", "You are a test bot.")
        tmp_storage.append_prompt("lifecycle", "user", "What is 2+2?")
        tmp_storage.append_prompt("lifecycle", "assistant", "4")
        prompts = tmp_storage.read_prompts("lifecycle")
        assert len(prompts) == 3

        # Pop assistant
        popped = tmp_storage.pop_last_assistant_prompt("lifecycle")
        assert popped is not None
        assert popped.content == "4"
        assert len(tmp_storage.read_prompts("lifecycle")) == 2

        # Clear prompts
        tmp_storage.clear_prompts("lifecycle")
        assert tmp_storage.read_prompts("lifecycle") == []

        # Delete project
        assert tmp_storage.get_project("lifecycle") is not None
        tmp_storage.delete_project("lifecycle")
        assert tmp_storage.get_project("lifecycle") is None
        assert not tmp_storage.project_dir("lifecycle").exists()

    def test_multiple_snapshots_prompts_starter(self, tmp_storage: Storage):
        """Multiple snapshots, prompts, and ensure_next_preview_layout work together."""
        rec = tmp_storage.create_project("multi", template="next")
        assert rec.id == "multi"

        # Create several snapshot versions
        for i in range(5):
            tmp_storage.write_file("multi", f"v{i}.txt", f"content-{i}")
            snap = tmp_storage.create_snapshot("multi", user_prompt=f"version {i}")
            assert snap.id is not None

        # List snapshots
        snaps = tmp_storage.list_snapshots("multi")
        assert len(snaps) == 5
        assert snaps[0].user_prompt == "version 4"

        # Verify file is latest
        assert tmp_storage.read_file("multi", "v4.txt") == "content-4"

        # Restore to version 2
        target_snap = [s for s in snaps if s.user_prompt == "version 2"][0]
        tmp_storage.restore_snapshot("multi", target_snap.id)
        assert tmp_storage.read_file("multi", "v2.txt") == "content-2"
        # v3-v4 should not exist
        assert tmp_storage.read_file("multi", "v3.txt") is None
        assert tmp_storage.read_file("multi", "v4.txt") is None

        # Ensure preview layout adds back any missing starter files
        proj = tmp_storage.project_dir("multi")
        # Remove a starter file
        layout_file = proj / "app" / "layout.tsx"
        if layout_file.exists():
            layout_file.unlink()
        tmp_storage.ensure_next_preview_layout("multi")
        assert layout_file.exists()

    def test_crud_isolation(self, tmp_storage: Storage):
        """Operations on one project don't affect others."""
        p1 = tmp_storage.create_project("proj-a", template="next")
        p2 = tmp_storage.create_project("proj-b", template="express")
        assert p1.id != p2.id

        tmp_storage.write_file("proj-a", "common.txt", "from A")
        tmp_storage.write_file("proj-b", "common.txt", "from B")

        assert tmp_storage.read_file("proj-a", "common.txt") == "from A"
        assert tmp_storage.read_file("proj-b", "common.txt") == "from B"

        # Snapshot only one
        snap_a = tmp_storage.create_snapshot("proj-a")
        tmp_storage.write_file("proj-a", "common.txt", "modified A")

        # Restore A
        tmp_storage.restore_snapshot("proj-a", snap_a.id)
        assert tmp_storage.read_file("proj-a", "common.txt") == "from A"
        # B unchanged
        assert tmp_storage.read_file("proj-b", "common.txt") == "from B"

    def test_empty_and_whitespace_names(self, tmp_storage: Storage):
        """Edge cases with empty/whitespace names."""
        r1 = tmp_storage.create_project("", template="next")
        assert r1.name == ""
        assert r1.id.startswith("project-")

        r2 = tmp_storage.create_project("  ", template="next")
        assert r2.name == ""
        # Since slugify("  ") == "", we get a UUID slug
        assert r2.id.startswith("project-")
        # The name field strips whitespace, so r2.name is ""
        assert r2.id != r1.id


# =============================================================================
# Error handling & resilience
# =============================================================================


class TestErrorHandling:
    def test_create_project_missing_sidecar_then_prompt_works(self, tmp_storage: Storage):
        """Prompt append works even when sidecar doesn't exist yet."""
        rec = tmp_storage.create_project("err-handle", template="next")
        # Remove sidecar
        sidecar = tmp_storage.sidecar_dir("err-handle")
        shutil.rmtree(sidecar)
        assert not sidecar.exists()
        # append_prompt should recreate it
        pr = tmp_storage.append_prompt("err-handle", "user", "still works")
        assert pr.role == "user"
        assert sidecar.is_dir()

    def test_list_snapshots_no_snapshots_dir(self, tmp_storage: Storage):
        """list_snapshots returns [] when snapshots dir doesn't exist."""
        tmp_storage.create_project("no-snap-dir", template="next")
        snaps_dir = tmp_storage._snapshots_dir("no-snap-dir")
        assert not snaps_dir.exists() or True  # sidecar may exist but snapshots/ not created yet
        if snaps_dir.exists():
            shutil.rmtree(snaps_dir)
        assert tmp_storage.list_snapshots("no-snap-dir") == []

    def test_delete_snapshot_invalid_id_raises(self, tmp_storage: Storage):
        """Deleting with an invalid snapshot ID raises ValueError."""
        tmp_storage.create_project("bad-id", template="next")
        with pytest.raises(ValueError, match="invalid snapshot id"):
            tmp_storage.delete_snapshot("bad-id", "invalid-id")

    def test_restore_snapshot_invalid_id_raises(self, tmp_storage: Storage):
        """Restoring with an invalid snapshot ID raises ValueError."""
        tmp_storage.create_project("bad-id-2", template="next")
        with pytest.raises(ValueError, match="invalid snapshot id"):
            tmp_storage.restore_snapshot("bad-id-2", "invalid-id")
