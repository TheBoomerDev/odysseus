"""Comprehensive pytest tests for cortex/git_automation.py.

Target >90% coverage of all public functions and internal helpers:
  - is_git_available, is_git_repo
  - init_git, commit, init_project_git
  - GITIGNORE_CONTENT, _default_commit_message
  - _run_git, _get_head_sha (implicitly via init_git)
Uses tmp_path for temp repos and monkeypatch for error-path injection.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import ANY, patch

import pytest

from cortex.git_automation import (
    GITIGNORE_CONTENT,
    _default_commit_message,
    _get_head_sha,
    _run_git,
    commit,
    init_git,
    init_project_git,
    is_git_available,
    is_git_repo,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    """Create and return a path to a fresh temp directory (not yet a git repo)."""
    d = tmp_path / "my_project"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def git_repo(repo_dir: Path) -> Path:
    """Return a path that has been initialised as a git repo."""
    result = init_git(repo_dir, project_name="test-project")
    assert result["success"] is True
    return repo_dir


# =============================================================================
# GITIGNORE_CONTENT
# =============================================================================


class TestGitignoreContent:
    def test_contains_node_modules(self):
        """GITIGNORE_CONTENT includes node_modules/ entry."""
        assert "node_modules/" in GITIGNORE_CONTENT

    def test_contains_env(self):
        """GITIGNORE_CONTENT includes .env entry."""
        assert ".env" in GITIGNORE_CONTENT

    def test_contains_odysseus(self):
        """GITIGNORE_CONTENT includes .odysseus/ entry."""
        assert ".odysseus/" in GITIGNORE_CONTENT

    def test_contains_pycache(self):
        """GITIGNORE_CONTENT includes __pycache__/ entry."""
        assert "__pycache__/" in GITIGNORE_CONTENT

    def test_contains_build_dirs(self):
        """GITIGNORE_CONTENT includes common build directories."""
        for entry in ("dist/", "build/", ".next/", ".cache/"):
            assert entry in GITIGNORE_CONTENT

    def test_contains_ide_and_os_files(self):
        """GITIGNORE_CONTENT includes IDE and OS file entries."""
        for entry in (".vscode/", ".DS_Store", "Thumbs.db"):
            assert entry in GITIGNORE_CONTENT

    def test_not_empty(self):
        """GITIGNORE_CONTENT is a non-empty string."""
        assert GITIGNORE_CONTENT
        assert len(GITIGNORE_CONTENT) > 50


# =============================================================================
# is_git_available
# =============================================================================


class TestIsGitAvailable:
    def test_returns_bool(self):
        """is_git_available returns a bool."""
        result = is_git_available()
        assert isinstance(result, bool)

    def test_returns_true_when_git_present(self):
        """On a system with git installed, returns True."""
        assert is_git_available() is True

    def test_false_on_file_not_found(self, monkeypatch):
        """Returns False when git executable is not found."""

        def _mock_run(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        assert is_git_available() is False

    def test_false_on_os_error(self, monkeypatch):
        """Returns False on OSError."""

        def _mock_run(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        assert is_git_available() is False

    def test_false_on_timeout(self, monkeypatch):
        """Returns False on subprocess.TimeoutExpired."""

        def _mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="git --version", timeout=5)

        monkeypatch.setattr(subprocess, "run", _mock_run)
        assert is_git_available() is False

    def test_false_on_nonzero_returncode(self, monkeypatch):
        """Returns False when git --version exits with non-zero."""

        def _mock_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=["git", "--version"], returncode=1, stdout="", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", _mock_run)
        assert is_git_available() is False

    def test_calls_git_version(self, monkeypatch):
        """Ensures is_git_available calls 'git --version'."""
        calls = []

        def _mock_run(*args, **kwargs):
            calls.append(args[0] if args else kwargs)
            return subprocess.CompletedProcess(
                args=["git", "--version"], returncode=0, stdout="git version 2.40.0", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", _mock_run)
        is_git_available()
        assert len(calls) == 1
        # check that the command included git --version
        cmd_args = calls[0]
        assert "git" in cmd_args or cmd_args[0] == "git"


# =============================================================================
# is_git_repo
# =============================================================================


class TestIsGitRepo:
    def test_false_on_empty_dir(self, tmp_path: Path):
        """A plain directory without .git is not a repo."""
        d = tmp_path / "empty"
        d.mkdir()
        assert is_git_repo(d) is False

    def test_false_on_missing_dir(self, tmp_path: Path):
        """A non-existent directory is not a repo."""
        assert is_git_repo(tmp_path / "nonexistent") is False

    def test_true_on_dot_git_dir(self, tmp_path: Path):
        """A directory with a .git subdirectory is a repo."""
        d = tmp_path / "repo"
        d.mkdir()
        (d / ".git").mkdir()
        assert is_git_repo(d) is True

    def test_true_on_dot_git_file(self, tmp_path: Path):
        """A directory with a .git file (submodule/worktree) is a repo."""
        d = tmp_path / "worktree"
        d.mkdir()
        (d / ".git").write_text("gitdir: ../other/.git", encoding="utf-8")
        assert is_git_repo(d) is True

    def test_after_init_git(self, repo_dir: Path):
        """After init_git, the directory is reported as a git repo."""
        init_git(repo_dir)
        assert is_git_repo(repo_dir) is True


# =============================================================================
# _default_commit_message
# =============================================================================


class TestDefaultCommitMessage:
    def test_format(self):
        """_default_commit_message produces a conventional commit message."""
        msg = _default_commit_message("my-project")
        assert msg == "feat: initial my-project project scaffold"

    def test_with_spaces(self):
        """Project name with spaces is preserved."""
        msg = _default_commit_message("Hello World")
        assert msg == "feat: initial Hello World project scaffold"

    def test_empty_name(self):
        """Empty project name produces a message with an empty placeholder."""
        msg = _default_commit_message("")
        assert msg == "feat: initial  project scaffold"


# =============================================================================
# init_git
# =============================================================================


class TestInitGitSuccess:
    def test_creates_dot_git(self, repo_dir: Path):
        """init_git creates a .git directory."""
        init_git(repo_dir, project_name="test-project")
        assert (repo_dir / ".git").exists()

    def test_creates_gitignore(self, repo_dir: Path):
        """init_git writes a .gitignore with standard content."""
        init_git(repo_dir, project_name="test-project")
        gitignore = repo_dir / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text(encoding="utf-8")
        assert "node_modules/" in content
        assert ".env" in content
        assert ".odysseus/" in content

    def test_does_not_overwrite_existing_gitignore(self, repo_dir: Path):
        """init_git preserves an existing .gitignore."""
        custom = "# custom\nnode_modules/\n"
        (repo_dir / ".gitignore").write_text(custom, encoding="utf-8")
        init_git(repo_dir, project_name="test-project")
        content = (repo_dir / ".gitignore").read_text(encoding="utf-8")
        assert content == custom

    def test_returns_success_with_sha(self, repo_dir: Path):
        """Successful init returns success=True with a non-empty commit_sha."""
        result = init_git(repo_dir, project_name="test-project")
        assert result["success"] is True
        assert "repo_dir" in result
        assert "commit_sha" in result
        assert result["commit_sha"] != ""
        assert result["note"] == "initialized and committed"

    def test_uses_branch_name(self, repo_dir: Path):
        """init_git uses the provided branch name."""
        result = init_git(repo_dir, project_name="test-project", branch="develop")
        assert result["success"] is True
        # Verify we're on that branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        assert branch_result.stdout.strip() == "develop"

    def test_default_branch_is_main(self, repo_dir: Path):
        """Default branch is main."""
        init_git(repo_dir, project_name="test-project")
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        assert branch_result.stdout.strip() == "main"

    def test_custom_commit_message(self, repo_dir: Path):
        """A custom commit message is used when provided."""
        result = init_git(
            repo_dir, project_name="test-project", commit_message="chore: bootstrap"
        )
        assert result["success"] is True
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "chore: bootstrap" in log.stdout

    def test_uses_directory_name_when_no_project_name(self, repo_dir: Path):
        """Uses directory name as project name fallback in commit message."""
        result = init_git(repo_dir)
        assert result["success"] is True
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "my_project" in log.stdout

    def test_repo_dir_in_result(self, repo_dir: Path):
        """The result contains the repo_dir as a string."""
        result = init_git(repo_dir, project_name="test-project")
        assert result["repo_dir"] == str(repo_dir)

    def test_commit_sha_is_valid(self, repo_dir: Path):
        """The commit_sha is a valid 40-char hex SHA."""
        result = init_git(repo_dir, project_name="test-project")
        sha = result["commit_sha"]
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


class TestInitGitAlreadyRepo:
    def test_returns_note(self, git_repo: Path):
        """Calling init_git on an existing repo returns note='already a git repository'."""
        result = init_git(git_repo, project_name="test-project")
        assert result["success"] is True
        assert result["note"] == "already a git repository"
        assert result["repo_dir"] == str(git_repo)

    def test_does_not_reinit(self, git_repo: Path):
        """Does not create a second commit on an already-initialised repo."""
        log_before = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        init_git(git_repo, project_name="test-project")
        log_after = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert log_before.stdout.strip() == log_after.stdout.strip()


class TestInitGitMissingDir:
    def test_returns_error(self, tmp_path: Path):
        """init_git on a non-existent directory returns success=False."""
        missing = tmp_path / "does_not_exist"
        result = init_git(missing, project_name="test-project")
        assert result["success"] is False
        assert "does not exist" in result.get("error", "")


class TestInitGitGitNotAvailable:
    def test_returns_error_when_git_missing(self, repo_dir: Path, monkeypatch):
        """Returns success=False when git is not installed."""

        def _mock_available(*args, **kwargs):
            return False

        monkeypatch.setattr("cortex.git_automation.is_git_available", _mock_available)
        result = init_git(repo_dir, project_name="test-project")
        assert result["success"] is False
        assert "git is not installed" in result.get("error", "")


class TestInitGitCalledProcessError:
    def test_caught_on_git_failure(self, repo_dir: Path, monkeypatch):
        """Catches CalledProcessError from _run_git and returns error dict."""

        def _mock_run_git(args, cwd, check=True):
            raise subprocess.CalledProcessError(
                returncode=128, cmd=["git"] + args, output="", stderr="fatal: some error"
            )

        monkeypatch.setattr("cortex.git_automation._run_git", _mock_run_git)
        result = init_git(repo_dir, project_name="test-project")
        assert result["success"] is False
        # str(CalledProcessError) includes the command and return code
        assert "returned non-zero exit status" in result.get("error", "")


class TestInitGitOSError:
    def test_caught_on_os_error(self, repo_dir: Path, monkeypatch):
        """Catches OSError from _run_git and returns error dict."""

        original_run_git = _run_git
        call_count = 0

        def _mock_run_git(args, cwd, check=True):
            nonlocal call_count
            call_count += 1
            # Let git init and checkout succeed, but fail on add
            if call_count <= 2:
                return original_run_git(args, cwd, check=check)
            raise OSError("disk full")

        monkeypatch.setattr("cortex.git_automation._run_git", _mock_run_git)
        result = init_git(repo_dir, project_name="test-project")
        assert result["success"] is False
        assert "disk full" in result.get("error", "")


class TestInitGitEmptyCommitFallback:
    def test_commit_issues_note(self, repo_dir: Path, monkeypatch):
        """When commit returns non-zero, result has note about commit issues."""

        # Create an empty repo by making git init succeed but no files to commit
        # We can force an empty commit scenario by making add succeed but commit
        # detect nothing to commit. Let's manipulate the git index.
        def _mock_run_git(args, cwd, check=True):
            if args[0] == "add":
                # Don't actually add anything — this creates an empty index
                return subprocess.CompletedProcess(
                    args=["git"] + args, returncode=0, stdout="", stderr=""
                )
            if args[0] == "commit":
                # Simulate "nothing to commit"
                return subprocess.CompletedProcess(
                    args=["git"] + args,
                    returncode=1,
                    stdout="",
                    stderr="nothing to commit",
                )
            # Let init/checkout go to real git
            return _run_git(args, cwd, check=check)

        monkeypatch.setattr("cortex.git_automation._run_git", _mock_run_git)
        result = init_git(repo_dir, project_name="test-project")
        assert result["success"] is True
        assert "commit had issues" in result.get("note", "")
        assert result["commit_sha"] == ""


# =============================================================================
# commit
# =============================================================================


class TestCommitSuccess:
    def test_with_add_all(self, git_repo: Path):
        """commit with add_all=True stages and commits changes."""
        # Make a change
        (git_repo / "README.md").write_text("# Hello", encoding="utf-8")
        result = commit(git_repo, "feat: add readme", add_all=True)
        assert result["success"] is True
        assert result["commit_sha"] != ""
        assert len(result["commit_sha"]) == 40

    def test_commit_sha_is_valid(self, git_repo: Path):
        """commit returns a valid SHA."""
        (git_repo / "README.md").write_text("# Hi", encoding="utf-8")
        result = commit(git_repo, "feat: add readme", add_all=True)
        sha = result["commit_sha"]
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_multiple_commits(self, git_repo: Path):
        """Multiple commits work correctly."""
        (git_repo / "a.txt").write_text("a", encoding="utf-8")
        r1 = commit(git_repo, "first", add_all=True)
        assert r1["success"] is True
        (git_repo / "b.txt").write_text("b", encoding="utf-8")
        r2 = commit(git_repo, "second", add_all=True)
        assert r2["success"] is True
        assert r1["commit_sha"] != r2["commit_sha"]

    def test_without_add_all_stages_manually(self, git_repo: Path):
        """commit with add_all=False works if changes were manually staged."""
        (git_repo / "manual.txt").write_text("staged", encoding="utf-8")
        subprocess.run(
            ["git", "add", "manual.txt"],
            cwd=git_repo,
            capture_output=True,
            check=True,
        )
        result = commit(git_repo, "feat: manual stage", add_all=False)
        assert result["success"] is True
        assert result["commit_sha"] != ""


class TestCommitNotARepo:
    def test_not_a_git_repo(self, tmp_path: Path):
        """commit on a non-repo directory returns error."""
        d = tmp_path / "not_a_repo"
        d.mkdir()
        result = commit(d, "msg")
        assert result["success"] is False
        assert "not a git repository" in result.get("error", "")


class TestCommitGitNotAvailable:
    def test_git_not_available(self, git_repo: Path, monkeypatch):
        """commit returns error when git is not available."""

        def _mock_available(*args, **kwargs):
            return False

        monkeypatch.setattr("cortex.git_automation.is_git_available", _mock_available)
        result = commit(git_repo, "msg")
        assert result["success"] is False
        assert "git is not installed" in result.get("error", "")


class TestCommitNoStagedChanges:
    def test_without_add_all_no_staged(self, git_repo: Path):
        """commit with add_all=False and no staged changes returns error."""
        result = commit(git_repo, "no changes", add_all=False)
        assert result["success"] is False
        # The error should mention nothing to commit or similar
        assert result.get("error", "")


class TestCommitCalledProcessError:
    def test_caught_on_error(self, git_repo: Path, monkeypatch):
        """Catches CalledProcessError from _run_git."""

        def _mock_run_git(args, cwd, check=True):
            raise subprocess.CalledProcessError(
                returncode=128, cmd=["git"] + args, output="", stderr="fatal: bad"
            )

        monkeypatch.setattr("cortex.git_automation._run_git", _mock_run_git)
        result = commit(git_repo, "msg")
        assert result["success"] is False
        # str(CalledProcessError) includes the command and return code
        assert "returned non-zero exit status" in result.get("error", "")


class TestCommitOSError:
    def test_caught_on_os_error(self, git_repo: Path, monkeypatch):
        """Catches OSError from _run_git."""

        def _mock_run_git(args, cwd, check=True):
            raise OSError("permission denied")

        monkeypatch.setattr("cortex.git_automation._run_git", _mock_run_git)
        result = commit(git_repo, "msg")
        assert result["success"] is False
        assert "permission denied" in result.get("error", "")


# =============================================================================
# init_project_git
# =============================================================================


class TestInitProjectGitAppProject:
    def test_app_project_single_repo(self, tmp_path: Path):
        """App project type creates a single git repo at project root."""
        project_dir = tmp_path / "my-app"
        project_dir.mkdir(parents=True)
        # Write project.json with app type
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        project_json = odysseus / "project.json"
        project_json.write_text(
            json.dumps({"name": "My App", "project_type": "app"}), encoding="utf-8"
        )

        results = init_project_git(tmp_path, "my-app")
        assert len(results) == 1
        result = results[0]
        assert result["success"] is True
        assert result["repo_dir"] == str(project_dir)

    def test_app_uses_project_name_from_json(self, tmp_path: Path):
        """App project uses name from project.json for commit message."""
        project_dir = tmp_path / "my-app"
        project_dir.mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"name": "My App", "project_type": "app"}), encoding="utf-8"
        )

        results = init_project_git(tmp_path, "my-app")
        assert len(results) == 1
        assert results[0]["success"] is True
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        # The commit message uses the project name from JSON
        assert "My App" in log.stdout

    def test_default_project_type_is_app(self, tmp_path: Path):
        """When project.json is missing, default project type is app."""
        project_dir = tmp_path / "default-app"
        project_dir.mkdir(parents=True)
        # No .odysseus/project.json
        results = init_project_git(tmp_path, "default-app")
        assert len(results) == 1
        assert results[0]["success"] is True


class TestInitProjectGitFullProject:
    def test_full_project_creates_multiple_repos(self, tmp_path: Path):
        """Full project creates repos for code/, docs/, and root."""
        project_dir = tmp_path / "my-full"
        project_dir.mkdir(parents=True)
        (project_dir / "code").mkdir(parents=True)
        (project_dir / "docs").mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"name": "My Full", "project_type": "full"}), encoding="utf-8"
        )

        results = init_project_git(tmp_path, "my-full")
        assert len(results) == 3

        # Check all three succeeded
        for result in results:
            assert result["success"] is True

        # Check each repo dir
        result_dirs = {r["repo_dir"] for r in results}
        assert str(project_dir / "code") in result_dirs
        assert str(project_dir / "docs") in result_dirs
        assert str(project_dir) in result_dirs

    def test_full_project_skips_missing_code_dir(self, tmp_path: Path):
        """Full project skips code/ if it doesn't exist."""
        project_dir = tmp_path / "my-full"
        project_dir.mkdir(parents=True)
        # code/ does NOT exist, only docs/
        (project_dir / "docs").mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"name": "My Full", "project_type": "full"}), encoding="utf-8"
        )

        results = init_project_git(tmp_path, "my-full")
        # Should have docs + root = 2 repos (code was skipped)
        assert len(results) == 2
        result_dirs = {r["repo_dir"] for r in results}
        assert str(project_dir / "docs") in result_dirs
        assert str(project_dir) in result_dirs

    def test_full_project_skips_missing_docs_dir(self, tmp_path: Path):
        """Full project skips docs/ if it doesn't exist."""
        project_dir = tmp_path / "my-full"
        project_dir.mkdir(parents=True)
        (project_dir / "code").mkdir(parents=True)
        # docs/ does NOT exist
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"name": "My Full", "project_type": "full"}), encoding="utf-8"
        )

        results = init_project_git(tmp_path, "my-full")
        assert len(results) == 2
        result_dirs = {r["repo_dir"] for r in results}
        assert str(project_dir / "code") in result_dirs
        assert str(project_dir) in result_dirs

    def test_full_project_code_repo_name(self, tmp_path: Path):
        """Full project code repo uses '(code)' suffix in commit message."""
        project_dir = tmp_path / "my-full"
        project_dir.mkdir(parents=True)
        (project_dir / "code").mkdir(parents=True)
        (project_dir / "docs").mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"name": "MyApp", "project_type": "full"}), encoding="utf-8"
        )

        init_project_git(tmp_path, "my-full")
        # Check the code repo commit message includes the suffix
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=project_dir / "code",
            capture_output=True,
            text=True,
        )
        assert "MyApp (code)" in log.stdout

    def test_full_project_docs_repo_name(self, tmp_path: Path):
        """Full project docs repo uses '(docs)' suffix in commit message."""
        project_dir = tmp_path / "my-full"
        project_dir.mkdir(parents=True)
        (project_dir / "code").mkdir(parents=True)
        (project_dir / "docs").mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"name": "MyApp", "project_type": "full"}), encoding="utf-8"
        )

        init_project_git(tmp_path, "my-full")
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=project_dir / "docs",
            capture_output=True,
            text=True,
        )
        assert "MyApp (docs)" in log.stdout

    def test_full_project_meta_repo_name(self, tmp_path: Path):
        """Full project root/meta repo uses '(meta)' suffix in commit message."""
        project_dir = tmp_path / "my-full"
        project_dir.mkdir(parents=True)
        (project_dir / "code").mkdir(parents=True)
        (project_dir / "docs").mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"name": "MyApp", "project_type": "full"}), encoding="utf-8"
        )

        init_project_git(tmp_path, "my-full")
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        assert "MyApp (meta)" in log.stdout


class TestInitProjectGitMissingDir:
    def test_missing_project_dir(self, tmp_path: Path):
        """init_project_git returns error when project directory doesn't exist."""
        results = init_project_git(tmp_path, "nonexistent")
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Project directory not found" in results[0].get("error", "")


class TestInitProjectGitInvalidJson:
    def test_invalid_json_does_not_crash(self, tmp_path: Path):
        """If project.json is malformed, init_project_git still succeeds with defaults."""
        project_dir = tmp_path / "broken"
        project_dir.mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text("not valid json", encoding="utf-8")

        results = init_project_git(tmp_path, "broken")
        assert len(results) == 1
        assert results[0]["success"] is True


class TestInitProjectGitFallbackSlugAsName:
    def test_uses_slug_when_json_missing_name(self, tmp_path: Path):
        """When project.json exists but has no 'name' field, uses slug as name."""
        project_dir = tmp_path / "slug-name"
        project_dir.mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"project_type": "app"}), encoding="utf-8"
        )

        results = init_project_git(tmp_path, "slug-name")
        assert len(results) == 1
        assert results[0]["success"] is True
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        assert "slug-name" in log.stdout


class TestInitProjectGitReadError:
    def test_oserror_reading_json_does_not_crash(self, tmp_path: Path, monkeypatch):
        """OSError while reading project.json is silently caught."""
        project_dir = tmp_path / "safe"
        project_dir.mkdir(parents=True)
        odysseus = project_dir / ".odysseus"
        odysseus.mkdir(parents=True)
        (odysseus / "project.json").write_text(
            json.dumps({"name": "Safe", "project_type": "full"}), encoding="utf-8"
        )
        (project_dir / "code").mkdir(parents=True)

        # Make read_text raise OSError by making it unreadable
        import os

        os.chmod(odysseus / "project.json", 0o000)

        try:
            results = init_project_git(tmp_path, "safe")
            assert len(results) >= 1
            # Should not crash; at least the root repo may still work
            assert any(r.get("success") is True or "error" in r for r in results)
        finally:
            os.chmod(odysseus / "project.json", 0o644)


# =============================================================================
# _get_head_sha
# =============================================================================


class TestGetHeadSha:
    def test_returns_sha_in_repo(self, git_repo: Path):
        """_get_head_sha returns the current HEAD SHA."""
        sha = _get_head_sha(git_repo)
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_returns_empty_on_non_repo(self, tmp_path: Path):
        """_get_head_sha returns empty string in non-repo directory."""
        d = tmp_path / "not_repo"
        d.mkdir()
        sha = _get_head_sha(d)
        assert sha == ""

    def test_returns_empty_on_missing_dir(self, tmp_path: Path):
        """_get_head_sha returns empty string when directory doesn't exist."""
        sha = _get_head_sha(tmp_path / "missing")
        assert sha == ""

    def test_returns_empty_after_error(self, tmp_path: Path, monkeypatch):
        """_get_head_sha returns empty when subprocess errors."""

        def _mock_run_git(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=128, cmd=["git", "rev-parse", "HEAD"], output="", stderr="fatal"
            )

        monkeypatch.setattr("cortex.git_automation._run_git", _mock_run_git)
        sha = _get_head_sha(tmp_path)
        assert sha == ""


# =============================================================================
# _run_git
# =============================================================================


class TestRunGit:
    def test_runs_successfully(self, git_repo: Path):
        """_run_git runs a command and returns a CompletedProcess."""
        result = _run_git(["rev-parse", "HEAD"], git_repo)
        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 0
        assert len(result.stdout.strip()) == 40

    def test_raises_on_non_zero_with_check(self, git_repo: Path):
        """_run_git raises CalledProcessError when check=True and non-zero exit."""
        with pytest.raises(subprocess.CalledProcessError):
            _run_git(["rev-parse", "INVALID_REF"], git_repo, check=True)

    def test_no_raise_with_check_false(self, git_repo: Path):
        """_run_git does NOT raise when check=False and non-zero exit."""
        result = _run_git(["rev-parse", "INVALID_REF"], git_repo, check=False)
        assert result.returncode != 0


# =============================================================================
# Logging (smoke test)
# =============================================================================


class TestLogging:
    def test_init_git_logs_on_error(self, repo_dir: Path, caplog, monkeypatch):
        """init_git logs an error when git operations fail."""
        caplog.set_level(logging.ERROR)

        def _mock_run_git(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=128, cmd=["git"] + args[0], output="", stderr="fatal: boom"
            )

        monkeypatch.setattr("cortex.git_automation._run_git", _mock_run_git)
        init_git(repo_dir, project_name="test")
        # The logger should have captured an error message
        assert any("Git init failed" in rec.message for rec in caplog.records)


# =============================================================================
# Integration / edge cases
# =============================================================================


class TestEdgeCases:
    def test_init_git_with_dot_in_directory_name(self, tmp_path: Path):
        """Directory names with dots are handled correctly."""
        d = tmp_path / "my.app.v2"
        d.mkdir()
        result = init_git(d, project_name="my.app.v2")
        assert result["success"] is True
        assert (d / ".git").exists()

    def test_gitignore_content_written_correctly(self, repo_dir: Path):
        """The .gitignore file written matches GITIGNORE_CONTENT exactly."""
        init_git(repo_dir, project_name="test")
        content = (repo_dir / ".gitignore").read_text(encoding="utf-8")
        assert content == GITIGNORE_CONTENT

    def test_commit_file_content_preserved(self, git_repo: Path):
        """Files committed can be read back."""
        (git_repo / "hello.md").write_text("# World", encoding="utf-8")
        commit(git_repo, "feat: hello", add_all=True)
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        content = subprocess.run(
            ["git", "show", f"{sha}:hello.md"],
            cwd=git_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert content.strip() == "# World"

    def test_add_all_stages_modified(self, git_repo: Path):
        """add_all=True stages modified files for commit."""
        (git_repo / "initial.txt").write_text("v1", encoding="utf-8")
        commit(git_repo, "first", add_all=True)
        (git_repo / "initial.txt").write_text("v2", encoding="utf-8")
        result = commit(git_repo, "second", add_all=True)
        assert result["success"] is True

    def test_init_git_then_commit(self, repo_dir: Path):
        """Full cycle: init, make change, commit."""
        r1 = init_git(repo_dir, project_name="test")
        assert r1["success"] is True
        (repo_dir / "new.txt").write_text("data", encoding="utf-8")
        r2 = commit(repo_dir, "feat: new file", add_all=True)
        assert r2["success"] is True
        assert r1["commit_sha"] != r2["commit_sha"]

    def test_multiple_branches(self, repo_dir: Path):
        """Can init on a non-main branch."""
        r1 = init_git(repo_dir, project_name="test", branch="dev")
        assert r1["success"] is True
        assert r1["note"] == "initialized and committed"
        # Switch to main and init again (already a repo)
        r2 = init_git(repo_dir, project_name="test", branch="main")
        assert r2["note"] == "already a git repository"
