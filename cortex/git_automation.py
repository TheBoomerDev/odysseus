"""
cortex/git_automation.py — Git initialization and commit for projects.

Provides automatic git repo initialization with .gitignore, conventional
commit messages, and support for both app and full project layouts.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


GITIGNORE_CONTENT = """# Dependencies
node_modules/
.pnp
.pnp.js

# Build
.next/
dist/
build/
.cache/
.turbo/

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.venv/
venv/

# Logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Coverage
coverage/
.coverage
htmlcov/

# Project metadata
.odysseus/
"""


def is_git_available() -> bool:
    """Check if git is installed and available."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def is_git_repo(directory: Path) -> bool:
    """Check if a directory is already a git repository."""
    return (directory / ".git").exists() or (directory / ".git").is_file()


def init_git(
    directory: Path,
    project_name: str = "",
    commit_message: Optional[str] = None,
    branch: str = "main",
) -> Dict[str, Any]:
    """Initialize a git repository in a directory.

    Args:
        directory: Project directory to initialize
        project_name: Display name for the commit message
        commit_message: Custom commit message (default: conventional)
        branch: Branch name (default: main)

    Returns:
        Dict with keys: success, repo_dir, commit_sha, error, note
    """
    if not directory.exists():
        return {
            "success": False,
            "error": f"Directory does not exist: {directory}",
        }

    if not is_git_available():
        return {"success": False, "error": "git is not installed"}

    if is_git_repo(directory):
        return {
            "success": True,
            "note": "already a git repository",
            "repo_dir": str(directory),
        }

    try:
        # Initialize
        _run_git(["init"], directory)
        _run_git(["checkout", "-b", branch], directory)

        # Write .gitignore if not present
        gitignore = directory / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(GITIGNORE_CONTENT, encoding="utf-8")

        # Stage and commit
        _run_git(["add", "-A"], directory)

        msg = commit_message or _default_commit_message(project_name or directory.name)
        result = _run_git(["commit", "-m", msg], directory, check=False)

        if result.returncode == 0:
            sha = _get_head_sha(directory)
            return {
                "success": True,
                "repo_dir": str(directory),
                "commit_sha": sha,
                "note": "initialized and committed",
            }
        else:
            return {
                "success": True,
                "repo_dir": str(directory),
                "commit_sha": "",
                "note": f"initialized but commit had issues: {result.stderr.strip() or 'empty commit?'}",
            }

    except subprocess.CalledProcessError as exc:
        logger.error("Git init failed for %s: %s", directory, exc)
        return {"success": False, "error": str(exc)}
    except OSError as exc:
        logger.error("Git init failed for %s: %s", directory, exc)
        return {"success": False, "error": str(exc)}


def _run_git(
    args: List[str],
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a git command."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
        timeout=30,
    )


def _get_head_sha(directory: Path) -> str:
    """Get the current HEAD commit SHA."""
    try:
        result = _run_git(["rev-parse", "HEAD"], directory)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def _default_commit_message(project_name: str) -> str:
    """Generate a conventional commit message for initial scaffold."""
    return f"feat: initial {project_name} project scaffold"


def init_project_git(
    storage_path: Path,
    project_slug: str,
) -> List[Dict[str, Any]]:
    """Initialize git repos for all subdirectories of a project.

    For full projects, initializes separate repos for code/, docs/, and root.
    For app projects, initializes a single repo at the project root.

    Args:
        storage_path: Root directory containing project folders
        project_slug: Project slug/directory name

    Returns:
        List of result dicts from each git init call
    """
    results: List[Dict[str, Any]] = []
    project_dir = storage_path / project_slug

    if not project_dir.exists():
        return [{"success": False, "error": f"Project directory not found: {project_dir}"}]

    # Read project metadata
    project_json = project_dir / ".odysseus" / "project.json"
    is_full = False
    project_name = project_slug

    if project_json.exists():
        try:
            data = json.loads(project_json.read_text(encoding="utf-8"))
            project_name = data.get("name", project_slug)
            is_full = data.get("project_type") == "full"
        except (json.JSONDecodeError, OSError):
            pass

    if is_full:
        # Init code/ repo
        code_dir = project_dir / "code"
        if code_dir.exists():
            results.append(init_git(code_dir, f"{project_name} (code)"))

        # Init docs/ repo
        docs_dir = project_dir / "docs"
        if docs_dir.exists():
            results.append(init_git(docs_dir, f"{project_name} (docs)"))

        # Also init root with metadata
        results.append(init_git(project_dir, f"{project_name} (meta)"))
    else:
        # Single repo at project root
        results.append(init_git(project_dir, project_name))

    return results


def commit(
    directory: Path,
    message: str,
    *,
    add_all: bool = True,
) -> Dict[str, Any]:
    """Make a git commit in an existing repo.

    Args:
        directory: Git repository directory
        message: Commit message
        add_all: Stage all changes first

    Returns:
        Dict with success, commit_sha, error
    """
    if not is_git_available():
        return {"success": False, "error": "git is not installed"}

    if not is_git_repo(directory):
        return {"success": False, "error": "not a git repository"}

    try:
        if add_all:
            _run_git(["add", "-A"], directory)

        result = _run_git(["commit", "-m", message], directory, check=False)

        if result.returncode == 0:
            sha = _get_head_sha(directory)
            return {"success": True, "commit_sha": sha}
        else:
            return {
                "success": False,
                "error": result.stderr.strip() or "commit failed",
            }

    except subprocess.CalledProcessError as exc:
        return {"success": False, "error": str(exc)}
    except OSError as exc:
        return {"success": False, "error": str(exc)}
