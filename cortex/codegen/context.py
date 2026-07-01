"""
cortex/codegen/context.py — Intelligent project context loading for codegen.

Selects the most relevant files from a project to include in the LLM prompt,
respecting a character budget (default 40KB). Prioritizes:
  1. Always-loaded files (layout.tsx, globals.css, etc.)
  2. Files mentioned in the user's prompt
  3. Recently modified files
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .patcher import ProjectContext
from .starter import get_starter, is_placeholder_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_CHAR_BUDGET = 40_000
ALWAYS_LOAD = ("app/page.tsx", "app/layout.tsx", "app/globals.css")
MAX_TREE_ENTRIES = 400
PLACEHOLDER_CANDIDATES = ("app/page.tsx", "app/layout.tsx", "app/globals.css")

# ---------------------------------------------------------------------------
# File tree
# ---------------------------------------------------------------------------


def _flatten(tree: Dict[str, Any], prefix: str = "") -> List[Tuple[str, int]]:
    """Flatten a nested file tree dict into a list of (path, size) tuples.

    Tree format:
        {
            "filename": {"file": {"contents": "..."}},
            "dirname": {"directory": {"filename": {"file": {"contents": "..."}}}},
        }
    """
    out: List[Tuple[str, int]] = []
    for name, node in tree.items():
        path = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
        if "directory" in node:
            out.extend(_flatten(node["directory"], path))
        elif "file" in node:
            contents = node["file"].get("contents", "")
            size = len(contents) if isinstance(contents, str) else 0
            out.append((path, size))
    return out


def _read_from_tree(tree: Dict[str, Any], path: str) -> Optional[str]:
    """Read a file's contents from a tree dict by path."""
    parts = path.split("/")
    node: Any = tree
    for i, part in enumerate(parts):
        if not isinstance(node, dict):
            return None
        is_last = i == len(parts) - 1
        if is_last:
            leaf = node.get(part)
            if isinstance(leaf, dict) and "file" in leaf:
                contents = leaf["file"].get("contents")
                return contents if isinstance(contents, str) else None
            return None
        nxt = node.get(part)
        if isinstance(nxt, dict) and "directory" in nxt:
            node = nxt["directory"]
        else:
            return None
    return None


def _mentioned_paths(prompt: str, candidates: List[str]) -> List[str]:
    """Find file paths that are mentioned in the user's prompt."""
    hits: List[str] = []
    for path in candidates:
        base = path.rsplit("/", 1)[-1]
        if path in prompt or (len(base) > 3 and base in prompt):
            hits.append(path)
    return hits


# ---------------------------------------------------------------------------
# File loader factory
# ---------------------------------------------------------------------------


def _build_loader(storage: Any, project_id: str) -> Callable[[str], Optional[str]]:
    """Build a file loader function from a storage backend.

    The storage must have a ``project_dir(slug)`` method returning a Path,
    and the loader reads files from that directory.
    """
    try:
        project_root = storage.project_dir(project_id)
    except Exception:
        project_root = None

    def _load(path: str) -> Optional[str]:
        if project_root is None:
            return None
        try:
            full = (project_root / path).resolve()
            # Prevent path traversal
            full.relative_to(project_root.resolve())
        except (ValueError, OSError):
            return None
        if not full.is_file():
            return None
        try:
            return full.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    return _load


# ---------------------------------------------------------------------------
# Context loader
# ---------------------------------------------------------------------------


def load_context(
    storage: Any,
    project_id: str,
    prompt: str,
    starter_name: Optional[str] = None,
) -> ProjectContext:
    """Load project context for a codegen turn.

    Builds a tree summary, selects the most relevant files within the
    context budget, and returns a ProjectContext.

    Args:
        storage: Storage backend (must have read_tree and project_dir)
        project_id: Project slug
        prompt: The user's prompt (used to find relevant files)
        starter_name: Optional starter name for placeholder detection

    Returns:
        ProjectContext with tree_summary, files, loader, placeholder_files
    """
    # Try to read tree from storage
    try:
        tree = storage.read_tree(project_id)
    except (FileNotFoundError, NotImplementedError, AttributeError):
        tree = {}

    # Flatten and sort
    flat = _flatten(tree)
    flat.sort(key=lambda row: row[0])

    # Build tree summary
    summary_lines = [f"{p} ({s})" for p, s in flat[:MAX_TREE_ENTRIES]]
    if len(flat) > MAX_TREE_ENTRIES:
        summary_lines.append(
            f"... ({len(flat) - MAX_TREE_ENTRIES} more files)"
        )
    tree_summary = "\n".join(summary_lines)

    # Select files to include
    candidate_paths = [p for p, _ in flat]

    # Priority 1: always-loaded files (if they exist)
    always = [p for p in ALWAYS_LOAD if p in candidate_paths]

    # Priority 2: files mentioned in prompt
    mentioned = _mentioned_paths(prompt, candidate_paths)

    # Deduplicated ordering: always first, then mentioned, preserving order
    wanted = list(dict.fromkeys(always + mentioned))

    # Load files within budget
    files: Dict[str, str] = {}
    budget = CONTEXT_CHAR_BUDGET
    for path in wanted:
        if budget <= 0:
            break
        content = _read_from_tree(tree, path)
        if content is None:
            continue
        if len(content) > budget:
            continue
        files[path] = content
        budget -= len(content)

    # Detect placeholder files (still holding starter scaffold content)
    placeholders: set[str] = set()
    if starter_name:
        starter = get_starter(starter_name)
        if starter:
            for path in PLACEHOLDER_CANDIDATES:
                if path in files and path in starter:
                    if files[path] == starter[path]:
                        placeholders.add(path)
    else:
        # Fallback: check against next starter if no starter specified
        from .starter import NEXT_STARTER
        for path in PLACEHOLDER_CANDIDATES:
            if path in files and path in NEXT_STARTER:
                if files[path] == NEXT_STARTER[path]:
                    placeholders.add(path)

    # Build file loader for on-demand reads during patching
    loader = _build_loader(storage, project_id)

    return ProjectContext(
        project_id=project_id,
        tree_summary=tree_summary,
        files=files,
        loader=loader,
        placeholder_files=frozenset(placeholders),
    )


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------


CONTEXT_FILE_DISPLAY_CAP = 12_000


def render_context_block(context: ProjectContext) -> str:
    """Render the project context as a string for the LLM prompt."""
    if not context.tree_summary and not context.files:
        return "Current project: (empty - this is the first turn)."

    parts: List[str] = []
    parts.append("Current project files (path (size in chars)):")
    parts.append(context.tree_summary or "(no files yet)")

    if context.placeholder_files:
        listed = ", ".join(sorted(context.placeholder_files))
        parts.append("")
        parts.append(
            "These files still hold unmodified starter-scaffold placeholder "
            "content and should be overwritten with `replace` (not `edit`) "
            f"when the user asks for any substantive change: {listed}."
        )

    if context.files:
        parts.append("")
        parts.append("Contents of the most relevant files:")
        for path, body in context.files.items():
            display = body
            if len(display) > CONTEXT_FILE_DISPLAY_CAP:
                display = (
                    display[:CONTEXT_FILE_DISPLAY_CAP]
                    + "\n/* ... truncated ... */"
                )
            marker = (
                " (placeholder scaffold)"
                if path in context.placeholder_files
                else ""
            )
            parts.append(f"\n----- {path}{marker} -----\n{display}")

    return "\n".join(parts)
