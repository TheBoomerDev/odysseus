"""
cortex/codegen/patcher.py — Apply file operations to project files.

Operations: create, replace, edit (search/replace), delete.
Path validation prevents traversal outside project root.
Auto-detects and adds "use client" directive for React client files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Literal, Optional, Protocol, Any

_MAX_FILE_CONTENT_CHARS = 80_000
_MAX_SEARCH_REPLACE_CHARS = 8_000
_MAX_EDITS_PER_FILE = 20

_FORBIDDEN_SEGMENTS: FrozenSet[str] = frozenset({".git", "node_modules", ".micracode"})

_CLIENT_DIRECTIVE_SUFFIXES = (".tsx", ".jsx", ".ts", ".js")
_SERVER_ONLY_APP_FILES = ("app/layout.tsx", "app/layout.jsx")

_CLIENT_ONLY_IMPORT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"""from\s+['"]framer-motion['"]"""),
    re.compile(r"""from\s+['"]@?react-spring(?:/[\\w-]+)?['"]"""),
)

_CLIENT_ONLY_HOOKS = (
    "useState", "useEffect", "useLayoutEffect", "useReducer",
    "useRef", "useContext", "useCallback", "useMemo",
    "useTransition", "useDeferredValue", "useSyncExternalStore",
    "useImperativeHandle",
)
_CLIENT_HOOK_RE = re.compile(
    r"\b(?:" + "|".join(_CLIENT_ONLY_HOOKS) + r")\s*\("
)

_USE_CLIENT_RE = re.compile(r"""^\s*['"]use client['"]\s*;?""")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Operation = Literal["create", "replace", "edit", "delete"]


@dataclass(frozen=True)
class SearchReplace:
    """A single search-and-replace operation within a file edit."""

    search: str
    replace: str


@dataclass(frozen=True)
class PatchFile:
    """A single file operation within a patch bundle."""

    path: str
    operation: Operation
    content: Optional[str] = None
    edits: Optional[List[SearchReplace]] = None


@dataclass(frozen=True)
class PatchBundle:
    """A bundle of file operations to apply atomically."""

    files: List[PatchFile]


@dataclass(frozen=True)
class PatchResult:
    """Result of applying a single PatchFile."""

    path: str
    kind: Literal["write", "delete", "error"]
    content: str = ""
    error: str = ""


class FileLoader(Protocol):
    """Protocol for loading file contents from a project."""

    def __call__(self, path: str) -> Optional[str]: ...


@dataclass
class ProjectContext:
    """Holds the context of a project during codegen."""

    project_id: str
    tree_summary: str
    files: Dict[str, str] = field(default_factory=dict)
    loader: Optional[FileLoader] = None
    placeholder_files: FrozenSet[str] = field(default_factory=frozenset)

    def get_file(self, path: str) -> Optional[str]:
        """Get a file's contents, loading from disk if needed."""
        if path in self.files:
            return self.files[path]
        if self.loader is None:
            return None
        content = self.loader(path)
        if content is not None:
            self.files[path] = content
        return content


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def _normalize_path(raw: str) -> Optional[str]:
    """Normalize a file path: strip leading /, normalize separators."""
    cleaned = raw.strip().replace("\\", "/").lstrip("/")
    return cleaned or None


def _path_is_safe(rel: str) -> bool:
    """Check that a relative path doesn't escape or touch forbidden dirs."""
    parts = Path(rel).parts
    if not parts:
        return False
    if any(seg in _FORBIDDEN_SEGMENTS for seg in parts):
        return False
    if ".." in parts:
        return False
    return True


# ---------------------------------------------------------------------------
# "use client" detection
# ---------------------------------------------------------------------------


def _needs_use_client(path: str, content: str) -> bool:
    """Check if a file needs a ``'use client'`` directive added."""
    if not path.endswith(_CLIENT_DIRECTIVE_SUFFIXES):
        return False
    if path.endswith(".d.ts"):
        return False
    if path in _SERVER_ONLY_APP_FILES:
        return False
    if _USE_CLIENT_RE.match(content):
        return False
    if any(rx.search(content) for rx in _CLIENT_ONLY_IMPORT_RES):
        return True
    if _CLIENT_HOOK_RE.search(content):
        return True
    return False


def _ensure_use_client(path: str, content: str) -> str:
    """Prepend ``'use client'`` directive if needed."""
    if not _needs_use_client(path, content):
        return content
    return '"use client";\n\n' + content.lstrip("\n")


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


class PatchError(Exception):
    """Raised when a search/replace op cannot be applied unambiguously."""


def apply_patch(original: str, ops: List[SearchReplace]) -> str:
    """Apply sequential search/replace operations to a string."""
    buffer = original
    for idx, op in enumerate(ops):
        count = buffer.count(op.search)
        if count == 0:
            raise PatchError(
                f"edit #{idx + 1}: search string not found in file"
            )
        if count > 1:
            raise PatchError(
                f"edit #{idx + 1}: search string matches {count} times, "
                f"expected 1"
            )
        buffer = buffer.replace(op.search, op.replace, 1)
    return buffer


def _truncate(content: str) -> str:
    """Truncate content to the maximum allowed size."""
    if len(content) > _MAX_FILE_CONTENT_CHARS:
        return content[:_MAX_FILE_CONTENT_CHARS]
    return content


def _apply_one(file: PatchFile, context: ProjectContext) -> PatchResult:
    """Apply a single PatchFile operation."""
    rel = _normalize_path(file.path)
    if rel is None:
        return PatchResult(path=file.path, kind="error", error="empty path")
    if not _path_is_safe(rel):
        return PatchResult(
            path=rel, kind="error", error=f"unsafe path: {rel}"
        )

    op = file.operation

    if op in ("create", "replace"):
        if file.content is None:
            return PatchResult(
                path=rel, kind="error", error=f"{op}: missing content"
            )
        final = _ensure_use_client(rel, file.content)
        return PatchResult(path=rel, kind="write", content=_truncate(final))

    if op == "edit":
        current = context.get_file(rel)
        if current is None:
            return PatchResult(
                path=rel,
                kind="error",
                error=f"edit: file not found on disk: {rel}",
            )
        try:
            patched = apply_patch(current, file.edits or [])
        except PatchError as exc:
            return PatchResult(
                path=rel, kind="error", error=f"edit: {exc}"
            )
        patched = _ensure_use_client(rel, patched)
        context.files[rel] = patched
        return PatchResult(path=rel, kind="write", content=_truncate(patched))

    if op == "delete":
        return PatchResult(path=rel, kind="delete")

    return PatchResult(
        path=rel, kind="error", error=f"unknown operation: {op}"
    )


def apply_bundle(bundle: PatchBundle, context: ProjectContext) -> List[PatchResult]:
    """Apply a bundle of file operations against a project context.

    Returns a list of PatchResult, one per file in the bundle.
    All operations are attempted even if some fail.
    """
    return [_apply_one(f, context) for f in bundle.files]


# ---------------------------------------------------------------------------
# Convenience: execute helpers for single file ops
# ---------------------------------------------------------------------------


def execute_write_file(
    path: str,
    content: str,
    project_root: Optional[Path] = None,
    storage: Any = None,
    project_id: Optional[str] = None,
) -> PatchResult:
    """Write a file directly (create or overwrite).

    If *storage* and *project_id* are provided, persists via storage.
    Otherwise writes directly to *project_root*.
    """
    rel = _normalize_path(path)
    if rel is None:
        return PatchResult(path=path, kind="error", error="empty path")
    if not _path_is_safe(rel):
        return PatchResult(path=rel, kind="error", error=f"unsafe path: {rel}")

    final = _ensure_use_client(rel, content)
    final = _truncate(final)

    try:
        if storage is not None and project_id is not None and project_root is not None:
            target = project_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(final, encoding="utf-8")
        elif project_root is not None:
            target = project_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(final, encoding="utf-8")
        else:
            return PatchResult(path=rel, kind="error", error="no project_root provided")
        return PatchResult(path=rel, kind="write", content=final)
    except (ValueError, OSError) as exc:
        return PatchResult(
            path=rel, kind="error", error=f"error writing file: {exc}"
        )


def execute_read_file(path: str, project_root: Path) -> str:
    """Read a file relative to project root; return content or error string."""
    rel = _normalize_path(path)
    if rel is None:
        return "error: empty path"
    if not _path_is_safe(rel):
        return f"error: path outside project root: {path!r}"
    try:
        target = project_root / rel
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"error: file not found: {path!r}"
    except OSError as exc:
        return f"error: {exc}"
