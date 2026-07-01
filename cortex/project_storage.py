"""
cortex/project_storage.py — Project storage with snapshots, prompts, and scaffolding.

Provides:
  - Project CRUD (create, list, get, delete)
  - File operations (write, read tree, read content, delete)
  - Snapshots with full restore
  - Prompt history (JSONL)
  - Safe path utilities (no traversal)
  - Starter scaffolding (Next.js, Express, etc.)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from .codegen.starter import get_starter, list_starters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
SNAPSHOT_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{4,8}$")

SIDECAR_DIR = ".odysseus"
PROJECT_FILE = "project.json"
PROMPTS_FILE = "prompts.jsonl"
SNAPSHOTS_DIR = "snapshots"
SNAPSHOT_FILES_DIR = "files"
SNAPSHOT_META_FILE = "project.json"
SNAPSHOT_KEEP = 20

_IGNORED_TOP_LEVEL: frozenset[str] = frozenset({
    SIDECAR_DIR, "node_modules", ".git", ".next", ".turbo", "dist", ".cache",
})


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ProjectRecord:
    """A project stored on disk."""

    id: str
    name: str
    template: str = "next"
    project_type: str = "app"  # "app" or "full"
    starter_name: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "project_type": self.project_type,
            "starter_name": self.starter_name,
            "created_at": self.created_at or _now().isoformat(),
            "updated_at": self.updated_at or _now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectRecord:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            template=data.get("template", "next"),
            project_type=data.get("project_type", "app"),
            starter_name=data.get("starter_name", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class PromptRecord:
    """A single prompt-reply pair in a project's history."""

    id: str
    role: str  # "user", "assistant", "system", "tool"
    content: str
    created_at: Optional[str] = None
    snapshot_id: Optional[str] = None


@dataclass
class SnapshotRecord:
    """A point-in-time snapshot of a project."""

    id: str
    created_at: str
    user_prompt: str = ""
    kind: str = "pre-turn"  # "pre-turn", "manual"


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Convert a project name to a URL-safe slug."""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned)
    cleaned = cleaned.strip("-")
    cleaned = cleaned[:63]
    if not cleaned or not cleaned[0].isalnum():
        return ""
    return cleaned


def validate_slug(slug: str) -> bool:
    """Check if a string is a valid project slug."""
    return bool(SLUG_RE.fullmatch(slug))


def safe_join(root: Path, rel: str) -> Path:
    """Resolve *rel* against *root*, blocking traversal + absolute paths.

    Raises ValueError if the path escapes the root.
    """
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {rel!r}")

    root_resolved = root.resolve(strict=False)
    candidate = (root / rel_path).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {rel!r}") from exc
    return candidate


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class Storage:
    """Local filesystem storage for projects, files, snapshots, and prompts.

    Directory layout:
        <root>/
            <slug>/
                .odysseus/
                    project.json
                    prompts.jsonl
                    snapshots/
                        <snapshot-id>/
                            project.json
                            files/
                                ...
                app/            (project files)
                package.json
                ...
    """

    def __init__(self, root: Path | str = "data/projects"):
        self.root = Path(root).expanduser().resolve()
        self._write_lock = Lock()

    # ------------------------------------------------------------------
    # Project CRUD
    # ------------------------------------------------------------------

    def ensure_root(self) -> None:
        """Create the storage root directory if it doesn't exist."""
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, slug: str) -> Path:
        """Get the on-disk directory for a project."""
        if not validate_slug(slug):
            raise ValueError(f"invalid project slug: {slug!r}")
        return self.root / slug

    def sidecar_dir(self, slug: str) -> Path:
        """Get the .odysseus metadata directory for a project."""
        return self.project_dir(slug) / SIDECAR_DIR

    def unique_slug(self, name: str) -> str:
        """Generate a unique slug from a name, appending -2, -3, etc. if needed."""
        base = slugify(name)
        if not base:
            base = f"project-{uuid.uuid4().hex[:8]}"
        candidate = base
        n = 2
        while (self.root / candidate).exists():
            suffix = f"-{n}"
            candidate = f"{base[:63 - len(suffix)]}{suffix}"
            n += 1
        return candidate

    def create_project(
        self,
        name: str,
        template: str = "next",
        project_type: str = "app",
        starter_name: Optional[str] = None,
    ) -> ProjectRecord:
        """Create a new project directory with starter files.

        Args:
            name: Human-friendly project name
            template: Starter template ('next', 'next-minimal', 'express', etc.)
            project_type: 'app' or 'full' (full creates code/ and docs/ subdirs)
            starter_name: Override starter name (defaults to template)

        Returns:
            ProjectRecord for the created project
        """
        self.ensure_root()

        slug = self.unique_slug(name)
        proj_dir = self.project_dir(slug)
        sidecar = self.sidecar_dir(slug)

        # Create directories
        sidecar.mkdir(parents=True, exist_ok=True)

        if project_type == "full":
            (proj_dir / "code").mkdir(exist_ok=True)
            (proj_dir / "docs").mkdir(exist_ok=True)

        now = _now()
        resolved_starter = starter_name or template
        record = ProjectRecord(
            id=slug,
            name=name.strip(),
            template=template,
            project_type=project_type,
            starter_name=resolved_starter,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )

        self._write_project_json(slug, record)

        # Create prompts file
        prompts_file = sidecar / PROMPTS_FILE
        prompts_file.touch()

        # Scaffold starter files
        starter_files = get_starter(resolved_starter)
        if starter_files:
            prefix = "code/" if project_type == "full" else ""
            for rel_path, content in starter_files.items():
                self.write_file(slug, f"{prefix}{rel_path}", content)

        logger.info(
            "Created project '%s' (slug=%s, starter=%s)",
            name, slug, resolved_starter,
        )
        return record

    def list_projects(self) -> List[ProjectRecord]:
        """List all projects sorted by most recently updated."""
        if not self.root.exists():
            return []

        records: List[ProjectRecord] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            if not validate_slug(child.name):
                continue
            rec = self._try_read_project_json(child.name)
            if rec is not None:
                records.append(rec)

        records.sort(key=lambda r: r.updated_at or "", reverse=True)
        return records

    def get_project(self, slug: str) -> Optional[ProjectRecord]:
        """Get a project by slug, or None if it doesn't exist."""
        if not validate_slug(slug):
            return None
        return self._try_read_project_json(slug)

    def delete_project(self, slug: str) -> bool:
        """Delete a project and all its files.

        Returns True if deleted, False if it didn't exist.
        """
        if not validate_slug(slug):
            return False
        target = self.project_dir(slug).resolve()
        if not target.exists():
            return False

        try:
            target.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("refusing to delete path outside storage root") from exc

        shutil.rmtree(target)
        logger.info("Deleted project '%s'", slug)
        return True

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def read_tree(self, slug: str) -> Dict[str, Any]:
        """Build a nested dict of the project's file tree with contents.

        Returns:
            {
                "filename.txt": {"file": {"contents": "..."}},
                "dirname": {"directory": {"filename.txt": {"file": {"contents": "..."}}}},
            }
        """
        proj = self.project_dir(slug)
        if not proj.exists():
            raise FileNotFoundError(f"project not found: {slug}")

        def walk(dir_path: Path, is_root: bool) -> Dict[str, Any]:
            tree: Dict[str, Any] = {}
            for entry in sorted(dir_path.iterdir(), key=lambda p: p.name):
                name = entry.name
                if is_root and name in _IGNORED_TOP_LEVEL:
                    continue
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    tree[name] = {"directory": walk(entry, is_root=False)}
                elif entry.is_file():
                    try:
                        contents = entry.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError):
                        continue
                    tree[name] = {"file": {"contents": contents}}
            return tree

        return walk(proj, is_root=True)

    def read_file(self, slug: str, rel_path: str) -> Optional[str]:
        """Read a file's contents from a project.

        Returns None if the file doesn't exist or can't be read.
        """
        try:
            target = safe_join(self.project_dir(slug), rel_path)
        except ValueError:
            return None

        if not target.is_file():
            return None

        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def write_file(self, slug: str, rel_path: str, content: str) -> Path:
        """Write content to a file in a project, creating directories as needed."""
        proj = self.project_dir(slug)
        if not proj.exists():
            raise FileNotFoundError(f"project not found: {slug}")

        target = safe_join(proj, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._write_lock:
            target.write_text(content, encoding="utf-8")

        self._touch_project(slug)
        return target

    def delete_file(self, slug: str, rel_path: str) -> bool:
        """Delete a file in a project.

        Returns True if deleted, False if it didn't exist.
        """
        try:
            target = safe_join(self.project_dir(slug), rel_path)
        except ValueError:
            return False

        if not target.exists():
            return False

        with self._write_lock:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

        self._touch_project(slug)
        return True

    # ------------------------------------------------------------------
    # Prompt history
    # ------------------------------------------------------------------

    def append_prompt(
        self,
        slug: str,
        role: str,
        content: str,
        *,
        snapshot_id: Optional[str] = None,
    ) -> PromptRecord:
        """Append a prompt record to the project's history."""
        sidecar = self.sidecar_dir(slug)
        sidecar.mkdir(parents=True, exist_ok=True)

        record = PromptRecord(
            id=uuid.uuid4().hex,
            role=role,
            content=content,
            created_at=_now().isoformat(),
            snapshot_id=snapshot_id,
        )

        path = sidecar / PROMPTS_FILE
        payload = json.dumps(record.__dict__, ensure_ascii=False)

        with self._write_lock, open(path, "a", encoding="utf-8") as fp:
            fp.write(payload + "\n")
            fp.flush()
            os.fsync(fp.fileno())

        self._touch_project(slug)
        return record

    def read_prompts(self, slug: str) -> List[PromptRecord]:
        """Read all prompt records for a project."""
        path = self.sidecar_dir(slug) / PROMPTS_FILE
        if not path.exists():
            return []

        records: List[PromptRecord] = []
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(PromptRecord(**data))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Skipping malformed prompt: %s", exc)
                    continue
        return records

    def pop_last_assistant_prompt(self, slug: str) -> Optional[PromptRecord]:
        """Remove and return the most recent assistant prompt.

        Returns None if no assistant prompt is found.
        """
        path = self.sidecar_dir(slug) / PROMPTS_FILE
        if not path.exists():
            return None

        with self._write_lock:
            raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

            drop_idx: Optional[int] = None
            dropped: Optional[PromptRecord] = None

            for i in range(len(raw_lines) - 1, -1, -1):
                line = raw_lines[i].strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("role") == "assistant":
                        drop_idx = i
                        dropped = PromptRecord(**data)
                        break
                except (json.JSONDecodeError, TypeError):
                    continue

            if drop_idx is None or dropped is None:
                return None

            remaining = raw_lines[:drop_idx] + raw_lines[drop_idx + 1:]
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fp:
                fp.writelines(remaining)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp, path)

        self._touch_project(slug)
        return dropped

    def clear_prompts(self, slug: str) -> bool:
        """Clear all prompt history for a project.

        Returns True if cleared, False if no prompts file existed.
        """
        path = self.sidecar_dir(slug) / PROMPTS_FILE
        if not path.exists():
            return False
        with self._write_lock:
            path.write_text("", encoding="utf-8")
        return True

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _snapshots_dir(self, slug: str) -> Path:
        return self.sidecar_dir(slug) / SNAPSHOTS_DIR

    def _snapshot_dir(self, slug: str, snapshot_id: str) -> Path:
        if not SNAPSHOT_ID_RE.fullmatch(snapshot_id):
            raise ValueError(f"invalid snapshot id: {snapshot_id!r}")
        return self._snapshots_dir(slug) / snapshot_id

    @staticmethod
    def _new_snapshot_id(now: datetime) -> str:
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def create_snapshot(
        self,
        slug: str,
        *,
        user_prompt: str = "",
        kind: str = "pre-turn",
    ) -> SnapshotRecord:
        """Create a point-in-time snapshot of the project.

        Copies all project files (excluding .odysseus, node_modules, etc.)
        into a snapshot directory under .odysseus/snapshots/<id>/.
        """
        proj = self.project_dir(slug)
        if not proj.exists():
            raise FileNotFoundError(f"project not found: {slug}")

        created_at = _now()

        # Find a unique snapshot ID
        for _ in range(8):
            snapshot_id = self._new_snapshot_id(created_at)
            dest = self._snapshot_dir(slug, snapshot_id)
            if not dest.exists():
                break
        else:
            raise RuntimeError(f"failed to allocate unique snapshot id for {slug}")

        record = SnapshotRecord(
            id=snapshot_id,
            created_at=created_at.isoformat(),
            user_prompt=user_prompt[:4000],
            kind=kind,
        )

        files_dir = dest / SNAPSHOT_FILES_DIR
        with self._write_lock:
            dest.mkdir(parents=True, exist_ok=False)
            files_dir.mkdir(parents=True, exist_ok=False)

            for entry in proj.iterdir():
                if entry.name in _IGNORED_TOP_LEVEL:
                    continue
                if entry.is_symlink():
                    continue
                target = files_dir / entry.name
                if entry.is_dir():
                    shutil.copytree(
                        entry,
                        target,
                        symlinks=False,
                        ignore=shutil.ignore_patterns(*_IGNORED_TOP_LEVEL),
                    )
                elif entry.is_file():
                    shutil.copy2(entry, target)

            # Write metadata
            meta_path = dest / SNAPSHOT_META_FILE
            with open(meta_path, "w", encoding="utf-8") as fp:
                json.dump(record.__dict__, fp, indent=2)
                fp.flush()
                os.fsync(fp.fileno())

        self._prune_snapshots(slug)
        logger.info("Created snapshot %s for '%s'", snapshot_id, slug)
        return record

    def list_snapshots(self, slug: str) -> List[SnapshotRecord]:
        """List all snapshots for a project, newest first."""
        root_path = self._snapshots_dir(slug)
        if not root_path.exists():
            return []

        records: List[SnapshotRecord] = []
        for child in sorted(root_path.iterdir()):
            if not child.is_dir():
                continue
            if not SNAPSHOT_ID_RE.fullmatch(child.name):
                continue
            meta = child / SNAPSHOT_META_FILE
            if not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                records.append(SnapshotRecord(**data))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping malformed snapshot %s: %s", child.name, exc)
                continue

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def restore_snapshot(self, slug: str, snapshot_id: str) -> bool:
        """Restore a project to a previous snapshot state.

        Returns True on success, False if the snapshot doesn't exist.
        """
        proj = self.project_dir(slug)
        if not proj.exists():
            raise FileNotFoundError(f"project not found: {slug}")

        snap_dir = self._snapshot_dir(slug, snapshot_id)
        files_dir = snap_dir / SNAPSHOT_FILES_DIR
        if not snap_dir.is_dir() or not files_dir.is_dir():
            return False

        with self._write_lock:
            # Remove current files (except ignored dirs)
            for entry in list(proj.iterdir()):
                if entry.name in _IGNORED_TOP_LEVEL:
                    continue
                if entry.is_symlink():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
                elif entry.is_file():
                    entry.unlink()

            # Copy snapshot files back
            for entry in files_dir.iterdir():
                target = proj / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, symlinks=False)
                elif entry.is_file():
                    shutil.copy2(entry, target)

        self._touch_project(slug)
        logger.info("Restored '%s' to snapshot %s", slug, snapshot_id)
        return True

    def delete_snapshot(self, slug: str, snapshot_id: str) -> bool:
        """Delete a specific snapshot."""
        snap_dir = self._snapshot_dir(slug, snapshot_id)
        if not snap_dir.exists():
            return False

        target = snap_dir.resolve()
        try:
            target.relative_to(self._snapshots_dir(slug).resolve())
        except ValueError as exc:
            raise ValueError("refusing to delete path outside snapshots root") from exc

        with self._write_lock:
            shutil.rmtree(target)
        logger.info("Deleted snapshot %s for '%s'", snapshot_id, slug)
        return True

    def _prune_snapshots(self, slug: str) -> None:
        """Remove oldest snapshots exceeding SNAPSHOT_KEEP."""
        records = self.list_snapshots(slug)
        if len(records) <= SNAPSHOT_KEEP:
            return
        for rec in records[SNAPSHOT_KEEP:]:
            try:
                self.delete_snapshot(slug, rec.id)
            except Exception as exc:
                logger.warning("Failed to prune snapshot %s: %s", rec.id, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_json_path(self, slug: str) -> Path:
        return self.sidecar_dir(slug) / PROJECT_FILE

    def _write_project_json(self, slug: str, record: ProjectRecord) -> None:
        """Write project metadata to .odysseus/project.json."""
        path = self._project_json_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False)
        with self._write_lock, open(path, "w", encoding="utf-8") as fp:
            fp.write(payload)
            fp.flush()
            os.fsync(fp.fileno())

    def _try_read_project_json(self, slug: str) -> Optional[ProjectRecord]:
        """Read project metadata, returning None if it doesn't exist."""
        path = self._project_json_path(slug)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ProjectRecord.from_dict(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read project.json for '%s': %s", slug, exc)
            return None

    def _touch_project(self, slug: str) -> None:
        """Update the project's updated_at timestamp."""
        rec = self._try_read_project_json(slug)
        if rec is None:
            return
        updated = ProjectRecord(
            id=rec.id,
            name=rec.name,
            template=rec.template,
            project_type=rec.project_type,
            starter_name=rec.starter_name,
            created_at=rec.created_at,
            updated_at=_now().isoformat(),
        )
        self._write_project_json(slug, updated)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def ensure_next_preview_layout(self, slug: str) -> None:
        """Ensure a Next.js project has the basic files needed for preview."""
        rec = self._try_read_project_json(slug)
        if rec is None or rec.template != "next":
            return
        proj = self.project_dir(slug)
        if not proj.exists():
            return

        # Ensure starter files exist
        starter = get_starter("next")
        if starter is None:
            return

        for rel_path, content in starter.items():
            try:
                target = safe_join(proj, rel_path)
            except ValueError:
                continue
            if target.is_file():
                continue
            self.write_file(slug, rel_path, content)

    @property
    def root_path(self) -> Path:
        """The storage root directory."""
        return self.root


# ---------------------------------------------------------------------------
# Singleton / convenience
# ---------------------------------------------------------------------------

_default_storage: Optional[Storage] = None


def get_storage(root: Optional[Path | str] = None) -> Storage:
    """Get or create the default storage instance."""
    global _default_storage
    if _default_storage is None or root is not None:
        _default_storage = Storage(root or "data/projects")
    return _default_storage


def reset_storage() -> None:
    """Reset the default storage singleton (useful in tests)."""
    global _default_storage
    _default_storage = None


def iter_ignored_top_level() -> Iterable[str]:
    """Return the set of top-level directory names that are ignored."""
    return iter(_IGNORED_TOP_LEVEL)
