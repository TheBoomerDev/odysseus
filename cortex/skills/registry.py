"""
cortex/skills/registry.py — Local skill registry + skills.sh marketplace bridge.

Manages CRUD for skills in ~/.odysseus/skills/ using the standard
SKILL.md format (YAML frontmatter + markdown body), compatible with
Hermes Agent and skills.sh marketplace.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_SKILLS_DIR = Path.home() / ".odysseus" / "skills"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A single skill parsed from SKILL.md."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    trigger: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    body: str = ""
    file_path: str = ""
    raw_frontmatter: Dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Optional[Path]:
        """Directory containing this skill, if known."""
        if self.file_path:
            return Path(self.file_path).parent
        return None

    def to_markdown(self) -> str:
        """Serialize back to SKILL.md format."""
        fm_lines = [
            "---",
            f"name: {self.name}",
            f"description: {self.description}",
            f"version: {self.version}",
            f"author: {self.author}",
        ]
        if self.tags:
            fm_lines.append(f"tags: [{', '.join(self.tags)}]")
        if self.categories:
            fm_lines.append(f"categories: [{', '.join(self.categories)}]")
        if self.trigger:
            fm_lines.append(f"trigger: [{', '.join(self.trigger)}]")
        if self.requires:
            fm_lines.append(f"requires: [{', '.join(self.requires)}]")
        fm_lines.append("---")
        fm_lines.append("")
        fm_lines.append(self.body)
        return "\n".join(fm_lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL
)
_LIST_RE = re.compile(r"^\[(.*)\]$")


def _parse_list(raw: str) -> List[str]:
    """Parse a YAML-ish list like [a, b, c] into a Python list."""
    match = _LIST_RE.match(raw.strip())
    if match:
        inner = match.group(1).strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",")]
    return [raw.strip().strip("'\"")]


def parse_skill_md(content: str, file_path: str = "") -> Optional[Skill]:
    """Parse a SKILL.md file into a Skill dataclass.

    Returns None if the file doesn't have valid frontmatter.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return None

    fm_text = match.group(1)
    body = match.group(2).strip()

    # Parse simple YAML frontmatter (key: value)
    fm: Dict[str, Any] = {}
    for line in fm_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        fm[key] = value

    # Build list fields
    tags = _parse_list(fm.get("tags", ""))
    categories = _parse_list(fm.get("categories", ""))
    trigger = _parse_list(fm.get("trigger", ""))
    requires = _parse_list(fm.get("requires", ""))

    return Skill(
        name=fm.get("name", ""),
        description=fm.get("description", ""),
        version=fm.get("version", "0.1.0"),
        author=fm.get("author", ""),
        tags=tags,
        categories=categories,
        trigger=trigger,
        requires=requires,
        body=body,
        file_path=file_path,
        raw_frontmatter=fm,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Manages skills in a local directory.

    Args:
        skills_dir: Directory containing skill subdirectories.
                    Defaults to ~/.odysseus/skills/
    """

    def __init__(self, skills_dir: Optional[Path | str] = None):
        self.skills_dir = Path(skills_dir or DEFAULT_SKILLS_DIR)
        self._cache: Dict[str, Skill] = {}
        self._loaded = False

    def ensure_dir(self) -> None:
        """Create the skills directory if it doesn't exist."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Scan skills_dir and load all skills into cache."""
        self._cache.clear()
        if not self.skills_dir.exists():
            self._loaded = True
            return

        for child in sorted(self.skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
                skill = parse_skill_md(content, str(skill_md))
                if skill and skill.name:
                    self._cache[skill.name] = skill
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Failed to parse %s: %s", skill_md, exc)

        self._loaded = True
        logger.info("Loaded %d skills from %s", len(self._cache), self.skills_dir)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def reload(self) -> None:
        """Force a reload from disk."""
        self._loaded = False
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_skills(self) -> List[Skill]:
        """Return all registered skills."""
        self._ensure_loaded()
        return sorted(self._cache.values(), key=lambda s: s.name)

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        self._ensure_loaded()
        return self._cache.get(name)

    def has_skill(self, name: str) -> bool:
        """Check if a skill exists."""
        self._ensure_loaded()
        return name in self._cache

    def install_skill(self, skill: Skill) -> Path:
        """Write a skill to disk in the skills directory.

        Creates the skill subdirectory and SKILL.md file.
        """
        self.ensure_dir()
        skill_dir = self.skills_dir / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(skill.to_markdown(), encoding="utf-8")

        # Update cache
        skill.file_path = str(skill_md)
        self._cache[skill.name] = skill

        logger.info("Installed skill '%s' at %s", skill.name, skill_dir)
        return skill_dir

    def delete_skill(self, name: str) -> bool:
        """Delete a skill directory.

        Returns True if deleted, False if it didn't exist.
        """
        skill = self.get_skill(name)
        if skill is None:
            return False

        skill_dir = skill.dir
        if skill_dir is None or not skill_dir.exists():
            return False

        shutil.rmtree(skill_dir)
        self._cache.pop(name, None)
        logger.info("Deleted skill '%s'", name)
        return True

    def update_skill(self, skill: Skill) -> Path:
        """Update an existing skill (or install if new)."""
        return self.install_skill(skill)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_by_tag(self, tag: str) -> List[Skill]:
        """Find skills that have a specific tag."""
        self._ensure_loaded()
        return [s for s in self._cache.values() if tag in s.tags]

    def find_by_category(self, category: str) -> List[Skill]:
        """Find skills in a specific category."""
        self._ensure_loaded()
        return [
            s for s in self._cache.values()
            if category in s.categories
        ]

    def find_by_trigger(self, text: str) -> List[Skill]:
        """Find skills whose trigger phrases appear in *text*."""
        self._ensure_loaded()
        text_lower = text.lower()
        results: List[Skill] = []
        for skill in self._cache.values():
            for trigger in skill.trigger:
                if trigger.lower() in text_lower:
                    results.append(skill)
                    break
        return results

    # ------------------------------------------------------------------
    # Marketplace bridge (skills.sh)
    # ------------------------------------------------------------------

    async def install_from_marketplace(
        self,
        name: str,
        marketplace_url: str = "https://skills.sh",
    ) -> Optional[Skill]:
        """Install a skill from skills.sh marketplace.

        Downloads the skill from the marketplace and installs it locally.
        """
        import httpx

        url = f"{marketplace_url.rstrip('/')}/api/skills/{name}"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch skill '%s' from marketplace: %s", name, exc)
            return None

        skill = Skill(
            name=data.get("name", name),
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            author=data.get("author", "marketplace"),
            tags=data.get("tags", []),
            categories=data.get("categories", []),
            trigger=data.get("trigger", []),
            requires=data.get("requires", []),
            body=data.get("body", data.get("content", "")),
        )

        self.install_skill(skill)
        logger.info("Installed '%s' from marketplace", name)
        return skill

    async def publish_to_marketplace(
        self,
        name: str,
        marketplace_url: str = "https://skills.sh",
        api_key: Optional[str] = None,
    ) -> bool:
        """Publish a local skill to the skills.sh marketplace."""
        import httpx

        skill = self.get_skill(name)
        if skill is None:
            logger.error("Skill '%s' not found locally", name)
            return False

        url = f"{marketplace_url.rstrip('/')}/api/skills"
        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "name": skill.name,
            "description": skill.description,
            "version": skill.version,
            "author": skill.author,
            "tags": skill.tags,
            "categories": skill.categories,
            "trigger": skill.trigger,
            "requires": skill.requires,
            "body": skill.body,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            logger.info("Published '%s' to marketplace", name)
            return True
        except Exception as exc:
            logger.error("Failed to publish '%s': %s", name, exc)
            return False

    async def search_marketplace(
        self,
        query: str,
        marketplace_url: str = "https://skills.sh",
    ) -> List[Dict[str, Any]]:
        """Search the skills.sh marketplace for skills matching *query*."""
        import httpx

        url = f"{marketplace_url.rstrip('/')}/api/skills/search"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params={"q": query})
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, list) else data.get("results", [])
        except Exception as exc:
            logger.error("Marketplace search failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: Optional[SkillRegistry] = None


def get_registry(skills_dir: Optional[Path | str] = None) -> SkillRegistry:
    """Get or create the default SkillRegistry singleton."""
    global _registry
    if _registry is None or skills_dir is not None:
        _registry = SkillRegistry(skills_dir)
    return _registry


def reset_registry() -> None:
    """Reset the singleton (useful in tests)."""
    global _registry
    _registry = None
