"""
cortex/skills_sh.py — Bridge to skills.sh marketplace.

Skills.sh is a Vercel-hosted agent skills directory. Skills are
stored on GitHub repos as SKILL.md files with YAML frontmatter.

This module searches and installs skills from skills.sh via the
GitHub API (since skills.sh is a Next.js SPA with no public REST API).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from urllib.request import urlopen, Request
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

SKILLS_SH_INDEX_URL = "https://raw.githubusercontent.com/vercel-labs/skills/refs/heads/main/data/skills.json"
SKILLS_SH_REPO = "https://api.github.com/repos/vercel-labs/skills/contents/data"
SKILLS_SH_RAW = "https://raw.githubusercontent.com/vercel-labs/skills/main/data"

ODYSSEUS_SKILLS_DIR = os.environ.get(
    "ODYSSEUS_SKILLS_DIR",
    os.path.join(os.getcwd(), "data", "skills"),
)


@dataclass
class SkillsShSkill:
    name: str
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    installs: int = 0


def _fetch_json(url: str) -> Optional[Any]:
    """Fetch JSON from a URL (GitHub API or raw)."""
    try:
        req = Request(url, headers={"User-Agent": "OdysseusCortex/1.0"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        logger.warning("HTTP %s fetching %s: %s", e.code, url, e.reason)
        return None
    except Exception as e:
        logger.warning("error fetching %s: %s", url, e)
        return None


def _fetch_text(url: str) -> Optional[str]:
    """Fetch raw text from a URL."""
    try:
        req = Request(url, headers={"User-Agent": "OdysseusCortex/1.0"})
        with urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        logger.warning("error fetching text from %s: %s", url, e)
        return None


def search(query: str, limit: int = 20) -> List[Dict]:
    """Search skills.sh for skills matching the query.

    Uses the skills.sh GitHub repo index to find skills.
    """
    index = _fetch_json(SKILLS_SH_INDEX_URL)
    if index is None:
        return _search_github_fallback(query, limit)

    results = []
    query_lower = query.lower()
    for skill in index:
        if not isinstance(skill, dict):
            continue
        name = skill.get("name", "")
        desc = skill.get("description", "")
        tags = " ".join(skill.get("tags", []))
        if (query_lower in name.lower()
                or query_lower in desc.lower()
                or query_lower in tags.lower()):
            results.append({
                "name": name,
                "description": desc,
                "category": skill.get("category", "general"),
                "tags": skill.get("tags", []),
                "platforms": skill.get("platforms", []),
                "installs": skill.get("installs", 0),
                "source": "skills.sh",
            })
            if len(results) >= limit:
                break
    return results


def _search_github_fallback(query: str, limit: int = 20) -> List[Dict]:
    """Fallback: search via GitHub API contents endpoint."""
    contents = _fetch_json(SKILLS_SH_REPO)
    if not contents or not isinstance(contents, list):
        return []
    results = []
    query_lower = query.lower()
    for item in contents:
        if item.get("type") != "dir":
            continue
        name = item.get("name", "")
        if query_lower in name.lower():
            results.append({
                "name": name,
                "description": f"Skill: {name}",
                "category": "general",
                "tags": [name],
                "platforms": [],
                "installs": 0,
                "source": "skills.sh",
            })
            if len(results) >= limit:
                break
    return results


def get_skill_markdown(name: str) -> Optional[str]:
    """Fetch the raw SKILL.md content for a named skill from skills.sh."""
    for category in ["general", "dev", "data", "writing", "productivity"]:
        url = f"{SKILLS_SH_RAW}/{category}/{name}/SKILL.md"
        content = _fetch_text(url)
        if content:
            return content
    return None


def install_skill(skills_manager: Any, name: str, owner: Optional[str] = None) -> Dict:
    """Install a skill from skills.sh into Odysseus.

    Args:
        skills_manager: Odysseus SkillsManager instance
        name: Skill name to install
        owner: Optional owner username

    Returns:
        Dict with install result
    """
    md_content = get_skill_markdown(name)
    if not md_content:
        return {"ok": False, "error": f"skill '{name}' not found on skills.sh"}

    md_content = _ensure_frontmatter(md_content, name)
    try:
        skill = skills_manager.add_skill(md_content, source="imported", owner=owner)
        return {"ok": True, "skill_id": skill.id if hasattr(skill, 'id') else name}
    except Exception as e:
        logger.exception("failed to install skill '%s'", name)
        return {"ok": False, "error": str(e)}


def _ensure_frontmatter(content: str, name: str) -> str:
    """Ensure SKILL.md content has proper frontmatter."""
    if content.startswith("---"):
        return content
    return f"""---
name: {name}
description: Skill imported from skills.sh
category: general
source: imported
status: draft
confidence: 0.5
---

{content}
"""
