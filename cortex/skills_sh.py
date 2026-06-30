"""
cortex/skills_sh.py — Bridge to skills.sh marketplace and skill discovery.

Searches for skills from multiple sources:
1. Local Hermes skills directory (~/.hermes/skills/)
2. GitHub code search for SKILL.md files
3. Direct URL-based skill registries (fallback)

Each source is tried in order until results are found.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.request import urlopen, Request
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

# Local Hermes skills directory (primary source)
HERMES_SKILLS_DIR = os.path.expanduser("~/.hermes/skills")
PROFILE_SKILLS_DIR = os.path.expanduser("~/.hermes/profiles/swarm1/skills")

# GitHub search API (requires token for high rate limits, works without for low volume)
GITHUB_SEARCH_URL = "https://api.github.com/search/code?q=SKILL.md+repo:vercel-labs/skills+extension:md&per_page=30"

# Static skill index (known working skills)
_STATIC_SKILLS: List[Dict] = [
    {
        "name": "python-backend",
        "description": "Python backend development patterns",
        "category": "dev",
        "tags": ["python", "backend"],
    },
    {
        "name": "react-frontend",
        "description": "React frontend development patterns",
        "category": "dev",
        "tags": ["react", "frontend"],
    },
    {
        "name": "docker-deployment",
        "description": "Docker deployment patterns",
        "category": "devops",
        "tags": ["docker", "deployment"],
    },
    {
        "name": "database-design",
        "description": "Database design patterns",
        "category": "dev",
        "tags": ["database", "sql"],
    },
    {
        "name": "testing-patterns",
        "description": "Testing patterns and best practices",
        "category": "dev",
        "tags": ["testing", "pytest"],
    },
    {
        "name": "api-design",
        "description": "REST API design patterns",
        "category": "dev",
        "tags": ["api", "rest"],
    },
    {
        "name": "security-hardening",
        "description": "Security hardening patterns",
        "category": "security",
        "tags": ["security", "hardening"],
    },
    {
        "name": "code-review",
        "description": "Code review best practices",
        "category": "dev",
        "tags": ["code-review", "quality"],
    },
    {
        "name": "cli-tools",
        "description": "CLI tool development patterns",
        "category": "dev",
        "tags": ["cli", "python"],
    },
    {
        "name": "documentation",
        "description": "Documentation generation patterns",
        "category": "writing",
        "tags": ["docs", "markdown"],
    },
]


@dataclass
class SkillsShSkill:
    name: str
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    installs: int = 0


def _fetch_json(url: str) -> Optional[Any]:
    """Fetch JSON from a URL."""
    try:
        req = Request(url, headers={"User-Agent": "OdysseusCortex/1.0"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        logger.debug("HTTP %s fetching %s: %s", e.code, url, e.reason)
        return None
    except Exception as e:
        logger.debug("error fetching %s: %s", url, e)
        return None


def _search_local_skills(query: str, limit: int = 20) -> List[Dict]:
    """Search local Hermes skills directories for matching skills."""
    results = []
    query_lower = query.lower()

    search_dirs = []
    for d in [HERMES_SKILLS_DIR, PROFILE_SKILLS_DIR]:
        p = Path(d)
        if p.exists():
            search_dirs.append(p)

    for skills_dir in search_dirs:
        for skill_file in skills_dir.rglob("SKILL.md"):
            try:
                content = skill_file.read_text(encoding="utf-8", errors="ignore")
                # Extract name from frontmatter
                name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
                name = (
                    name_match.group(1).strip()
                    if name_match
                    else skill_file.parent.name
                )
                desc_match = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
                desc = desc_match.group(1).strip() if desc_match else ""

                # Extract category from path
                rel_path = skill_file.relative_to(skills_dir)
                category = (
                    rel_path.parent.name if rel_path.parent.name != "." else "general"
                )
                tags = (
                    [category, rel_path.parent.parent.name]
                    if rel_path.parent.parent.name != "."
                    else [category]
                )

                # Match against query
                search_text = f"{name} {desc} {category}".lower()
                if query_lower in search_text:
                    results.append(
                        {
                            "name": name,
                            "description": desc or f"Skill from {rel_path.parent}",
                            "category": category,
                            "tags": [t for t in tags if t],
                            "platforms": ["hermes"],
                            "installs": 1,
                            "source": "local",
                            "path": str(skill_file),
                        }
                    )
                    if len(results) >= limit:
                        return results
            except Exception as e:
                logger.debug("Error reading %s: %s", skill_file, e)
                continue

    return results


def _search_github(query: str, limit: int = 20) -> List[Dict]:
    """Search GitHub for SKILL.md files matching the query."""
    results = []
    query_lower = query.lower()

    # Try GitHub search
    data = _fetch_json(GITHUB_SEARCH_URL)
    if data and isinstance(data, dict):
        items = data.get("items", [])
        for item in items:
            name = item.get("name", "")
            path = item.get("path", "")
            repo = item.get("repository", {}).get("full_name", "")
            if query_lower in path.lower() or query_lower in name.lower():
                results.append(
                    {
                        "name": Path(path).stem,
                        "description": f"From {repo}/{path}",
                        "category": Path(path).parent.name
                        if "/" in path
                        else "general",
                        "tags": [repo.split("/")[-1] if "/" in repo else ""],
                        "platforms": ["github"],
                        "installs": 0,
                        "source": "github",
                    }
                )
                if len(results) >= limit:
                    return results

    return results


def _search_static(query: str, limit: int = 20) -> List[Dict]:
    """Search the static skill index."""
    results = []
    query_lower = query.lower()
    for skill in _STATIC_SKILLS:
        search_text = (
            f"{skill['name']} {skill['description']} {' '.join(skill['tags'])}".lower()
        )
        if query_lower in search_text:
            results.append({**skill, "source": "index"})
            if len(results) >= limit:
                break
    return results


def search(query: str, limit: int = 20) -> List[Dict]:
    """Search for skills matching the query across multiple sources.

    Tries sources in order: local → GitHub → static index.
    Returns combined results up to ``limit``.
    """
    if not query or not query.strip():
        return []

    seen = set()
    results = []

    for searcher in [_search_local_skills, _search_github, _search_static]:
        try:
            batch = searcher(query, limit - len(results))
            for item in batch:
                dedup_key = f"{item.get('name', '')}:{item.get('source', '')}"
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    results.append(item)
            if len(results) >= limit:
                break
        except Exception as e:
            logger.debug("Skill search error (%s): %s", searcher.__name__, e)
            continue

    return results[:limit]


def get_skill_markdown(name: str) -> Optional[str]:
    """Fetch the raw SKILL.md content for a named skill.

    Searches local skills directories first, then falls back to URL fetch.
    """
    # Local search
    for base in [Path(HERMES_SKILLS_DIR), Path(PROFILE_SKILLS_DIR)]:
        if not base.exists():
            continue
        for skill_file in base.rglob("SKILL.md"):
            try:
                content = skill_file.read_text(encoding="utf-8", errors="ignore")
                name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
                skill_name = (
                    name_match.group(1).strip()
                    if name_match
                    else skill_file.parent.name
                )
                if skill_name.lower() == name.lower():
                    return content
            except Exception:
                continue

    return None


def install_skill(skills_manager: Any, name: str, owner: Optional[str] = None) -> Dict:
    """Install a skill from local registry into Odysseus.

    Args:
        skills_manager: Odysseus SkillsManager instance
        name: Skill name to install
        owner: Optional owner username

    Returns:
        Dict with install result
    """
    md_content = get_skill_markdown(name)
    if not md_content:
        return {"ok": False, "error": f"skill '{name}' not found in local registry"}

    md_content = _ensure_frontmatter(md_content, name)
    try:
        skill = skills_manager.add_skill(md_content, source="imported", owner=owner)
        return {"ok": True, "skill_id": skill.id if hasattr(skill, "id") else name}
    except Exception as e:
        logger.exception("failed to install skill '%s'", name)
        return {"ok": False, "error": str(e)}


def _ensure_frontmatter(content: str, name: str) -> str:
    """Ensure SKILL.md content has proper frontmatter."""
    if content.startswith("---"):
        return content
    return f"""---
name: {name}
description: Skill imported from skills directory
category: general
source: imported
status: draft
confidence: 0.5
---

{content}

"""
