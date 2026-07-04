"""
cortex/skills/matcher.py — Semantic matching of tasks to skills.

Finds the best skill for a given task using:
  1. Trigger phrase matching (exact keyword hits)
  2. Category matching (task type → skill category)
  3. Tag similarity (overlapping tags)
  4. Description keyword overlap
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .registry import Skill, SkillRegistry, get_registry
from .tracker import SkillTracker

logger = logging.getLogger(__name__)


@dataclass
class SkillMatch:
    """A skill match result with relevance score."""

    skill: Skill
    score: float
    matched_on: str  # "trigger", "category", "tag", "description"


class SkillMatcher:
    """Matches tasks to skills using multi-signal scoring."""

    # Scoring weights
    TRIGGER_WEIGHT = 1.0
    CATEGORY_WEIGHT = 0.7
    TAG_WEIGHT = 0.5
    DESCRIPTION_WEIGHT = 0.3

    # Task type → category mapping
    TASK_CATEGORIES = {
        "code_generation": ["code_generation", "scaffolding", "codegen"],
        "documentation": ["documentation", "docs"],
        "testing": ["testing", "qa"],
        "analysis": ["analysis", "research"],
        "writing": ["writing", "content"],
        "planning": ["planning", "strategy"],
        "debugging": ["debugging", "code_review"],
        "deployment": ["deployment", "devops"],
    }

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        tracker: Optional[SkillTracker] = None,
    ):
        self._registry = registry or get_registry()
        self._tracker = tracker

    def match(self, task: str, task_type: str = "") -> List[SkillMatch]:
        """Find skills that match a task description.

        Args:
            task: The task description / user prompt
            task_type: Optional task type (e.g. "code_generation")

        Returns:
            List of SkillMatch sorted by score (highest first)
        """
        all_skills = self._registry.list_skills()
        if not all_skills:
            return []

        matches: List[SkillMatch] = []
        task_lower = task.lower()
        task_words = set(re.findall(r"\w+", task_lower))

        # Get target categories from task_type
        target_categories = self.TASK_CATEGORIES.get(task_type, [])

        for skill in all_skills:
            score = 0.0
            matched_on = ""

            # 1. Trigger matching (highest weight)
            for trigger in skill.trigger:
                if trigger.lower() in task_lower:
                    score += self.TRIGGER_WEIGHT
                    matched_on = "trigger"

            # 2. Category matching
            if target_categories:
                for cat in skill.categories:
                    if cat in target_categories:
                        score += self.CATEGORY_WEIGHT
                        if not matched_on:
                            matched_on = "category"

            # 3. Tag overlap
            skill_tags = set(t.lower() for t in skill.tags)
            tag_overlap = skill_tags & task_words
            if tag_overlap:
                score += self.TAG_WEIGHT * len(tag_overlap)
                if not matched_on:
                    matched_on = "tag"

            # 4. Description keyword overlap
            desc_words = set(re.findall(r"\w+", skill.description.lower()))
            desc_overlap = desc_words & task_words
            if desc_overlap:
                score += self.DESCRIPTION_WEIGHT * min(len(desc_overlap), 3)
                if not matched_on:
                    matched_on = "description"

            if score > 0:
                # Boost by success rate if tracker available
                if self._tracker:
                    stats = self._tracker.get_stats(skill.name)
                    if stats.total_uses > 0:
                        score *= (0.5 + stats.success_rate * 0.5)

                matches.append(SkillMatch(
                    skill=skill,
                    score=round(score, 3),
                    matched_on=matched_on,
                ))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def best_match(self, task: str, task_type: str = "") -> Optional[SkillMatch]:
        """Return the single best matching skill, or None."""
        matches = self.match(task, task_type)
        return matches[0] if matches else None

    def suggest_skills(
        self,
        task: str,
        task_type: str = "",
        limit: int = 3,
    ) -> List[SkillMatch]:
        """Suggest up to *limit* skills for a task.

        If no skills match, returns empty list.
        """
        matches = self.match(task, task_type)
        return matches[:limit]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_matcher: Optional[SkillMatcher] = None


def get_matcher(
    registry: Optional[SkillRegistry] = None,
    tracker: Optional[SkillTracker] = None,
) -> SkillMatcher:
    """Get or create the default SkillMatcher."""
    global _matcher
    if _matcher is None or registry is not None or tracker is not None:
        _matcher = SkillMatcher(registry=registry, tracker=tracker)
    return _matcher


def reset_matcher() -> None:
    """Reset the singleton."""
    global _matcher
    _matcher = None
