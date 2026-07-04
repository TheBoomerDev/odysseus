"""
cortex/skills/updater.py — Auto-update skills from user feedback.

When a user corrects an output that was generated using a skill, the
updater refines the skill's body and trigger phrases to incorporate
the correction. Supports version bumping and diff-based updates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .registry import Skill, SkillRegistry, get_registry
from .tracker import SkillTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feedback types
# ---------------------------------------------------------------------------


@dataclass
class SkillFeedback:
    """Feedback from a skill usage."""

    skill_name: str
    original_output: str
    corrected_output: str
    task: str = ""
    feedback_type: str = "correction"  # "correction", "refinement", "approval"


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------


class SkillUpdater:
    """Updates skills based on user feedback.

    Usage:
        updater = SkillUpdater()
        updater.apply_feedback(SkillFeedback(
            skill_name="codegen-landing",
            original_output="...",
            corrected_output="...",
        ))
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        tracker: Optional[SkillTracker] = None,
    ):
        self._registry = registry or get_registry()
        self._tracker = tracker

    def apply_feedback(self, feedback: SkillFeedback) -> Optional[Skill]:
        """Apply user feedback to a skill.

        Updates the skill body with the correction and bumps version.
        Returns the updated skill, or None if the skill doesn't exist.
        """
        skill = self._registry.get_skill(feedback.skill_name)
        if skill is None:
            logger.warning(
                "Cannot apply feedback: skill '%s' not found",
                feedback.skill_name,
            )
            return None

        if feedback.feedback_type == "approval":
            # Just log the approval, no update needed
            if self._tracker:
                self._tracker.log_use(
                    skill_name=feedback.skill_name,
                    success=True,
                    task=feedback.task,
                )
            return skill

        # Extract the diff between original and corrected
        diff = self._compute_diff(
            feedback.original_output,
            feedback.corrected_output,
        )

        if not diff:
            logger.info("No meaningful diff in feedback for '%s'", feedback.skill_name)
            return skill

        # Append the correction as a "Lessons Learned" section
        updated_body = self._inject_lesson(skill.body, diff, feedback.task)

        # Version bump
        new_version = self._bump_version(skill.version)

        updated_skill = Skill(
            name=skill.name,
            description=skill.description,
            version=new_version,
            author=skill.author,
            tags=skill.tags,
            categories=skill.categories,
            trigger=skill.trigger,
            requires=skill.requires,
            body=updated_body,
            raw_frontmatter=skill.raw_frontmatter,
        )

        self._registry.update_skill(updated_skill)
        logger.info(
            "Updated skill '%s' to v%s based on feedback",
            feedback.skill_name, new_version,
        )

        # Log the correction in tracker
        if self._tracker:
            self._tracker.log_use(
                skill_name=feedback.skill_name,
                success=False,
                task=feedback.task,
                user_corrected=True,
            )

        return updated_skill

    def merge_multiple_feedback(
        self,
        feedbacks: List[SkillFeedback],
    ) -> Optional[Skill]:
        """Apply multiple feedback entries to a skill at once.

        More efficient than calling apply_feedback() multiple times
        because it batches the lessons learned.
        """
        if not feedbacks:
            return None

        skill_name = feedbacks[0].skill_name
        skill = self._registry.get_skill(skill_name)
        if skill is None:
            return None

        lessons: List[str] = []
        for fb in feedbacks:
            diff = self._compute_diff(fb.original_output, fb.corrected_output)
            if diff:
                lessons.append(f"- Task: {fb.task[:80]}\n  Correction: {diff}")

        if not lessons:
            return skill

        updated_body = self._inject_lessons_batch(skill.body, lessons)
        new_version = self._bump_version(skill.version)

        updated_skill = Skill(
            name=skill.name,
            description=skill.description,
            version=new_version,
            author=skill.author,
            tags=skill.tags,
            categories=skill.categories,
            trigger=skill.trigger,
            requires=skill.requires,
            body=updated_body,
            raw_frontmatter=skill.raw_frontmatter,
        )

        self._registry.update_skill(updated_skill)
        return updated_skill

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    @staticmethod
    def _bump_version(version: str, bump_type: str = "patch") -> str:
        """Bump a semver version string.

        Args:
            version: Version string like "1.2.3"
            bump_type: "patch", "minor", or "major"
        """
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
        if not match:
            return "0.1.0"

        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1

        return f"{major}.{minor}.{patch}"

    # ------------------------------------------------------------------
    # Diff computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diff(original: str, corrected: str) -> str:
        """Compute a human-readable diff between two outputs.

        Uses a simple line-based diff to identify what changed.
        Returns a summary string, or empty string if no meaningful diff.
        """
        if original == corrected:
            return ""

        orig_lines = original.strip().splitlines()
        corr_lines = corrected.strip().splitlines()

        changes: List[str] = []

        # Find added lines (in corrected but not in original)
        orig_set = set(l.strip() for l in orig_lines if l.strip())
        corr_set = set(l.strip() for l in corr_lines if l.strip())

        added = corr_set - orig_set
        removed = orig_set - corr_set

        if added:
            changes.append("Added: " + "; ".join(list(added)[:3]))
        if removed:
            changes.append("Removed: " + "; ".join(list(removed)[:3]))

        return " | ".join(changes) if changes else ""

    # ------------------------------------------------------------------
    # Body injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_lesson(body: str, lesson: str, task: str = "") -> str:
        """Inject a lesson learned into the skill body."""
        now = datetime.now(timezone.utc).isoformat()
        lesson_block = f"""

## Lessons Learned

### {now}
- Task: {task[:100]}
- Correction: {lesson}
"""

        # Check if "Lessons Learned" section already exists
        if "## Lessons Learned" in body:
            # Append to existing section
            return body + f"\n### {now}\n- Task: {task[:100]}\n- Correction: {lesson}\n"
        else:
            return body + lesson_block

    @staticmethod
    def _inject_lessons_batch(body: str, lessons: List[str]) -> str:
        """Inject multiple lessons at once."""
        now = datetime.now(timezone.utc).isoformat()
        block = f"\n\n## Lessons Learned (batch {now})\n\n" + "\n".join(lessons)

        if "## Lessons Learned" in body:
            return body + "\n" + "\n".join(lessons) + "\n"
        else:
            return body + block
