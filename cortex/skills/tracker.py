"""
cortex/skills/tracker.py — Telemetry for skill usage.

Tracks when skills are used, whether they succeeded, how long they took,
and provides analytics for pruning stale skills and ranking by popularity.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import DEFAULT_SKILLS_DIR

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_DIR = Path.home() / ".odysseus" / "skill_stats"
STALE_THRESHOLD_DAYS = 30


@dataclass
class SkillUseRecord:
    """A single use of a skill."""

    skill_name: str
    timestamp: str
    success: bool
    duration_ms: float = 0.0
    task: str = ""
    error: str = ""
    user_corrected: bool = False


@dataclass
class SkillStats:
    """Aggregate statistics for a skill."""

    name: str
    total_uses: int = 0
    successes: int = 0
    failures: int = 0
    last_used: str = ""
    avg_duration_ms: float = 0.0
    corrections: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if self.total_uses == 0:
            return 0.0
        return self.successes / self.total_uses

    @property
    def is_stale(self) -> bool:
        """Whether this skill hasn't been used recently."""
        if not self.last_used:
            return True
        try:
            last = datetime.fromisoformat(self.last_used.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age = (now - last).days
            return age > STALE_THRESHOLD_DAYS
        except (ValueError, TypeError):
            return True


class SkillTracker:
    """Tracks skill usage over time.

    Records are persisted as JSONL files under ~/.odysseus/skill_stats/.
    """

    def __init__(self, tracking_dir: Optional[Path | str] = None):
        self.tracking_dir = Path(tracking_dir or DEFAULT_TRACKING_DIR)
        self._records: List[SkillUseRecord] = []

    def _log_file(self) -> Path:
        """Return the current day's log file path."""
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.tracking_dir / f"usage-{today}.jsonl"

    def log_use(
        self,
        skill_name: str,
        success: bool,
        duration_ms: float = 0.0,
        task: str = "",
        error: str = "",
        user_corrected: bool = False,
    ) -> SkillUseRecord:
        """Record a single skill use."""
        record = SkillUseRecord(
            skill_name=skill_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=success,
            duration_ms=duration_ms,
            task=task[:200],
            error=error[:500],
            user_corrected=user_corrected,
        )

        line = json.dumps(asdict(record), ensure_ascii=False)
        path = self._log_file()
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

        self._records.append(record)
        logger.debug(
            "Logged skill use: %s (success=%s)", skill_name, success
        )
        return record

    def _load_all_records(self) -> List[SkillUseRecord]:
        """Load all records from all log files."""
        records: List[SkillUseRecord] = []
        if not self.tracking_dir.exists():
            return records

        for f in sorted(self.tracking_dir.glob("usage-*.jsonl")):
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    records.append(SkillUseRecord(**data))
            except (json.JSONDecodeError, TypeError, OSError) as exc:
                logger.warning("Skipping malformed tracking file %s: %s", f, exc)

        return records

    def get_stats(self, skill_name: str) -> SkillStats:
        """Get aggregate statistics for a skill."""
        records = self._load_all_records()
        relevant = [r for r in records if r.skill_name == skill_name]

        stats = SkillStats(name=skill_name)
        stats.total_uses = len(relevant)
        stats.successes = sum(1 for r in relevant if r.success)
        stats.failures = sum(1 for r in relevant if not r.success)
        stats.corrections = sum(1 for r in relevant if r.user_corrected)

        if relevant:
            durations = [r.duration_ms for r in relevant if r.duration_ms > 0]
            if durations:
                stats.avg_duration_ms = sum(durations) / len(durations)
            sorted_records = sorted(relevant, key=lambda r: r.timestamp)
            stats.last_used = sorted_records[-1].timestamp

        return stats

    def get_all_stats(self) -> List[SkillStats]:
        """Get statistics for all tracked skills."""
        records = self._load_all_records()
        names = set(r.skill_name for r in records)
        return [self.get_stats(name) for name in sorted(names)]

    def stale_skills(self) -> List[str]:
        """Return names of skills that haven't been used recently."""
        all_stats = self.get_all_stats()
        return [s.name for s in all_stats if s.is_stale]

    def most_used(self, limit: int = 10) -> List[SkillStats]:
        """Return the most-used skills by total use count."""
        all_stats = self.get_all_stats()
        all_stats.sort(key=lambda s: s.total_uses, reverse=True)
        return all_stats[:limit]

    def clear_stats(self) -> int:
        """Delete all tracking files.

        Returns the number of files deleted.
        """
        if not self.tracking_dir.exists():
            return 0
        count = 0
        for f in self.tracking_dir.glob("usage-*.jsonl"):
            f.unlink()
            count += 1
        self._records.clear()
        return count
