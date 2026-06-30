"""Tests for cortex/improve.py — trajectory recording and pattern detection.

Tests: Trajectory, DetectedPattern, record_trajectory, load_all_trajectories,
       detect_patterns, load_patterns, get_improvement_stats.
"""

import json
import os
import pytest
import tempfile


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_create(self):
        """Creates a Trajectory with required fields."""
        from cortex.improve import Trajectory

        t = Trajectory(
            session_id="s1", agent="codex", prompt="fix bug", status="success"
        )
        assert t.session_id == "s1"
        assert t.agent == "codex"
        assert t.status == "success"
        assert t.exit_code is None
        assert t.files_changed == []

    def test_with_all_fields(self):
        """Creates a Trajectory with all fields."""
        from cortex.improve import Trajectory

        t = Trajectory(
            session_id="s2",
            agent="claude",
            prompt="refactor module",
            status="error",
            exit_code=1,
            duration_seconds=30.5,
            files_changed=["main.py"],
            stdout="output",
            stderr="error msg",
            errors=[{"message": "timeout"}],
            timestamp="2026-06-30T12:00:00",
        )
        assert t.exit_code == 1
        assert t.duration_seconds == 30.5
        assert len(t.errors) == 1


# ---------------------------------------------------------------------------
# DetectedPattern dataclass
# ---------------------------------------------------------------------------


class TestDetectedPattern:
    """Tests for DetectedPattern dataclass."""

    def test_create(self):
        """Creates a DetectedPattern."""
        from cortex.improve import DetectedPattern

        p = DetectedPattern(
            id="p1",
            description="repeated commits",
            occurrence_count=5,
            example_prompts=["commit changes"],
            suggested_skill_name="git-commit",
            suggested_skill_content="## Steps\n1. commit",
            estimated_tokens_saved=2000,
        )
        assert p.id == "p1"
        assert p.occurrence_count == 5
        assert p.confidence == 0.0  # default


# ---------------------------------------------------------------------------
# Trajectory storage
# ---------------------------------------------------------------------------


class TestTrajectoryStorage:
    """Tests for record_trajectory and load_all_trajectories."""

    def test_record_and_load(self, monkeypatch, tmp_path):
        """Records a trajectory and loads it back."""
        from cortex.improve import (
            record_trajectory,
            load_all_trajectories,
            Trajectory,
            TRAJECTORIES_DIR,
        )

        monkeypatch.setattr("cortex.improve.TRAJECTORIES_DIR", str(tmp_path / "trajs"))
        monkeypatch.setattr("cortex.improve.PATTERNS_FILE", str(tmp_path / "patterns.json"))

        t = Trajectory(
            session_id="test-rec-1",
            agent="codex",
            prompt="do something",
            status="success",
        )
        record_trajectory(t)

        loaded = load_all_trajectories()
        assert len(loaded) == 1
        assert loaded[0].session_id == "test-rec-1"
        assert loaded[0].status == "success"

    def test_load_empty(self, monkeypatch, tmp_path):
        """Load from empty directory returns empty list."""
        from cortex.improve import load_all_trajectories

        monkeypatch.setattr("cortex.improve.TRAJECTORIES_DIR", str(tmp_path / "empty"))
        result = load_all_trajectories()
        assert result == []

    def test_load_corrupted_file(self, monkeypatch, tmp_path):
        """Corrupted trajectory file is skipped gracefully."""
        from cortex.improve import (
            load_all_trajectories,
            TRAJECTORIES_DIR,
        )

        traj_dir = tmp_path / "trajs"
        traj_dir.mkdir(parents=True)
        monkeypatch.setattr("cortex.improve.TRAJECTORIES_DIR", str(traj_dir))

        # Write a corrupted file
        with open(os.path.join(traj_dir, "corrupt.json"), "w") as f:
            f.write("not-json")

        result = load_all_trajectories()
        assert result == []


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    """Tests for detect_patterns function."""

    def test_detect_commit_pattern(self, monkeypatch, tmp_path):
        """Detects git commit patterns from 3+ matching trajectories."""
        from cortex.improve import (
            detect_patterns,
            Trajectory,
            PATTERNS_FILE,
        )

        monkeypatch.setattr("cortex.improve.PATTERNS_FILE", str(tmp_path / "patterns.json"))

        trajs = [
            Trajectory(session_id=f"s{i}", agent="cli", prompt="git commit -m 'fix'", status="success")
            for i in range(5)
        ]
        patterns = detect_patterns(trajs)
        assert len(patterns) >= 1
        commit_patterns = [p for p in patterns if "commit" in p.id]
        assert len(commit_patterns) >= 1

    def test_below_threshold(self, monkeypatch, tmp_path):
        """Fewer than 3 matches returns no patterns."""
        from cortex.improve import (
            detect_patterns,
            Trajectory,
            PATTERNS_FILE,
        )

        monkeypatch.setattr("cortex.improve.PATTERNS_FILE", str(tmp_path / "patterns.json"))

        trajs = [
            Trajectory(session_id="s1", agent="cli", prompt="git commit -m 'fix'", status="success"),
        ]
        patterns = detect_patterns(trajs)
        assert len(patterns) == 0

    def test_no_patterns_for_random(self, monkeypatch, tmp_path):
        """Random trajectories don't match patterns."""
        from cortex.improve import (
            detect_patterns,
            Trajectory,
            PATTERNS_FILE,
        )

        monkeypatch.setattr("cortex.improve.PATTERNS_FILE", str(tmp_path / "patterns.json"))

        trajs = [
            Trajectory(session_id=f"s{i}", agent="cli", prompt="random task number {i}", status="success")
            for i in range(10)
        ]
        patterns = detect_patterns(trajs)
        assert len(patterns) == 0

    def test_saves_patterns_to_disk(self, monkeypatch, tmp_path):
        """Patterns are saved to disk after detection."""
        from cortex.improve import (
            detect_patterns,
            Trajectory,
            PATTERNS_FILE,
        )

        monkeypatch.setattr("cortex.improve.PATTERNS_FILE", str(tmp_path / "patterns.json"))

        trajs = [
            Trajectory(session_id=f"s{i}", agent="cli", prompt="pip install flask", status="success")
            for i in range(4)
        ]
        detect_patterns(trajs)

        assert os.path.exists(tmp_path / "patterns.json")
        with open(tmp_path / "patterns.json") as f:
            data = json.load(f)
        assert len(data) >= 1


class TestLoadPatterns:
    """Tests for load_patterns function."""

    def test_load_existing(self, monkeypatch, tmp_path):
        """Loads saved patterns from disk."""
        from cortex.improve import load_patterns, PATTERNS_FILE

        patterns_path = tmp_path / "patterns.json"
        monkeypatch.setattr("cortex.improve.PATTERNS_FILE", str(patterns_path))

        # Write test patterns
        with open(patterns_path, "w") as f:
            json.dump([{
                "id": "test-pattern",
                "description": "test",
                "occurrence_count": 3,
                "example_prompts": ["test"],
                "suggested_skill_name": "test-skill",
                "suggested_skill_content": "test",
                "estimated_tokens_saved": 100,
                "confidence": 0.5,
            }], f)

        patterns = load_patterns()
        assert len(patterns) == 1
        assert patterns[0].id == "test-pattern"

    def test_load_missing(self, monkeypatch, tmp_path):
        """No patterns file returns empty list."""
        from cortex.improve import load_patterns, PATTERNS_FILE

        monkeypatch.setattr("cortex.improve.PATTERNS_FILE", str(tmp_path / "nonexistent.json"))
        result = load_patterns()
        assert result == []


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestGetImprovementStats:
    """Tests for get_improvement_stats."""

    def test_stats_empty(self, monkeypatch, tmp_path):
        """Empty system returns zero stats."""
        from cortex.improve import (
            get_improvement_stats,
            TRAJECTORIES_DIR,
            PATTERNS_FILE,
        )

        monkeypatch.setattr("cortex.improve.TRAJECTORIES_DIR", str(tmp_path / "trajs"))
        monkeypatch.setattr("cortex.improve.PATTERNS_FILE", str(tmp_path / "patterns.json"))

        stats = get_improvement_stats()
        assert stats["total_trajectories"] == 0
        assert stats["total_patterns"] == 0
        assert stats["tokens_saved"] == 0
        assert stats["success_rate"] == 0
