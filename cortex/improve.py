"""
cortex/improve.py — Self-improvement loop.

Tracks trajectories of agent invocations, detects patterns,
and generates SkillOpt proposals to improve the system over time.

Port of CORTEX improve/ modules to Python.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    session_id: str
    agent: str
    prompt: str
    status: str  # success | error | timeout | fallback_used
    exit_code: Optional[int] = None
    duration_seconds: float = 0.0
    files_changed: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    errors: List[Dict[str, str]] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class DetectedPattern:
    id: str
    description: str
    occurrence_count: int
    example_prompts: List[str]
    suggested_skill_name: str
    suggested_skill_content: str
    estimated_tokens_saved: int
    confidence: float = 0.0


TRAJECTORIES_DIR = "data/cortex/trajectories"
PATTERNS_FILE = "data/cortex/patterns.json"


# ---------------------------------------------------------------------------
# Trajectory storage
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    Path(TRAJECTORIES_DIR).mkdir(parents=True, exist_ok=True)
    Path("data/cortex").mkdir(parents=True, exist_ok=True)


def record_trajectory(traj: Trajectory) -> None:
    """Persist a trajectory to disk."""
    _ensure_dirs()
    path = os.path.join(TRAJECTORIES_DIR, f"{traj.session_id}.json")
    with open(path, "w") as f:
        json.dump(asdict(traj), f, indent=2)
    logger.debug("recorded trajectory %s", traj.session_id)


def load_all_trajectories() -> List[Trajectory]:
    """Load all stored trajectories."""
    _ensure_dirs()
    trajs: List[Trajectory] = []
    if not os.path.isdir(TRAJECTORIES_DIR):
        return trajs
    for fname in sorted(os.listdir(TRAJECTORIES_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(TRAJECTORIES_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            trajs.append(Trajectory(**data))
        except Exception as e:
            logger.warning("failed to load trajectory %s: %s", fname, e)
    return trajs


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

_COMMON_PATTERNS: List[Dict[str, Any]] = [
    {
        "id": "config-change",
        "match": lambda t: (
            "change" in t.prompt.lower()
            or "update" in t.prompt.lower()
            or "config" in t.prompt.lower()
        ),
        "description": "Repeated configuration changes",
        "skill_name": "config-update",
        "skill_content": """## Procedure
1. Run `config status` to see current config
2. Make targeted changes with specific commands
3. Verify with `config validate`
""",
        "tokens_saved": 500,
    },
    {
        "id": "dependency-install",
        "match": lambda t: (
            "install" in t.prompt.lower()
            and (
                "pip" in t.prompt.lower()
                or "npm" in t.prompt.lower()
                or "apt" in t.prompt.lower()
            )
        ),
        "description": "Repeated package installations",
        "skill_name": "install-package",
        "skill_content": """## Procedure
1. Check if package already installed
2. Install with lockfile update
3. Pin version in requirements
""",
        "tokens_saved": 300,
    },
    {
        "id": "git-commit",
        "match": lambda t: "commit" in t.prompt.lower() and "git" in t.prompt.lower(),
        "description": "Repeated git commit patterns",
        "skill_name": "git-commit-conventional",
        "skill_content": """## Procedure
1. Stage relevant files with `git add`
2. Commit with conventional format: `type(scope): description`
3. Verify commit with `git log --oneline -1`
""",
        "tokens_saved": 400,
    },
]


def detect_patterns(trajs: Optional[List[Trajectory]] = None) -> List[DetectedPattern]:
    """Detect repeated patterns from trajectories."""
    if trajs is None:
        trajs = load_all_trajectories()

    patterns: List[DetectedPattern] = []
    for pat_def in _COMMON_PATTERNS:
        matches = [t for t in trajs if pat_def["match"](t)]
        if len(matches) >= 3:  # minimum 3 occurrences
            patterns.append(
                DetectedPattern(
                    id=pat_def["id"],
                    description=pat_def["description"],
                    occurrence_count=len(matches),
                    example_prompts=[m.prompt[:100] for m in matches[:3]],
                    suggested_skill_name=pat_def["skill_name"],
                    suggested_skill_content=pat_def["skill_content"],
                    estimated_tokens_saved=pat_def["tokens_saved"] * len(matches),
                    confidence=min(1.0, len(matches) / 10),
                )
            )

    # Save patterns
    _ensure_dirs()
    with open(PATTERNS_FILE, "w") as f:
        json.dump([asdict(p) for p in patterns], f, indent=2)

    return patterns


def load_patterns() -> List[DetectedPattern]:
    """Load detected patterns from disk."""
    _ensure_dirs()
    if not os.path.exists(PATTERNS_FILE):
        return []
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    return [DetectedPattern(**d) for d in data]


def get_improvement_stats() -> Dict:
    """Get stats about the self-improvement system."""
    trajs = load_all_trajectories()
    patterns = load_patterns()
    return {
        "total_trajectories": len(trajs),
        "total_patterns": len(patterns),
        "tokens_saved": sum(p.estimated_tokens_saved for p in patterns),
        "success_rate": (
            sum(1 for t in trajs if t.status == "success") / len(trajs) * 100
            if trajs
            else 0
        ),
        "agents_used": len(set(t.agent for t in trajs)),
    }
