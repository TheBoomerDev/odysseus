"""
cortex — CORTEX capabilities for Odysseus.

This module ports the unique CORTEX features into the Odysseus fork:

    goals/       — Goal decomposition engine (templates + heuristic)
    router/      — Multi-agent routing/scorer
    improve/     — Self-improvement loop (trajectories, patterns, SkillOpt)
    skills_sh/   — Bridge to skills.sh marketplace
    cli_invoker/ — CLI agent orchestrator

Each submodule exposes a public API that can be used directly from
Python or via the `/api/cortex/*` FastAPI routes in routes/cortex_routes.py.
"""

from . import goals
from . import router
from . import improve
from . import skills_sh
from . import cli_invoker

__all__ = ["goals", "router", "improve", "skills_sh", "cli_invoker"]
