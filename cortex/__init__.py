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
from . import router as cortex_router
from . import improve
from . import skills_sh
from . import cli_invoker
from . import smart_router
from . import csuite
from . import sdd

__all__ = ["goals", "cortex_router", "improve", "skills_sh", "cli_invoker",
           "smart_router", "csuite", "sdd"]
