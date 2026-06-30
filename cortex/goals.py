"""
cortex/goals.py — Goal decomposition engine.

Decomposes high-level goals into structured subtasks using:
1. Template matching (fast path for common patterns)
2. Heuristic decomposition (fallback)

Port of CORTEX goals/engine.ts to Python, adapted for Odysseus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SubTask:
    id: str
    description: str
    agent: str
    capabilities: List[str]
    depends_on: List[str]
    optional: bool = False
    estimated_cost_usd: float = 0.05
    estimated_duration_seconds: int = 60


@dataclass
class GoalDecomposition:
    goal: str
    tasks: List[SubTask]
    total_estimated_cost_usd: float = 0.0
    total_estimated_duration_seconds: int = 0
    source: str = "heuristic"  # "template" | "heuristic"
    template_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_TEMPLATES: List[dict] = [
    {
        "id": "migrate-database",
        "match": lambda g: bool(re.search(r"migrate.*(database|db|mongo|postgres|sql)", g, re.I)),
        "tasks": [
            {"description": "Research breaking changes and migration guide", "agent": "research",
             "capabilities": ["web_search", "doc_synthesis"], "depends_on": [], "optional": False},
            {"description": "Design new schema mapping", "agent": "architect",
             "capabilities": ["architecture_design", "schema_design"], "depends_on": ["t1"], "optional": False},
            {"description": "Generate migration scripts", "agent": "codex",
             "capabilities": ["code_edit", "multi_file_diff"], "depends_on": ["t2"], "optional": False},
            {"description": "Write tests for new schema", "agent": "codex",
             "capabilities": ["test_generation"], "depends_on": ["t3"], "optional": False},
            {"description": "Validate — run tests, adversarial review", "agent": "verifier",
             "capabilities": ["test_execution", "adversarial_review"], "depends_on": ["t4"], "optional": False},
            {"description": "Document migration in README/changelog", "agent": "architect",
             "capabilities": ["doc_generation"], "depends_on": ["t5"], "optional": True},
        ],
    },
    {
        "id": "audit-security",
        "match": lambda g: bool(re.search(r"audit|security|vulnerabilit|cve", g, re.I)),
        "tasks": [
            {"description": "Scan dependencies for known CVEs", "agent": "research",
             "capabilities": ["web_search"], "depends_on": [], "optional": False},
            {"description": "Static analysis: OWASP top 10 patterns", "agent": "codex",
             "capabilities": ["code_review", "adversarial_review"], "depends_on": [], "optional": False},
            {"description": "Secrets scanning (hardcoded keys, tokens)", "agent": "research",
             "capabilities": ["code_review"], "depends_on": [], "optional": False},
            {"description": "Synthesize findings into prioritized report", "agent": "architect",
             "capabilities": ["doc_generation", "trade_off_analysis"],
             "depends_on": ["t1", "t2", "t3"], "optional": False},
        ],
    },
    {
        "id": "refactor-module",
        "match": lambda g: bool(re.search(r"refactor|split|divide|modular", g, re.I)),
        "tasks": [
            {"description": "Analyze current module structure", "agent": "research",
             "capabilities": ["code_review"], "depends_on": [], "optional": False},
            {"description": "Design target module boundaries", "agent": "architect",
             "capabilities": ["architecture_design"], "depends_on": ["t1"], "optional": False},
            {"description": "Apply refactor — split files, update imports", "agent": "codex",
             "capabilities": ["code_edit", "multi_file_diff"], "depends_on": ["t2"], "optional": False},
            {"description": "Verify tests still pass", "agent": "verifier",
             "capabilities": ["test_execution"], "depends_on": ["t3"], "optional": False},
        ],
    },
]


def _make_template_tasks(template: dict) -> List[SubTask]:
    """Build SubTask list from a template dict."""
    tasks: List[SubTask] = []
    for idx, t in enumerate(template["tasks"]):
        tasks.append(SubTask(
            id=f"t{idx + 1}",
            description=t["description"],
            agent=t["agent"],
            capabilities=t["capabilities"],
            depends_on=list(t.get("depends_on", [])),
            optional=t.get("optional", False),
            estimated_cost_usd=0.05,
            estimated_duration_seconds=60,
        ))
    return tasks


def _heuristic_tasks(goal: str) -> List[SubTask]:
    """Fallback decomposition for unmatched goals."""
    return [
        SubTask(id="t1-research", description=f'Research: gather context for "{goal[:80]}"',
                agent="research", capabilities=["web_search", "doc_synthesis"],
                depends_on=[], estimated_cost_usd=0.05, estimated_duration_seconds=60),
        SubTask(id="t2-design", description="Design approach and trade-offs",
                agent="architect", capabilities=["architecture_design"],
                depends_on=["t1-research"], estimated_cost_usd=0.08, estimated_duration_seconds=90),
        SubTask(id="t3-implement", description="Implement changes",
                agent="codex", capabilities=["code_edit", "multi_file_diff"],
                depends_on=["t2-design"], estimated_cost_usd=0.15, estimated_duration_seconds=120),
        SubTask(id="t4-validate", description="Run tests and review",
                agent="verifier", capabilities=["test_execution", "code_review"],
                depends_on=["t3-implement"], estimated_cost_usd=0.07, estimated_duration_seconds=60),
    ]


def decompose(goal: str) -> GoalDecomposition:
    """Decompose a high-level goal into a structured plan."""
    if not goal or not goal.strip():
        raise ValueError("goal cannot be empty")
    if len(goal) > 5000:
        raise ValueError(f"goal too long ({len(goal)} chars, max 5000)")

    for tmpl in _TEMPLATES:
        if tmpl["match"](goal):
            tasks = _make_template_tasks(tmpl)
            total_cost = sum(t.estimated_cost_usd for t in tasks)
            total_dur = sum(t.estimated_duration_seconds for t in tasks)
            return GoalDecomposition(
                goal=goal,
                tasks=tasks,
                total_estimated_cost_usd=round(total_cost, 4),
                total_estimated_duration_seconds=total_dur,
                source="template",
                template_id=tmpl["id"],
            )

    # Heuristic fallback
    tasks = _heuristic_tasks(goal)
    total_cost = sum(t.estimated_cost_usd for t in tasks)
    total_dur = sum(t.estimated_duration_seconds for t in tasks)
    return GoalDecomposition(
        goal=goal,
        tasks=tasks,
        total_estimated_cost_usd=round(total_cost, 4),
        total_estimated_duration_seconds=total_dur,
        source="heuristic",
    )


def render_tree(decomp: GoalDecomposition) -> str:
    """Render a decomposition as a text tree."""
    lines = [
        f"Goal: {decomp.goal}",
        f"Source: {decomp.source}{f' ({decomp.template_id})' if decomp.template_id else ''}",
        f"Total: ~${decomp.total_estimated_cost_usd:.3f} · ~{decomp.total_estimated_duration_seconds}s",
        "",
    ]
    for task in decomp.tasks:
        deps = f" (depends on {', '.join(task.depends_on)})" if task.depends_on else ""
        opt = " [optional]" if task.optional else ""
        lines.append(f"  ├─ {task.id}: {task.description} [{task.agent}] ~${task.estimated_cost_usd:.3f}{deps}{opt}")
    return "\n".join(lines)
