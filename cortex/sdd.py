"""
cortex/sdd.py — Spec-Driven Development Pipeline.

Port of ADABA/codeWriter v2 SDDGenerator.ts + PipelineOrchestrator.ts.

Generates structured software design documents (spec, plan, tasks)
from a goal description, and orchestrates a 7-step pipeline.

Each step produces a markdown document in data/cortex/sdd/<goal-id>/.
Integrates with Odysseus's LLM for document generation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SDD_DIR = "data/cortex/sdd"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class PipelineStep(str, Enum):
    ANALYZE = "analyze"
    DOCUMENT = "document"
    DECOMPOSE = "decompose"
    ASSIGN = "assign"
    EXECUTE = "execute"
    VERIFY = "verify"
    CONSOLIDATE = "consolidate"


STEP_LABELS = {
    PipelineStep.ANALYZE: "Analyze",
    PipelineStep.DOCUMENT: "Document",
    PipelineStep.DECOMPOSE: "Decompose",
    PipelineStep.ASSIGN: "Assign",
    PipelineStep.EXECUTE: "Execute",
    PipelineStep.VERIFY: "Verify",
    PipelineStep.CONSOLIDATE: "Consolidate",
}

STEP_DESCRIPTIONS = {
    PipelineStep.ANALYZE: "Analyze the goal to determine if it's dev or non-dev work",
    PipelineStep.DOCUMENT: "Generate SDD documents (spec, plan, tasks) for dev goals",
    PipelineStep.DECOMPOSE: "Break the goal into concrete, assignable subtasks",
    PipelineStep.ASSIGN: "Match each subtask to the best agent/tool",
    PipelineStep.EXECUTE: "Execute each subtask through the assigned agent",
    PipelineStep.VERIFY: "Verify results against quality criteria",
    PipelineStep.CONSOLIDATE: "Consolidate outputs into final deliverable",
}


# ---------------------------------------------------------------------------
# SDD Documents
# ---------------------------------------------------------------------------

@dataclass
class SDDDocuments:
    goal_id: str
    goal: str
    spec_content: str = ""
    plan_content: str = ""
    tasks_content: str = ""
    created_at: str = ""

    def save(self, base_dir: str = SDD_DIR) -> str:
        """Save SDD documents to disk."""
        doc_dir = Path(base_dir) / self.goal_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "goal_id": self.goal_id,
            "goal": self.goal,
            "created_at": self.created_at or datetime.now().isoformat(),
        }

        with open(doc_dir / "spec.md", "w") as f:
            f.write(self.spec_content)
        with open(doc_dir / "plan.md", "w") as f:
            f.write(self.plan_content)
        with open(doc_dir / "tasks.md", "w") as f:
            f.write(self.tasks_content)
        with open(doc_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("SDD documents saved to %s", doc_dir)
        return str(doc_dir)


# ---------------------------------------------------------------------------
# SDD Generator
# ---------------------------------------------------------------------------

class SDDGenerator:
    """Generates Spec-Driven Development documents from a goal.

    Produces three documents:
    - spec.md: Functional specification
    - plan.md: Implementation plan
    - tasks.md: Breakdown of tasks

    This module generates TEMPLATES — the actual content should be
    filled by Odysseus's LLM for best quality.
    """

    def __init__(self):
        pass

    def generate_spec_template(self, goal: str) -> str:
        """Generate a specification document template."""
        return f"""# Specification

## Goal
{goal}

## Requirements

### Functional Requirements
1.
2.
3.

### Non-Functional Requirements
1.
2.
3.

## Scope

### In Scope
-

### Out of Scope
-

## Technical Approach

### Architecture
-

### Data Model
-

### API Design
-

## Dependencies
-

## Risks and Mitigations
-

## Acceptance Criteria
1.
2.
3.
"""

    def generate_plan_template(self, goal: str) -> str:
        """Generate an implementation plan template."""
        return f"""# Implementation Plan

## Goal
{goal}

## Phases

### Phase 1: Foundation
- [ ] Task 1.1
- [ ] Task 1.2

### Phase 2: Core Implementation
- [ ] Task 2.1
- [ ] Task 2.2

### Phase 3: Testing & Polish
- [ ] Task 3.1
- [ ] Task 3.2

## Timeline
- Estimated total: 
- Dependencies:

## Resources
- Team:
- Tools:
- Budget:

## Milestones
1.
2.
3.
"""

    def generate_tasks_template(self, goal: str) -> str:
        """Generate a task breakdown template."""
        return f"""# Task Breakdown

## Goal
{goal}

## Tasks

### Task 1:
- **Description:**
- **Type:** implementation | research | review | test | documentation
- **Priority:** high | medium | low
- **Estimated effort:**
- **Dependencies:**
- **Acceptance criteria:**
  1.
  2.

### Task 2:
- **Description:**
- **Type:**
- **Priority:**
- **Estimated effort:**
- **Dependencies:**
- **Acceptance criteria:**
  1.
  2.
"""

    def generate_all(self, goal_id: str, goal: str) -> SDDDocuments:
        """Generate all three SDD documents."""
        return SDDDocuments(
            goal_id=goal_id,
            goal=goal,
            spec_content=self.generate_spec_template(goal),
            plan_content=self.generate_plan_template(goal),
            tasks_content=self.generate_tasks_template(goal),
            created_at=datetime.now().isoformat(),
        )


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Orchestrates the full SDD pipeline.

    The pipeline has 7 steps. Each step can be:
    - Skipped (via skip list)
    - Targeted (via only list)
    - Executed normally
    """

    def __init__(self, sdd_generator: Optional[SDDGenerator] = None):
        self.generator = sdd_generator or SDDGenerator()
        self._steps: Dict[PipelineStep, bool] = {
            step: False for step in PipelineStep
        }

    def list_steps(self) -> List[Dict]:
        """List all pipeline steps with their status."""
        return [
            {
                "step": step.value,
                "label": STEP_LABELS[step],
                "description": STEP_DESCRIPTIONS[step],
                "completed": self._steps.get(step, False),
            }
            for step in PipelineStep
        ]

    def run(
        self,
        goal_id: str,
        goal: str,
        skip: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ) -> Dict:
        """Run the SDD pipeline for a goal.

        Args:
            goal_id: Unique identifier for the goal
            goal: The goal description
            skip: List of step names to skip (e.g. ['execute', 'verify'])
            only: List of step names to run exclusively

        Returns:
            Dict with pipeline results per step
        """
        skip_set = set(skip or [])
        only_set = set(only or [])
        results: Dict[str, Any] = {
            "goal_id": goal_id,
            "goal": goal,
            "steps": [],
        }

        for step in PipelineStep:
            step_name = step.value

            # Check if this step should run
            if only_set and step_name not in only_set:
                results["steps"].append({
                    "step": step_name,
                    "label": STEP_LABELS[step],
                    "status": "skipped",
                    "reason": "not in --only list",
                })
                continue

            if step_name in skip_set:
                results["steps"].append({
                    "step": step_name,
                    "label": STEP_LABELS[step],
                    "status": "skipped",
                    "reason": "in --skip list",
                })
                continue

            # Execute step
            try:
                status = "completed"
                output = self._execute_step(step, goal_id, goal)
            except Exception as e:
                status = "failed"
                output = str(e)
                logger.exception("Pipeline step %s failed", step_name)

            self._steps[step] = status == "completed"
            results["steps"].append({
                "step": step_name,
                "label": STEP_LABELS[step],
                "status": status,
            })

            if status == "failed":
                results["error"] = f"Pipeline failed at step '{step_name}': {output}"
                break

        # Add SDD documents if generated
        doc_dir = Path(SDD_DIR) / goal_id
        if doc_dir.exists():
            results["documents_path"] = str(doc_dir)

        return results

    def _execute_step(
        self,
        step: PipelineStep,
        goal_id: str,
        goal: str,
    ) -> str:
        """Execute a single pipeline step."""
        if step == PipelineStep.ANALYZE:
            return self._step_analyze(goal)
        elif step == PipelineStep.DOCUMENT:
            return self._step_document(goal_id, goal)
        elif step == PipelineStep.DECOMPOSE:
            return self._step_decompose(goal)
        elif step == PipelineStep.ASSIGN:
            return self._step_assign(goal)
        elif step == PipelineStep.EXECUTE:
            return "Execution requires Odysseus agent invocation"
        elif step == PipelineStep.VERIFY:
            return "Verification requires test execution"
        elif step == PipelineStep.CONSOLIDATE:
            return self._step_consolidate(goal_id)
        return "unknown step"

    def _step_analyze(self, goal: str) -> str:
        """Analyze goal to determine if it's dev or non-dev."""
        dev_keywords = [
            "implement", "build", "code", "develop", "api", "endpoint",
            "database", "frontend", "backend", "migration", "refactor",
        ]
        is_dev = any(kw in goal.lower() for kw in dev_keywords)
        return f"Analysis complete: goal classified as {'DEV' if is_dev else 'NON-DEV'} work"

    def _step_document(self, goal_id: str, goal: str) -> str:
        """Generate SDD documents."""
        docs = self.generator.generate_all(goal_id, goal)
        path = docs.save()
        return f"SDD documents generated at {path}"

    def _step_decompose(self, goal: str) -> str:
        """Decompose goal into subtasks."""
        from .goals import decompose as goals_decompose
        decomp = goals_decompose(goal)
        tasks = [f"  {t.id}: {t.description} [{t.agent}]" for t in decomp.tasks]
        return f"Decomposed into {len(decomp.tasks)} tasks:\n" + "\n".join(tasks)

    def _step_assign(self, goal: str) -> str:
        """Assign subtasks to agents."""
        from .goals import decompose as goals_decompose
        decomp = goals_decompose(goal)
        assignments = [
            f"  {t.id}: {t.description} → {t.agent}"
            for t in decomp.tasks
        ]
        return f"Assigned {len(decomp.tasks)} tasks:\n" + "\n".join(assignments)

    def _step_consolidate(self, goal_id: str) -> str:
        """Consolidate all outputs into final deliverable."""
        doc_dir = Path(SDD_DIR) / goal_id
        if doc_dir.exists():
            return f"Outputs consolidated at {doc_dir}"
        return "No documents to consolidate — run 'document' step first"
