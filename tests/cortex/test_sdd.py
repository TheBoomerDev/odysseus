"""
Tests for cortex/sdd.py — Spec-Driven Development Pipeline.

Tests:
  - generate_all() produces templates
  - generate_all_with_llm() falls back to templates when LLM unavailable
  - SDDDocuments.save()
  - PipelineOrchestrator.list_steps()
"""

import json
import tempfile
from pathlib import Path

import pytest

from cortex.sdd import (
    SDDGenerator,
    SDDDocuments,
    PipelineOrchestrator,
    PipelineStep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def generator():
    return SDDGenerator()


@pytest.fixture
def orchestrator():
    return PipelineOrchestrator()


# ---------------------------------------------------------------------------
# SDDGenerator — generate_all (templates)
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_generate_all_returns_sdd_documents(self, generator):
        """generate_all() returns an SDDDocuments instance."""
        docs = generator.generate_all("goal-1", "build a login system")
        assert isinstance(docs, SDDDocuments)
        assert docs.goal_id == "goal-1"
        assert docs.goal == "build a login system"

    def test_generate_all_spec_content(self, generator):
        """Spec content is a markdown template with goal reference."""
        docs = generator.generate_all("g1", "build an API")
        assert "# Specification" in docs.spec_content
        assert "build an API" in docs.spec_content

    def test_generate_all_plan_content(self, generator):
        """Plan content is a markdown template with goal reference."""
        docs = generator.generate_all("g1", "build an API")
        assert "# Implementation Plan" in docs.plan_content
        assert "build an API" in docs.plan_content

    def test_generate_all_tasks_content(self, generator):
        """Tasks content is a markdown template with goal reference."""
        docs = generator.generate_all("g1", "build an API")
        assert "# Task Breakdown" in docs.tasks_content
        assert "build an API" in docs.tasks_content

    def test_generate_all_created_at(self, generator):
        """Generated documents have a created_at timestamp."""
        docs = generator.generate_all("g1", "test")
        assert docs.created_at != ""

    def test_generate_all_spec_contains_sections(self, generator):
        """Spec template includes all standard sections."""
        docs = generator.generate_all("g1", "test")
        assert "Functional Requirements" in docs.spec_content
        assert "Non-Functional Requirements" in docs.spec_content
        assert "Technical Approach" in docs.spec_content
        assert "Dependencies" in docs.spec_content
        assert "Acceptance Criteria" in docs.spec_content


# ---------------------------------------------------------------------------
# SDDGenerator — generate_all_with_llm fallback
# ---------------------------------------------------------------------------

class TestGenerateAllWithLLM:
    def test_fallback_to_templates(self, generator):
        """When LLM is unavailable, generate_all_with_llm falls back to templates."""
        generator._llm_available = False
        docs = generator.generate_all_with_llm("g1", "build a login system")
        assert "# Specification" in docs.spec_content
        assert "# Implementation Plan" in docs.plan_content
        assert "# Task Breakdown" in docs.tasks_content

    def test_llm_available_false_by_default(self, generator):
        """In test environments, LLM is typically unavailable."""
        assert generator._llm_available is False

    def test_spec_fallback_matches_template(self, generator):
        """generate_spec_with_llm falls back to generate_spec_template."""
        generator._llm_available = False
        result = generator.generate_spec_with_llm("my goal")
        template = generator.generate_spec_template("my goal")
        assert result == template

    def test_plan_fallback_matches_template(self, generator):
        """generate_plan_with_llm falls back to generate_plan_template."""
        generator._llm_available = False
        result = generator.generate_plan_with_llm("my goal")
        template = generator.generate_plan_template("my goal")
        assert result == template

    def test_tasks_fallback_matches_template(self, generator):
        """generate_tasks_with_llm falls back to generate_tasks_template."""
        generator._llm_available = False
        result = generator.generate_tasks_with_llm("my goal")
        template = generator.generate_tasks_template("my goal")
        assert result == template


# ---------------------------------------------------------------------------
# SDDDocuments.save()
# ---------------------------------------------------------------------------

class TestSDDDocumentsSave:
    def test_save_creates_files(self, temp_dir):
        """save() creates spec.md, plan.md, tasks.md, meta.json."""
        docs = SDDDocuments(
            goal_id="test-g1",
            goal="test goal",
            spec_content="# Spec",
            plan_content="# Plan",
            tasks_content="# Tasks",
            created_at="2026-01-01T00:00:00",
        )
        path = docs.save(base_dir=temp_dir)
        dir_path = Path(path)
        assert (dir_path / "spec.md").exists()
        assert (dir_path / "plan.md").exists()
        assert (dir_path / "tasks.md").exists()
        assert (dir_path / "meta.json").exists()

    def test_save_file_contents(self, temp_dir):
        """Saved files contain the correct document content."""
        docs = SDDDocuments(
            goal_id="test-g2",
            goal="test goal",
            spec_content="# My Spec Content",
            plan_content="# My Plan Content",
            tasks_content="# My Tasks Content",
            created_at="2026-06-01T12:00:00",
        )
        path = docs.save(base_dir=temp_dir)
        dir_path = Path(path)
        assert dir_path.joinpath("spec.md").read_text() == "# My Spec Content"
        assert dir_path.joinpath("plan.md").read_text() == "# My Plan Content"
        assert dir_path.joinpath("tasks.md").read_text() == "# My Tasks Content"

    def test_save_meta_json(self, temp_dir):
        """meta.json contains goal_id, goal, and created_at."""
        docs = SDDDocuments(
            goal_id="test-g3",
            goal="save test",
            spec_content="",
            plan_content="",
            tasks_content="",
            created_at="2026-06-30T00:00:00",
        )
        path = docs.save(base_dir=temp_dir)
        meta = json.loads(Path(path).joinpath("meta.json").read_text())
        assert meta["goal_id"] == "test-g3"
        assert meta["goal"] == "save test"
        assert meta["created_at"] == "2026-06-30T00:00:00"

    def test_save_returns_dir_path(self, temp_dir):
        """save() returns the path to the created directory."""
        docs = SDDDocuments(goal_id="path-test", goal="x", spec_content="", plan_content="", tasks_content="")
        path = docs.save(base_dir=temp_dir)
        assert Path(path).name == "path-test"


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------

class TestPipelineOrchestrator:
    def test_list_steps_contains_all_7(self, orchestrator):
        """list_steps() returns all 7 pipeline steps."""
        steps = orchestrator.list_steps()
        assert len(steps) == 7

    def test_list_steps_keys(self, orchestrator):
        """Each listed step has step, label, description, completed."""
        steps = orchestrator.list_steps()
        for s in steps:
            assert "step" in s
            assert "label" in s
            assert "description" in s
            assert "completed" in s
            assert isinstance(s["completed"], bool)

    def test_list_steps_not_completed_by_default(self, orchestrator):
        """Steps are not completed by default."""
        steps = orchestrator.list_steps()
        for s in steps:
            assert s["completed"] is False

    def test_list_steps_all_pipeline_steps_present(self, orchestrator):
        """All PipelineStep enum values appear in the list."""
        step_values = {s["step"] for s in orchestrator.list_steps()}
        for step in PipelineStep:
            assert step.value in step_values

    def test_list_steps_ordering(self, orchestrator):
        """Steps are listed in PipelineStep enum order."""
        steps = orchestrator.list_steps()
        expected_order = [s.value for s in PipelineStep]
        actual_order = [s["step"] for s in steps]
        assert actual_order == expected_order

    def test_run_analyze_step(self, orchestrator, generator):
        """Running the ANALYZE step classifies the goal."""
        orchestrator.generator = generator
        result = orchestrator.run("g1", "implement a new API", only=["analyze"])
        assert result["goal_id"] == "g1"
        analyze_step = next(s for s in result["steps"] if s["step"] == "analyze")
        assert analyze_step["status"] == "completed"

    def test_run_document_step_creates_files(self, orchestrator, generator, temp_dir):
        """Running DOCUMENT generates SDD files."""
        import cortex.sdd as sdd_mod
        sdd_mod.SDD_DIR = temp_dir
        orchestrator.generator = generator
        result = orchestrator.run("g-doc", "build a login", only=["document"])
        doc_step = next(s for s in result["steps"] if s["step"] == "document")
        assert doc_step["status"] == "completed"

    def test_run_skip_steps(self, orchestrator, generator):
        """Skipped steps are reported with 'skipped' status."""
        orchestrator.generator = generator
        result = orchestrator.run("g-skip", "build an API", skip=["execute", "verify"])
        skipped = [s for s in result["steps"] if s.get("status") == "skipped"]
        assert any("execute" in s["step"] for s in skipped)
        assert any("verify" in s["step"] for s in skipped)

    def test_run_only_steps(self, orchestrator, generator):
        """When 'only' is set, only specified steps run (others are skipped)."""
        orchestrator.generator = generator
        result = orchestrator.run("g-only", "build an API", only=["analyze", "document"])
        # All 7 steps are returned; only-specified ones are 'completed', rest 'skipped'
        assert len(result["steps"]) == 7
        for s in result["steps"]:
            if s["step"] in ("analyze", "document"):
                assert s["status"] == "completed"
            else:
                assert s["status"] == "skipped"

    def test_run_default_runs_all(self, orchestrator, generator):
        """Without skip/only, all 7 steps are attempted."""
        orchestrator.generator = generator
        result = orchestrator.run("g-all", "build an API")
        # All 7 should be present (some may be completed, some may have status text)
        assert len(result["steps"]) == 7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSDDEdgeCases:
    def test_empty_goal(self, generator):
        """generate_all handles empty goal strings."""
        docs = generator.generate_all("empty", "")
        assert docs.goal == ""
        assert docs.goal_id == "empty"
        assert "# Specification" in docs.spec_content

    def test_long_goal(self, generator):
        """generate_all handles very long goals."""
        long_goal = "x" * 10000
        docs = generator.generate_all("long", long_goal)
        assert long_goal in docs.spec_content

    def test_special_characters_in_goal(self, generator):
        """Special characters appear correctly in templates."""
        docs = generator.generate_all("special", "goal with $pecial ch@rs & more!")
        assert "$pecial" in docs.spec_content
        assert "ch@rs" in docs.spec_content

    def test_save_empty_documents(self, temp_dir):
        """save() works even with empty document content."""
        docs = SDDDocuments(goal_id="empty-doc", goal="empty", spec_content="", plan_content="", tasks_content="")
        path = docs.save(base_dir=temp_dir)
        assert Path(path).exists()
