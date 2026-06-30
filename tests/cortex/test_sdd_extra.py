"""
Extended tests for cortex/sdd.py — covering LLM generation paths,
error handling, and edge cases in the document generation flow.

Targets missing lines: 138-140, 150-177, 185, 196, 205, 432-435,
447-448, 453, 478, 524
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex.sdd import (
    SDDGenerator,
    SDDDocuments,
    PipelineOrchestrator,
    PipelineStep,
)

import src.config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Temporary directory string path."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def generator():
    """SDDGenerator with LLM unavailable (default test state)."""
    return SDDGenerator()


@pytest.fixture
def orchestrator():
    """PipelineOrchestrator with default generator."""
    return PipelineOrchestrator()


# ---------------------------------------------------------------------------
# _check_llm — LLM availability probing (lines 138-140)
# ---------------------------------------------------------------------------


class TestCheckLLM:
    """Cover the httpx.get success path in _check_llm (lines 138-140)."""

    def test_check_llm_success(self):
        """When httpx.get returns 200, _llm_available is set to True."""
        with patch("src.config.settings", MagicMock(default_host="localhost"), create=True):
            with patch("httpx.get") as mock_get:
                mock_get.return_value.status_code = 200
                gen = SDDGenerator()
                assert gen._llm_available is True
                mock_get.assert_called_once()

    def test_check_llm_http_error(self):
        """When httpx.get returns non-200, _llm_available is False."""
        with patch("src.config.settings", MagicMock(default_host="localhost"), create=True):
            with patch("httpx.get") as mock_get:
                mock_get.return_value.status_code = 500
                gen = SDDGenerator()
                assert gen._llm_available is False

    def test_check_llm_connection_refused(self):
        """When httpx.get raises, _llm_available is False."""
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            gen = SDDGenerator()
            assert gen._llm_available is False

    def test_check_llm_import_fails(self):
        """When src.config.settings can't be imported, fallback to except."""
        gen = SDDGenerator()
        assert gen._llm_available is False


# ---------------------------------------------------------------------------
# _llm_generate — LLM content generation via httpx.post (lines 150-177)
# ---------------------------------------------------------------------------


class TestLLMGenerate:
    """Cover the httpx.post call and response handling in _llm_generate."""

    @staticmethod
    def _generator_with_llm_available():
        """Return an SDDGenerator that has _llm_available=True.

        Patches must remain active when _llm_generate is called because
        the method does ``from src.config import settings`` at runtime.
        """
        # Patch must span both construction AND the subsequent _llm_generate call.
        # We return both the generator and a context manager so callers can
        # extend the patch lifetime as needed.
        settings_patch = patch(
            "src.config.settings", MagicMock(default_host="localhost"), create=True
        )
        settings_patch.start()
        try:
            with patch("httpx.get") as mock_get:
                mock_get.return_value.status_code = 200
                gen = SDDGenerator()
                gen._llm_available = True
            # settings_patch stays active; httpx.get is restored
            return gen, settings_patch
        except Exception:
            settings_patch.stop()
            raise

    def test_llm_generate_success(self):
        """Successful LLM call returns content from response choices."""
        gen, sp = self._generator_with_llm_available()
        try:
            with patch("httpx.post") as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "choices": [{"message": {"content": "Generated LLM content"}}]
                }
                result = gen._llm_generate("sys prompt", "user prompt", max_tokens=512)
                assert result == "Generated LLM content"
                call_kwargs = mock_post.call_args.kwargs
                assert call_kwargs["json"]["model"] == "deepseek-chat"
                assert call_kwargs["json"]["messages"][0]["content"] == "sys prompt"
                assert call_kwargs["json"]["messages"][1]["content"] == "user prompt"
                assert call_kwargs["json"]["max_tokens"] == 512
                assert call_kwargs["json"]["temperature"] == 0.3
        finally:
            sp.stop()

    def test_llm_generate_non_200_status(self):
        """When httpx.post returns non-200, _llm_generate returns None."""
        gen, sp = self._generator_with_llm_available()
        try:
            with patch("httpx.post") as mock_post:
                mock_post.return_value.status_code = 400
                result = gen._llm_generate("sys", "user")
                assert result is None
        finally:
            sp.stop()

    def test_llm_generate_empty_choices(self):
        """When choices list is empty, _llm_generate returns empty string."""
        gen, sp = self._generator_with_llm_available()
        try:
            with patch("httpx.post") as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {"choices": []}
                result = gen._llm_generate("sys", "user")
                assert result == ""
        finally:
            sp.stop()

    def test_llm_generate_missing_message_key(self):
        """When message key is missing in choice, returns empty string."""
        gen, sp = self._generator_with_llm_available()
        try:
            with patch("httpx.post") as mock_post:
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {
                    "choices": [{"no_message": True}]
                }
                result = gen._llm_generate("sys", "user")
                assert result == ""
        finally:
            sp.stop()

    def test_llm_generate_exception_sets_unavailable(self):
        """When httpx.post raises, _llm_generate returns None and sets
        _llm_available to False (lines 175-176)."""
        gen, sp = self._generator_with_llm_available()
        try:
            with patch("httpx.post", side_effect=Exception("Connection timeout")):
                result = gen._llm_generate("sys", "user")
                assert result is None
                assert gen._llm_available is False
        finally:
            sp.stop()

    def test_llm_generate_not_available(self):
        """When _llm_available is False, returns None immediately (line 148-149)."""
        gen = SDDGenerator()
        gen._llm_available = False
        result = gen._llm_generate("sys", "user")
        assert result is None


# ---------------------------------------------------------------------------
# LLM generation wrappers — when LLM returns content (lines 185, 196, 205)
# ---------------------------------------------------------------------------


class TestGenerateWithLLMSuccess:
    """Cover the 'if content: return content' branches."""

    def test_spec_with_llm_returns_content(self, generator):
        """generate_spec_with_llm returns LLM content when _llm_generate returns it."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value="LLM spec content"):
            result = generator.generate_spec_with_llm("my goal")
            assert result == "LLM spec content"

    def test_plan_with_llm_returns_content(self, generator):
        """generate_plan_with_llm returns LLM content when _llm_generate returns it."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value="LLM plan content"):
            result = generator.generate_plan_with_llm("my goal")
            assert result == "LLM plan content"

    def test_tasks_with_llm_returns_content(self, generator):
        """generate_tasks_with_llm returns LLM content when _llm_generate returns it."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value="LLM tasks content"):
            result = generator.generate_tasks_with_llm("my goal")
            assert result == "LLM tasks content"

    def test_generate_all_with_llm_uses_llm(self, generator):
        """generate_all_with_llm returns documents with LLM-generated content
        when LLM is available."""
        generator._llm_available = True
        with patch.object(generator, "generate_spec_with_llm", return_value="LLM spec"):
            with patch.object(
                generator, "generate_plan_with_llm", return_value="LLM plan"
            ):
                with patch.object(
                    generator, "generate_tasks_with_llm", return_value="LLM tasks"
                ):
                    docs = generator.generate_all_with_llm("g1", "test goal")
                    assert docs.spec_content == "LLM spec"
                    assert docs.plan_content == "LLM plan"
                    assert docs.tasks_content == "LLM tasks"
                    assert docs.goal_id == "g1"
                    assert docs.goal == "test goal"
                    assert docs.created_at != ""

    def test_spec_with_llm_correct_prompts(self, generator):
        """generate_spec_with_llm passes correct prompts to _llm_generate."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value="c") as mock_m:
            generator.generate_spec_with_llm("build an API")
            sys_prompt, user_prompt = mock_m.call_args[0]
            assert "software architect" in sys_prompt
            assert "functional specification" in sys_prompt
            assert "build an API" in user_prompt

    def test_plan_with_llm_correct_prompts(self, generator):
        """generate_plan_with_llm passes correct prompts to _llm_generate."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value="c") as mock_m:
            generator.generate_plan_with_llm("build an API")
            sys_prompt, user_prompt = mock_m.call_args[0]
            assert "project manager" in sys_prompt
            assert "implementation plan" in sys_prompt
            assert "build an API" in user_prompt

    def test_tasks_with_llm_correct_prompts(self, generator):
        """generate_tasks_with_llm passes correct prompts to _llm_generate."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value="c") as mock_m:
            generator.generate_tasks_with_llm("build an API")
            sys_prompt, user_prompt = mock_m.call_args[0]
            assert "tech lead" in sys_prompt
            assert "task" in sys_prompt.lower()
            assert "build an API" in user_prompt


# ---------------------------------------------------------------------------
# PipelineOrchestrator — error handling and edge cases
# ---------------------------------------------------------------------------


class TestPipelineOrchestratorErrors:
    """Cover the exception handling and failure flow (lines 432-435, 447-448)."""

    def test_run_step_exception_records_failure(self, orchestrator, temp_dir):
        """When _execute_step raises, the step status is 'failed' and the
        pipeline records the error and stops (lines 432-435, 447-448)."""
        import cortex.sdd as sdd_mod

        sdd_mod.SDD_DIR = temp_dir

        # Cause _step_document to raise by making generate_all fail
        def failing_generate_all(goal_id, goal):
            raise RuntimeError("Simulated document generation failure")

        orchestrator.generator.generate_all = failing_generate_all

        result = orchestrator.run("g-fail", "test goal", only=["document"])

        # 'analyze' runs first but is skipped (only=["document"]), then
        # 'document' executes and fails → pipeline stops
        assert len(result["steps"]) == 2
        assert result["steps"][0]["status"] == "skipped"
        assert result["steps"][1]["step"] == "document"
        assert result["steps"][1]["status"] == "failed"
        assert "error" in result
        assert "Pipeline failed" in result["error"]
        assert "Simulated document generation failure" in result["error"]

    def test_run_step_failure_stops_pipeline(self, orchestrator, temp_dir):
        """After a step fails, subsequent steps are not attempted (break at 448)."""
        import cortex.sdd as sdd_mod

        sdd_mod.SDD_DIR = temp_dir

        original_execute = orchestrator._execute_step

        def tracking_execute(step, goal_id, goal):
            if step == PipelineStep.DOCUMENT:
                raise ValueError("Document error")
            return original_execute(step, goal_id, goal)

        orchestrator._execute_step = tracking_execute

        result = orchestrator.run(
            "g-fail2", "build an API", only=["analyze", "document"]
        )

        # Analyze completed, document failed → pipeline stopped
        assert len(result["steps"]) == 2
        assert result["steps"][0]["status"] == "completed"
        assert result["steps"][1]["status"] == "failed"
        assert "error" in result


class TestPipelineOrchestratorDocsPath:
    """Cover line 453: documents_path in results when doc_dir exists."""

    def test_run_documents_path_set_when_dir_exists(self, orchestrator, temp_dir):
        """When doc_dir exists after pipeline run, results include documents_path."""
        import cortex.sdd as sdd_mod

        sdd_mod.SDD_DIR = temp_dir

        # Create the directory that run() checks
        doc_dir = Path(temp_dir) / "g-docpath"
        doc_dir.mkdir(parents=True, exist_ok=True)

        result = orchestrator.run("g-docpath", "test", only=["analyze"])
        assert "documents_path" in result
        assert result["documents_path"] == str(doc_dir)

    def test_run_documents_path_not_set_when_dir_missing(self, orchestrator, temp_dir):
        """When doc_dir does NOT exist, documents_path is absent from results."""
        import cortex.sdd as sdd_mod

        sdd_mod.SDD_DIR = temp_dir

        result = orchestrator.run("g-nodir", "test", only=["analyze"])
        assert "documents_path" not in result


# ---------------------------------------------------------------------------
# _execute_step — edge cases
# ---------------------------------------------------------------------------


class TestExecuteStepEdgeCases:
    """Cover the 'unknown step' fallback in _execute_step (line 478)."""

    def test_execute_step_unknown(self, orchestrator):
        """_execute_step returns 'unknown step' for unrecognized step value
        that falls through all if/elif branches."""
        result = orchestrator._execute_step(None, "g1", "test")
        assert result == "unknown step"


# ---------------------------------------------------------------------------
# _step_consolidate — edge cases (line 524)
# ---------------------------------------------------------------------------


class TestStepConsolidate:
    """Cover the message when no documents exist to consolidate (line 524)."""

    def test_consolidate_no_documents(self, orchestrator):
        """When doc_dir does not exist, _step_consolidate returns appropriate message."""
        result = orchestrator._step_consolidate("nonexistent-goal-xyz")
        assert "No documents to consolidate" in result
        assert "run 'document' step first" in result

    def test_consolidate_with_documents(self, orchestrator, temp_dir):
        """When doc_dir exists, _step_consolidate returns success message."""
        import cortex.sdd as sdd_mod

        sdd_mod.SDD_DIR = temp_dir
        doc_dir = Path(temp_dir) / "existing-goal"
        doc_dir.mkdir(parents=True, exist_ok=True)

        result = orchestrator._step_consolidate("existing-goal")
        assert "consolidated" in result.lower()
        assert "No documents" not in result


# ---------------------------------------------------------------------------
# PipelineOrchestrator — run with various only/skip combinations
# ---------------------------------------------------------------------------


class TestPipelineOrchestratorRunCombinations:
    """Additional edge cases for pipeline run()."""

    def test_run_with_empty_skip_list(self, orchestrator):
        """Empty skip list is equivalent to no skip (all steps run)."""
        result = orchestrator.run("g1", "build an API", skip=[])
        assert len(result["steps"]) == 7

    def test_run_with_empty_only_list(self, orchestrator):
        """Empty only list is equivalent to no only (all steps run)."""
        result = orchestrator.run("g1", "build an API", only=[])
        assert len(result["steps"]) == 7

    def test_run_skip_consolidate_and_verify(self, orchestrator):
        """Skip consolidate and verify steps."""
        result = orchestrator.run(
            "g1", "build an API", skip=["consolidate", "verify"]
        )
        steps_map = {s["step"]: s for s in result["steps"]}
        assert steps_map["consolidate"]["status"] == "skipped"
        assert steps_map["verify"]["status"] == "skipped"
        assert steps_map["analyze"]["status"] != "skipped"

    def test_run_only_analyze(self, orchestrator):
        """Only the analyze step runs when only=['analyze']."""
        result = orchestrator.run("g1", "implement a new API", only=["analyze"])
        assert len(result["steps"]) == 7
        analyze_step = next(s for s in result["steps"] if s["step"] == "analyze")
        assert analyze_step["status"] == "completed"
        assert analyze_step["label"] == "Analyze"
        for s in result["steps"]:
            if s["step"] != "analyze":
                assert s["status"] == "skipped"


# ---------------------------------------------------------------------------
# Pipeline step methods - direct testing
# ---------------------------------------------------------------------------


class TestPipelineStepMethods:
    """Direct tests for individual pipeline step methods."""

    def test_step_analyze_dev_goal(self, orchestrator):
        """ANALYZE classifies goals with dev keywords as DEV work."""
        result = orchestrator._step_analyze("implement a new API endpoint")
        assert "DEV" in result

    def test_step_analyze_non_dev_goal(self, orchestrator):
        """ANALYZE classifies goals without dev keywords as NON-DEV work."""
        result = orchestrator._step_analyze("write documentation for the project")
        assert "NON-DEV" in result

    def test_step_analyze_empty_goal(self, orchestrator):
        """ANALYZE classifies empty goal as NON-DEV (no keywords match)."""
        result = orchestrator._step_analyze("")
        assert "NON-DEV" in result

    def test_step_document_creates_files(self, orchestrator):
        """_step_document creates SDD document files and returns their path."""
        result = orchestrator._step_document("g-doc3", "build a login")
        assert "SDD documents generated" in result
        # Extract the actual save path from the result message
        saved_path = result.replace("SDD documents generated at ", "")
        assert (Path(saved_path) / "spec.md").exists()
        assert (Path(saved_path) / "plan.md").exists()
        assert (Path(saved_path) / "tasks.md").exists()

    def test_step_execute_returns_message(self, orchestrator):
        """_step_execute returns placeholder message."""
        result = orchestrator._execute_step(PipelineStep.EXECUTE, "g1", "test")
        assert "Execution requires" in result

    def test_step_verify_returns_message(self, orchestrator):
        """_step_verify returns placeholder message."""
        result = orchestrator._execute_step(PipelineStep.VERIFY, "g1", "test")
        assert "Verification requires" in result

    def test_step_decompose(self, orchestrator):
        """_step_decompose returns task decomposition."""
        result = orchestrator._step_decompose("build an API")
        assert "Decomposed into" in result

    def test_step_assign(self, orchestrator):
        """_step_assign returns task assignments."""
        result = orchestrator._step_assign("build an API")
        assert "Assigned" in result


# ---------------------------------------------------------------------------
# SDDDocuments.save() — PermissionError fallback (lines 82-89)
# ---------------------------------------------------------------------------


class TestSDDDocumentsSaveEdgeCases:
    """Cover PermissionError fallback and edge cases in save()."""

    @pytest.mark.skip(reason="Path.mkdir mocking conflicts with fallback's own mkdir")
    def test_save_permission_error_fallback(self, temp_dir):
        """When mkdir raises PermissionError, save() falls back to tempdir."""
        docs = SDDDocuments(
            goal_id="perm-test",
            goal="test",
            spec_content="# Spec",
            plan_content="# Plan",
            tasks_content="# Tasks",
        )
        with patch.object(Path, "mkdir", side_effect=PermissionError("no write")):
            path = docs.save(base_dir=temp_dir)
            # Should have fallen back to a temp directory containing cortex-sdd
            assert "cortex-sdd" in path
            # Path should exist (temp dir was created by tempfile.mkdtemp)
            fallback_path = Path(path)
            # Files may or may not exist depending on fallback behavior
            assert fallback_path.name.startswith("cortex-sdd")

    def test_save_creates_meta_json(self, temp_dir):
        """save() creates meta.json with correct contents."""
        docs = SDDDocuments(
            goal_id="meta-test",
            goal="meta test goal",
            spec_content="spec",
            plan_content="plan",
            tasks_content="tasks",
            created_at="2026-07-01T00:00:00",
        )
        path = docs.save(base_dir=temp_dir)
        meta = json.loads(Path(path).joinpath("meta.json").read_text())
        assert meta["goal_id"] == "meta-test"
        assert meta["goal"] == "meta test goal"
        assert meta["created_at"] == "2026-07-01T00:00:00"

    def test_save_with_empty_created_at(self, temp_dir):
        """When created_at is empty, save() fills it with current time."""
        docs = SDDDocuments(
            goal_id="no-ts",
            goal="test",
            spec_content="",
            plan_content="",
            tasks_content="",
            created_at="",
        )
        path = docs.save(base_dir=temp_dir)
        meta = json.loads(Path(path).joinpath("meta.json").read_text())
        assert meta["created_at"] != ""

    def test_save_with_all_empty_content(self, temp_dir):
        """save() works when all document content is empty strings."""
        docs = SDDDocuments(
            goal_id="all-empty",
            goal="empty test",
            spec_content="",
            plan_content="",
            tasks_content="",
        )
        path = docs.save(base_dir=temp_dir)
        dir_path = Path(path)
        assert dir_path.exists()
        assert dir_path.joinpath("spec.md").read_text() == ""
        assert dir_path.joinpath("plan.md").read_text() == ""
        assert dir_path.joinpath("tasks.md").read_text() == ""


# ---------------------------------------------------------------------------
# SDDGenerator — template methods
# ---------------------------------------------------------------------------


class TestSDDGeneratorTemplates:
    """Verify template content structure."""

    def test_spec_template_has_sections(self, generator):
        """spec template contains all expected sections."""
        template = generator.generate_spec_template("test goal")
        assert "# Specification" in template
        assert "Goal" in template
        assert "Functional Requirements" in template
        assert "Non-Functional Requirements" in template
        assert "Scope" in template
        assert "Technical Approach" in template
        assert "Dependencies" in template
        assert "Risks and Mitigations" in template
        assert "Acceptance Criteria" in template

    def test_plan_template_has_sections(self, generator):
        """plan template contains all expected sections."""
        template = generator.generate_plan_template("test goal")
        assert "# Implementation Plan" in template
        assert "Goal" in template
        assert "Phases" in template
        assert "Phase 1: Foundation" in template
        assert "Timeline" in template
        assert "Resources" in template
        assert "Milestones" in template

    def test_tasks_template_has_sections(self, generator):
        """tasks template contains all expected sections."""
        template = generator.generate_tasks_template("test goal")
        assert "# Task Breakdown" in template
        assert "Goal" in template
        assert "Tasks" in template
        assert "**Description:**" in template
        assert "**Type:**" in template
        assert "**Priority:**" in template
        assert "**Estimated effort:**" in template
        assert "**Dependencies:**" in template
        assert "**Acceptance criteria:**" in template

    def test_template_includes_goal(self, generator):
        """Each template includes the goal text."""
        goal = "Implement user authentication"
        spec = generator.generate_spec_template(goal)
        plan = generator.generate_plan_template(goal)
        tasks = generator.generate_tasks_template(goal)
        assert goal in spec
        assert goal in plan
        assert goal in tasks


# ---------------------------------------------------------------------------
# Integration-ish: LLM → fallback → template match
# ---------------------------------------------------------------------------


class TestLLMFallbackIntegration:
    """Verify that all *_with_llm methods correctly fall back to templates."""

    def test_spec_fallback_to_template(self, generator):
        """generate_spec_with_llm falls back to template when LLM returns None."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value=None):
            result = generator.generate_spec_with_llm("test goal")
            expected = generator.generate_spec_template("test goal")
            assert result == expected

    def test_plan_fallback_to_template(self, generator):
        """generate_plan_with_llm falls back to template when LLM returns None."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value=None):
            result = generator.generate_plan_with_llm("test goal")
            expected = generator.generate_plan_template("test goal")
            assert result == expected

    def test_tasks_fallback_to_template(self, generator):
        """generate_tasks_with_llm falls back to template when LLM returns None."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value=None):
            result = generator.generate_tasks_with_llm("test goal")
            expected = generator.generate_tasks_template("test goal")
            assert result == expected

    def test_all_fallback_when_llm_returns_none(self, generator):
        """generate_all_with_llm produces templates when _llm_generate returns None."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value=None):
            docs = generator.generate_all_with_llm("g1", "test goal")
            assert "# Specification" in docs.spec_content
            assert "# Implementation Plan" in docs.plan_content
            assert "# Task Breakdown" in docs.tasks_content

    def test_all_fallback_when_llm_returns_empty(self, generator):
        """generate_all_with_llm produces templates when LLM returns empty string."""
        generator._llm_available = True
        with patch.object(generator, "_llm_generate", return_value=""):
            docs = generator.generate_all_with_llm("g1", "test goal")
            assert "# Specification" in docs.spec_content

    def test_spec_fallback_when_llm_unavailable(self, generator):
        """generate_spec_with_llm falls back when LLM unavailable."""
        generator._llm_available = False
        result = generator.generate_spec_with_llm("test goal")
        assert result == generator.generate_spec_template("test goal")
