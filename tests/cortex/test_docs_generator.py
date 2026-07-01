"""Comprehensive tests for cortex/docs_generator.py."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.codegen.llm import LLMResult
from cortex.docs_generator import (
    DOC_SECTIONS,
    DocSection,
    DocsGenerationError,
    DocsGenerator,
    DocsResult,
    generate_project_docs,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_result() -> LLMResult:
    """A successful LLM result with sample markdown content."""
    return LLMResult(
        content="# Generated Content\n\nThis is the test document content.\n\n## Section 1\n\nDetails here.",
        provider="test",
        model="test-model",
    )


@pytest.fixture
def mock_llm_client(mock_llm_result: LLMResult) -> AsyncMock:
    """An LLM client mock whose chat() returns a successful result."""
    client = AsyncMock()
    client.model = "test-model"
    client.chat = AsyncMock(return_value=mock_llm_result)
    return client


@pytest.fixture
def mock_get_llm_client(mock_llm_client: AsyncMock) -> MagicMock:
    """Patch get_llm_client so DocGenerator uses our mock."""
    with patch("cortex.docs_generator.get_llm_client", return_value=mock_llm_client) as m:
        yield m


@pytest.fixture
def generator(mock_get_llm_client: MagicMock) -> DocsGenerator:
    """A DocsGenerator instance with a mocked LLM client."""
    return DocsGenerator(provider="hermes", profile="swarm1")


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """A temporary directory for writing doc outputs."""
    d = tmp_path / "docs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================================
# DOC_SECTIONS
# ============================================================================


class TestDocSections:
    """Verify the DOC_SECTIONS constant."""

    def test_has_eight_sections(self) -> None:
        assert len(DOC_SECTIONS) == 8

    def test_section_ids_and_filenames(self) -> None:
        expected = [
            ("prd", "01-prd.md"),
            ("specs", "02-specs.md"),
            ("functional-memory", "03-functional-memory.md"),
            ("technical-memory", "04-technical-memory.md"),
            ("roadmap", "05-roadmap.md"),
            ("visual-design", "06-visual-design.md"),
            ("buyer-persona", "07-buyer-persona.md"),
            ("competitor-analysis", "08-competitor-analysis.md"),
        ]
        for section, (expected_id, expected_fn) in zip(DOC_SECTIONS, expected):
            assert section.id == expected_id, f"Mismatch for {expected_id}"
            assert section.filename == expected_fn, f"Mismatch for {expected_id}"

    def test_each_section_has_required_fields(self) -> None:
        for section in DOC_SECTIONS:
            assert isinstance(section.id, str) and section.id
            assert isinstance(section.title, str) and section.title
            assert isinstance(section.filename, str) and section.filename
            assert isinstance(section.description, str) and section.description
            assert isinstance(section.prompt_template, str) and section.prompt_template

    def test_section_titles_are_unique(self) -> None:
        titles = [s.title for s in DOC_SECTIONS]
        assert len(titles) == len(set(titles))

    def test_section_ids_are_unique(self) -> None:
        ids = [s.id for s in DOC_SECTIONS]
        assert len(ids) == len(set(ids))

    def test_first_section_is_prd(self) -> None:
        assert DOC_SECTIONS[0].id == "prd"
        assert DOC_SECTIONS[0].filename == "01-prd.md"

    def test_last_section_is_competitor_analysis(self) -> None:
        assert DOC_SECTIONS[-1].id == "competitor-analysis"
        assert DOC_SECTIONS[-1].filename == "08-competitor-analysis.md"


# ============================================================================
# DocsResult dataclass
# ============================================================================


class TestDocsResult:
    """Verify the DocsResult dataclass."""

    def test_minimal_instantiation(self) -> None:
        result = DocsResult(section_id="prd", title="PRD", filename="01-prd.md", success=True)
        assert result.section_id == "prd"
        assert result.title == "PRD"
        assert result.filename == "01-prd.md"
        assert result.success is True
        assert result.error is None
        assert result.file_path is None
        assert result.char_count == 0

    def test_full_instantiation(self) -> None:
        result = DocsResult(
            section_id="specs",
            title="Technical Specs",
            filename="02-specs.md",
            success=True,
            error=None,
            file_path="/tmp/docs/02-specs.md",
            char_count=1234,
        )
        assert result.section_id == "specs"
        assert result.char_count == 1234
        assert result.file_path == "/tmp/docs/02-specs.md"

    def test_error_instantiation(self) -> None:
        result = DocsResult(
            section_id="prd",
            title="PRD",
            filename="01-prd.md",
            success=False,
            error="LLM returned empty content",
        )
        assert result.success is False
        assert result.error == "LLM returned empty content"

    def test_docs_result_is_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(DocsResult)


# ============================================================================
# DocsGenerationError
# ============================================================================


class TestDocsGenerationError:
    def test_is_exception(self) -> None:
        err = DocsGenerationError("something failed")
        assert isinstance(err, Exception)
        assert str(err) == "something failed"


# ============================================================================
# DocsGenerator.__init__ and properties
# ============================================================================


class TestDocsGeneratorInit:
    """Verify DocsGenerator construction."""

    def test_default_hermes_provider(self, mock_get_llm_client: MagicMock) -> None:
        gen = DocsGenerator()
        assert gen.provider == "hermes"
        # model should default to whatever the mock llm client reports
        assert gen.model == "test-model"
        mock_get_llm_client.assert_called_once_with(
            provider="hermes", model=None, api_key=None, profile="swarm1"
        )

    def test_openai_provider(self, mock_get_llm_client: MagicMock) -> None:
        gen = DocsGenerator(
            provider="openai", model="gpt-4o", api_key="sk-test123"
        )
        assert gen.provider == "openai"
        mock_get_llm_client.assert_called_with(
            provider="openai", model="gpt-4o", api_key="sk-test123", profile="swarm1"
        )

    def test_custom_profile(self, mock_get_llm_client: MagicMock) -> None:
        DocsGenerator(provider="hermes", profile="custom-profile")
        # Profile is forwarded to get_llm_client, not stored on the generator
        mock_get_llm_client.assert_called_once_with(
            provider="hermes", model=None, api_key=None, profile="custom-profile"
        )

    def test_custom_temperature(self, mock_get_llm_client: MagicMock) -> None:
        gen = DocsGenerator(temperature=0.7)
        # temperature is stored as an attribute for use in chat calls
        assert gen._temperature == 0.7  # noqa: SLF001

    @pytest.mark.parametrize("provider,model", [
        ("hermes", None),
        ("openai", "gpt-4o"),
        ("gemini", "gemini-2.0-flash"),
        ("ollama", "llama3"),
    ])
    def test_provider_and_model_properties(
        self,
        provider: str,
        model: Optional[str],
        mock_get_llm_client: MagicMock,
        mock_llm_client: AsyncMock,
    ) -> None:
        mock_llm_client.model = model or f"{provider}/swarm1"
        gen = DocsGenerator(provider=provider, model=model)
        assert gen.provider == provider
        if model:
            assert gen.model == model
        else:
            assert gen.model is not None


# ============================================================================
# DocsGenerator._format_doc
# ============================================================================


class TestFormatDoc:
    """Verify the static _format_doc method."""

    def test_creates_yaml_frontmatter(self) -> None:
        section = DOC_SECTIONS[0]
        content = "# My Content"
        result = DocsGenerator._format_doc(section, "MyApp", content)

        assert result.startswith("---\n")
        assert "title: Product Requirements Document (PRD)" in result
        assert "project: MyApp" in result
        assert "generated_at:" in result
        assert "generator: odysseus/docs_generator" in result
        assert "---" in result
        assert "# My Content" in result

    def test_frontmatter_has_iso_timestamp(self) -> None:
        section = DOC_SECTIONS[0]
        result = DocsGenerator._format_doc(section, "MyApp", "content")
        # Parse the generated_at timestamp
        start = result.find("generated_at: ") + len("generated_at: ")
        end = result.find("\n", start)
        ts_str = result[start:end].strip()
        parsed = datetime.fromisoformat(ts_str)
        assert parsed.tzinfo is not None  # timezone-aware

    def test_strips_markdown_code_fences(self) -> None:
        section = DOC_SECTIONS[0]
        content = "```markdown\n# Heading\n\nSome text\n```"
        result = DocsGenerator._format_doc(section, "MyApp", content)
        # The fences should be removed, leaving clean content
        assert "```" not in result
        assert "# Heading" in result
        assert "Some text" in result

    def test_strips_fences_without_language(self) -> None:
        section = DOC_SECTIONS[0]
        content = "```\n# Heading\n\nSome text\n```"
        result = DocsGenerator._format_doc(section, "MyApp", content)
        assert "```" not in result
        assert "# Heading" in result

    def test_no_fences_content_unchanged(self) -> None:
        section = DOC_SECTIONS[0]
        content = "# Heading\n\nThis is plain content.\n"
        result = DocsGenerator._format_doc(section, "MyApp", content)
        assert "# Heading" in result
        assert "This is plain content." in result
        # Should still be in the output after frontmatter
        assert result.strip().endswith("# Heading\n\nThis is plain content.")

    def test_fences_only_at_start_and_end(self) -> None:
        """Only outermost fences wrapping entire content are stripped."""
        section = DOC_SECTIONS[0]
        content = "```\n```python\ncode here\n```\n```"
        result = DocsGenerator._format_doc(section, "MyApp", content)
        # The outer fences get stripped; inner ones remain
        assert "```python" in result
        assert "code here" in result

    def test_fences_when_opening_has_no_closing(self) -> None:
        """When content starts with ``` but doesn't end with ```, the last
        occurrence of ``` is stripped."""
        section = DOC_SECTIONS[0]
        content = "```\nsome content\n```\nextra trailing text"
        result = DocsGenerator._format_doc(section, "MyApp", content)
        # The fences and the trailing text after the last ``` should be removed
        assert "```" not in result
        assert "some content" in result
        assert "extra trailing text" not in result

    def test_static_method(self) -> None:
        """Verify it's a @staticmethod and can be called on the class."""
        section = DOC_SECTIONS[0]
        result = DocsGenerator._format_doc(section, "App", "hi")
        assert isinstance(result, str)


# ============================================================================
# DocsGenerator._generate_section
# ============================================================================


class TestGenerateSection:
    """Verify the internal _generate_section method."""

    async def test_success(
        self,
        generator: DocsGenerator,
        mock_llm_client: AsyncMock,
        tmp_output_dir: Path,
    ) -> None:
        section = DOC_SECTIONS[0]
        result = await generator._generate_section(
            section=section,
            project_name="TestProject",
            user_prompt="A test project",
            output_path=tmp_output_dir,
        )

        assert result.success is True
        assert result.section_id == "prd"
        assert result.filename == "01-prd.md"
        assert result.error is None
        assert result.file_path is not None
        assert result.char_count > 0

        # Verify file was written
        file_path = tmp_output_dir / "01-prd.md"
        assert file_path.exists()
        written = file_path.read_text(encoding="utf-8")
        assert "title: Product Requirements Document (PRD)" in written
        assert "project: TestProject" in written
        assert "# Generated Content" in written

        # Verify LLM was called correctly
        mock_llm_client.chat.assert_awaited_once()
        call_args = mock_llm_client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "TestProject" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert mock_llm_client.chat.call_args.kwargs["temperature"] == 0.3
        assert mock_llm_client.chat.call_args.kwargs["max_tokens"] == 4096

    async def test_llm_returns_error(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        mock_llm_client.chat.return_value = LLMResult(
            content="",
            error="API rate limit exceeded",
            provider="test",
            model="test-model",
            finish_reason="error",
        )
        gen = DocsGenerator()
        section = DOC_SECTIONS[0]
        result = await gen._generate_section(
            section=section,
            project_name="TestProject",
            user_prompt="A test project",
            output_path=tmp_output_dir,
        )

        assert result.success is False
        assert result.error == "API rate limit exceeded"
        assert result.file_path is None

    async def test_llm_returns_empty_content(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        mock_llm_client.chat.return_value = LLMResult(
            content="   ",
            provider="test",
            model="test-model",
        )
        gen = DocsGenerator()
        section = DOC_SECTIONS[0]
        result = await gen._generate_section(
            section=section,
            project_name="TestProject",
            user_prompt="A test project",
            output_path=tmp_output_dir,
        )

        assert result.success is False
        assert result.error == "LLM returned empty content"

    async def test_llm_raises_exception(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        mock_llm_client.chat.side_effect = RuntimeError("Connection refused")
        gen = DocsGenerator()
        section = DOC_SECTIONS[0]
        result = await gen._generate_section(
            section=section,
            project_name="TestProject",
            user_prompt="A test project",
            output_path=tmp_output_dir,
        )

        assert result.success is False
        assert result.error == "Connection refused"

    async def test_no_file_written_on_failure(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        mock_llm_client.chat.return_value = LLMResult(
            content="",
            error="API error",
            provider="test",
            model="test-model",
        )
        gen = DocsGenerator()
        section = DOC_SECTIONS[0]
        await gen._generate_section(
            section=section,
            project_name="TestProject",
            user_prompt="A test project",
            output_path=tmp_output_dir,
        )

        # No file should be written on error
        file_path = tmp_output_dir / "01-prd.md"
        assert not file_path.exists()

    async def test_content_with_code_fences_is_stripped(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        mock_llm_client.chat.return_value = LLMResult(
            content="```markdown\n# Clean Heading\n\nClean content.\n```",
            provider="test",
            model="test-model",
        )
        gen = DocsGenerator()
        section = DOC_SECTIONS[0]
        result = await gen._generate_section(
            section=section,
            project_name="App",
            user_prompt="desc",
            output_path=tmp_output_dir,
        )

        assert result.success is True
        file_path = tmp_output_dir / "01-prd.md"
        written = file_path.read_text(encoding="utf-8")
        assert "```" not in written
        assert "# Clean Heading" in written
        assert "Clean content." in written


# ============================================================================
# DocsGenerator.generate_all
# ============================================================================


class TestGenerateAll:
    """Verify the generate_all method."""

    async def test_generates_all_sections(
        self,
        generator: DocsGenerator,
        tmp_output_dir: Path,
    ) -> None:
        results = await generator.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
        )

        assert len(results) == 8
        assert all(r.success for r in results)
        assert all(r.file_path is not None for r in results)

        # Verify all 8 files were created
        for section in DOC_SECTIONS:
            file_path = tmp_output_dir / section.filename
            assert file_path.exists(), f"Missing file: {file_path}"
            content = file_path.read_text(encoding="utf-8")
            assert f"title: {section.title}" in content

    async def test_generates_filtered_sections(
        self,
        generator: DocsGenerator,
        tmp_output_dir: Path,
    ) -> None:
        results = await generator.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
            sections=["prd", "specs"],
        )

        assert len(results) == 2
        assert results[0].section_id == "prd"
        assert results[1].section_id == "specs"

        # Only those two files should exist
        assert (tmp_output_dir / "01-prd.md").exists()
        assert (tmp_output_dir / "02-specs.md").exists()
        assert not (tmp_output_dir / "03-functional-memory.md").exists()

    async def test_invalid_section_ids_ignored(
        self,
        generator: DocsGenerator,
        tmp_output_dir: Path,
    ) -> None:
        results = await generator.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
            sections=["prd", "nonexistent-section"],
        )

        assert len(results) == 1
        assert results[0].section_id == "prd"

    async def test_empty_sections_list_generates_all(
        self,
        generator: DocsGenerator,
        tmp_output_dir: Path,
    ) -> None:
        results = await generator.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
            sections=[],
        )

        assert len(results) == 8

    async def test_creates_output_directory(
        self,
        generator: DocsGenerator,
        tmp_path: Path,
    ) -> None:
        new_dir = tmp_path / "new-docs" / "subdir"
        assert not new_dir.exists()

        results = await generator.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=new_dir,
        )

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert len(results) == 8

    async def test_partial_failure(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        """When some sections fail, others still succeed."""
        # Make the third call fail
        call_count = 0

        async def failing_chat(*args: Any, **kwargs: Any) -> LLMResult:
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # third section (functional-memory)
                return LLMResult(content="", error="Failed on purpose", provider="test", model="test-model")
            return LLMResult(
                content="# OK content",
                provider="test",
                model="test-model",
            )

        mock_llm_client.chat.side_effect = failing_chat
        gen = DocsGenerator()

        results = await gen.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
        )

        assert len(results) == 8
        # Section at index 2 (functional-memory) should have failed
        assert results[2].success is False
        assert results[2].error == "Failed on purpose"
        # All others should succeed
        assert all(r.success for i, r in enumerate(results) if i != 2)

    async def test_reports_correct_counts(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        """Verify char_count in results."""
        mock_llm_client.chat.return_value = LLMResult(
            content="# Content",
            provider="test",
            model="test-model",
        )
        gen = DocsGenerator()
        results = await gen.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
            sections=["prd"],
        )

        assert len(results) == 1
        assert results[0].success is True
        # char_count should include frontmatter
        assert results[0].char_count > 20


# ============================================================================
# DocsGenerator.generate_prd
# ============================================================================


class TestGeneratePRD:
    """Verify the generate_prd convenience method."""

    async def test_generates_only_prd(
        self,
        generator: DocsGenerator,
        tmp_output_dir: Path,
    ) -> None:
        result = await generator.generate_prd(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
        )

        assert result.section_id == "prd"
        assert result.filename == "01-prd.md"
        assert result.success is True

        # Only PRD file should exist
        assert (tmp_output_dir / "01-prd.md").exists()
        assert not (tmp_output_dir / "02-specs.md").exists()

    async def test_prd_creates_output_dir(
        self,
        generator: DocsGenerator,
        tmp_path: Path,
    ) -> None:
        new_dir = tmp_path / "prd-output"
        result = await generator.generate_prd(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=new_dir,
        )

        assert new_dir.exists()
        assert result.success is True
        assert (new_dir / "01-prd.md").exists()

    async def test_prd_failure(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        mock_llm_client.chat.return_value = LLMResult(
            content="",
            error="API error",
            provider="test",
            model="test-model",
            finish_reason="error",
        )
        gen = DocsGenerator()
        result = await gen.generate_prd(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
        )

        assert result.success is False
        assert result.error == "API error"
        assert not (tmp_output_dir / "01-prd.md").exists()


# ============================================================================
# generate_project_docs (convenience factory)
# ============================================================================


class TestGenerateProjectDocs:
    """Verify the generate_project_docs convenience function."""

    async def test_default_call(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        results = await generate_project_docs(
            project_name="MyProject",
            user_prompt="A cool project",
            output_dir=tmp_output_dir,
        )

        assert len(results) == 8
        assert all(r.success for r in results)
        mock_get_llm_client.assert_called_once_with(
            provider="hermes", model=None, api_key=None, profile="swarm1"
        )

    async def test_with_explicit_provider(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        results = await generate_project_docs(
            project_name="MyProject",
            user_prompt="A cool project",
            output_dir=tmp_output_dir,
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
        )

        assert len(results) == 8
        mock_get_llm_client.assert_called_once_with(
            provider="openai", model="gpt-4o", api_key="sk-test", profile="swarm1"
        )

    async def test_with_custom_profile(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        results = await generate_project_docs(
            project_name="MyProject",
            user_prompt="A cool project",
            output_dir=tmp_output_dir,
            profile="my-profile",
        )

        assert len(results) == 8
        mock_get_llm_client.assert_called_once_with(
            provider="hermes", model=None, api_key=None, profile="my-profile"
        )

    async def test_with_sections_filter(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        results = await generate_project_docs(
            project_name="MyProject",
            user_prompt="A cool project",
            output_dir=tmp_output_dir,
            sections=["prd", "roadmap"],
        )

        assert len(results) == 2
        assert results[0].section_id == "prd"
        assert results[1].section_id == "roadmap"

    async def test_passes_sections_to_generate_all(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        """Verify docs are written with correct filenames."""
        results = await generate_project_docs(
            project_name="MyProject",
            user_prompt="A cool project",
            output_dir=tmp_output_dir,
            sections=["specs"],
        )

        assert len(results) == 1
        assert results[0].filename == "02-specs.md"
        assert (tmp_output_dir / "02-specs.md").exists()


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Edge cases not covered above."""

    async def test_section_with_unicode_content(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        """Unicode content is handled correctly."""
        mock_llm_client.chat.return_value = LLMResult(
            content="# Café\n\nüber cool 😊\n\n- Résumé\n- 中文测试",
            provider="test",
            model="test-model",
        )
        gen = DocsGenerator()
        results = await gen.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=tmp_output_dir,
            sections=["prd"],
        )

        assert results[0].success is True
        content = (tmp_output_dir / "01-prd.md").read_text(encoding="utf-8")
        assert "Café" in content
        assert "über" in content
        assert "😊" in content
        assert "中文测试" in content

    async def test_generate_all_with_str_output_dir(
        self,
        generator: DocsGenerator,
        tmp_output_dir: Path,
    ) -> None:
        """output_dir can be a string, not just Path."""
        results = await generator.generate_all(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=str(tmp_output_dir),
        )

        assert len(results) == 8
        assert (tmp_output_dir / "01-prd.md").exists()

    async def test_generate_prd_with_str_output_dir(
        self,
        generator: DocsGenerator,
        tmp_output_dir: Path,
    ) -> None:
        """generate_prd also accepts string output_dir."""
        result = await generator.generate_prd(
            project_name="TestProject",
            user_prompt="A test project",
            output_dir=str(tmp_output_dir),
        )
        assert result.success is True

    def test_doc_section_dataclass(self) -> None:
        """DocSection is a proper dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(DocSection)
        s = DocSection(
            id="test",
            title="Test",
            filename="test.md",
            description="A test section",
            prompt_template="Generate {{project}}",
        )
        assert s.id == "test"
        assert s.title == "Test"
        assert s.filename == "test.md"
        assert s.description == "A test section"
        assert s.prompt_template == "Generate {{project}}"

    def test_get_llm_client_called_with_correct_args(
        self, mock_get_llm_client: MagicMock
    ) -> None:
        DocsGenerator(provider="openai", model="gpt-4", api_key="key123", profile="p1")
        mock_get_llm_client.assert_called_once_with(
            provider="openai", model="gpt-4", api_key="key123", profile="p1"
        )

    async def test_section_prompt_template_used(
        self,
        mock_llm_client: AsyncMock,
        mock_get_llm_client: MagicMock,
        tmp_output_dir: Path,
    ) -> None:
        """Verify the prompt template for each section is included in LLM call."""
        gen = DocsGenerator()
        section = DOC_SECTIONS[2]  # functional-memory
        await gen._generate_section(
            section=section,
            project_name="MyApp",
            user_prompt="A task manager",
            output_path=tmp_output_dir,
        )

        call_args = mock_llm_client.chat.call_args
        system_prompt = call_args.kwargs["messages"][0]["content"]
        # The prompt template should be embedded in the system message
        assert section.prompt_template in system_prompt
        assert "MyApp" in system_prompt
        assert "A task manager" in call_args.kwargs["messages"][1]["content"]

    async def test_generate_all_no_output_dir_creation_if_empty_sections(
        self,
        generator: DocsGenerator,
        tmp_path: Path,
    ) -> None:
        """Even with zero valid sections, the output dir is still created."""
        output_dir = tmp_path / "nonexistent"
        results = await generator.generate_all(
            project_name="TestProject",
            user_prompt="desc",
            output_dir=output_dir,
            sections=["does-not-exist"],
        )
        assert output_dir.exists()
        assert len(results) == 0
