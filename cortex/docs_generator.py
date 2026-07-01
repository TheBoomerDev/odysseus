"""
cortex/docs_generator.py — Automatic documentation generation for projects.

Uses an LLM to produce comprehensive project documents across 8 sections:
  - Product Requirements Document (PRD)
  - Technical Specifications
  - Functional Memory
  - Technical Memory
  - Roadmap & Future Improvements
  - Visual Identity & Design System
  - Buyer Persona & Target Audience
  - Competitor Analysis

Each section is saved as a markdown file under ``docs/`` in the project.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .codegen.llm import LLMClient, LLMResult, get_llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document section definitions
# ---------------------------------------------------------------------------


@dataclass
class DocSection:
    """A single document section to generate."""

    id: str
    title: str
    filename: str
    description: str
    prompt_template: str


DOC_SECTIONS: List[DocSection] = [
    DocSection(
        id="prd",
        title="Product Requirements Document (PRD)",
        filename="01-prd.md",
        description="Complete PRD covering problem statement, target audience, features, user stories, success metrics.",
        prompt_template="""Generate a comprehensive Product Requirements Document (PRD) for the project described below.

Include:
1. Problem statement and target audience
2. User personas (2-3)
3. Functional requirements (categorized)
4. Non-functional requirements
5. User stories (at least 5)
6. Success metrics and KPIs
7. Scope (in scope / out of scope)
8. Risks and mitigations

Format in clean Markdown with proper headings, bullet points, and tables where appropriate.
""",
    ),
    DocSection(
        id="specs",
        title="Technical Specifications",
        filename="02-specs.md",
        description="Technical architecture, stack decisions, data model, API design, security considerations.",
        prompt_template="""Generate a comprehensive Technical Specifications document for the project described below.

Include:
1. Technical architecture overview (with diagram description)
2. Technology stack decisions and rationale
3. Data model / database schema
4. API design (endpoints, request/response formats)
5. Security considerations
6. Performance considerations
7. Deployment architecture
8. Testing strategy

Be specific with technology names, version numbers, and concrete design decisions.
Format in clean Markdown with code blocks for schema definitions.
""",
    ),
    DocSection(
        id="functional-memory",
        title="Functional Memory",
        filename="03-functional-memory.md",
        description="How the system behaves from a user perspective: workflows, state machines, edge cases.",
        prompt_template="""Generate a Functional Memory document for the project described below.

Include:
1. Core user workflows (step-by-step)
2. System behavior descriptions
3. State machines and transitions
4. Edge cases and error handling
5. User-facing messages and notifications
6. Permission levels and access control

Describe how the system behaves from the user's perspective.
Format in clean Markdown with numbered steps and flow descriptions.
""",
    ),
    DocSection(
        id="technical-memory",
        title="Technical Memory",
        filename="04-technical-memory.md",
        description="Implementation details, dependencies, configuration, deployment, environment variables.",
        prompt_template="""Generate a Technical Memory document for the project described below.

Include:
1. Implementation details for each major component
2. All dependencies and their versions
3. Configuration options and environment variables
4. Build and deployment instructions
5. Required system dependencies
6. Monitoring and logging setup
7. Backup and recovery procedures
8. Known limitations and technical debt

Format in clean Markdown with code blocks for configuration examples.
""",
    ),
    DocSection(
        id="roadmap",
        title="Roadmap & Future Improvements",
        filename="05-roadmap.md",
        description="Phased development plan, v1 scope, v2 ambitions, potential pivots.",
        prompt_template="""Generate a Roadmap document for the project described below.

Include:
1. Phased development plan (v1, v2, v3)
2. v1 scope (MVP) — what is absolutely necessary
3. v2 ambitions — what comes next
4. v3+ potential — long-term vision
5. Potential pivots and alternatives
6. Technical debt roadmap
7. Innovation opportunities

Be realistic about timelines and priorities.
Format in clean Markdown with phase headings and milestone lists.
""",
    ),
    DocSection(
        id="visual-design",
        title="Visual Identity & Design System",
        filename="06-visual-design.md",
        description="Color palette, typography, layout principles, component design, logo concepts.",
        prompt_template="""Generate a Visual Identity and Design System document for the project described below.

Include:
1. Design direction and philosophy
2. Color palette (4-6 colors with hex codes and usage)
3. Typography (heading and body font pairs with sizes)
4. Spacing and layout principles
5. Component design patterns
6. Logo and iconography concepts
7. Dark mode considerations
8. Accessibility guidelines (contrast, focus states, etc.)

Reference shadcn/ui and Radix UI primitives where applicable.
Format in clean Markdown with hex codes, size specifications, and usage examples.
""",
    ),
    DocSection(
        id="buyer-persona",
        title="Buyer Persona & Target Audience",
        filename="07-buyer-persona.md",
        description="ICP definition, demographics, pain points, jobs-to-be-done, acquisition channels.",
        prompt_template="""Generate a Buyer Persona and Target Audience document for the project described below.

Include:
1. Ideal Customer Profile (ICP) definition
2. Primary persona (demographics, goals, pain points, behaviors)
3. Secondary persona (if applicable)
4. Jobs-to-be-done framework
5. Acquisition channels and marketing strategy
6. Pricing considerations
7. Competitive alternatives they currently use

Be specific and realistic about who would use this product.
Format in clean Markdown with persona profile sections.
""",
    ),
    DocSection(
        id="competitor-analysis",
        title="Competitor Analysis",
        filename="08-competitor-analysis.md",
        description="Market landscape, direct/indirect competitors, gaps identified, differentiation strategy.",
        prompt_template="""Generate a Competitor Analysis document for the project described below.

Include:
1. Market landscape overview
2. Direct competitors (3-5) with:
   - Company name and product
   - Key features and strengths
   - Weaknesses and gaps
   - Pricing model
   - Market position
3. Indirect competitors
4. Competitive gaps and opportunities
5. Differentiation strategy
6. SWOT analysis

Be specific with competitor names and features.
Format in clean Markdown with comparison tables.
""",
    ),
]


# ---------------------------------------------------------------------------
# Docs generator
# ---------------------------------------------------------------------------


class DocsGenerationError(Exception):
    """Raised when documentation generation fails."""


@dataclass
class DocsResult:
    """Result of generating a single document section."""

    section_id: str
    title: str
    filename: str
    success: bool
    error: Optional[str] = None
    file_path: Optional[str] = None
    char_count: int = 0


class DocsGenerator:
    """Generates project documentation using an LLM.

    Produces up to 8 markdown documents covering product, technical,
    design, and business aspects of a project.

    Usage:
        gen = DocsGenerator(provider="openai", model="gpt-4o", api_key="...")
        results = await gen.generate_all(
            project_name="My App",
            user_prompt="A task management app with kanban boards",
            output_dir="/path/to/docs",
        )
    """

    def __init__(
        self,
        provider: str = "hermes",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        profile: str = "swarm1",
        temperature: float = 0.3,
    ):
        self._llm = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            profile=profile,
        )
        self._provider = provider
        self._model = model or self._llm.model
        self._temperature = temperature

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def generate_all(
        self,
        project_name: str,
        user_prompt: str,
        output_dir: str | Path,
        sections: Optional[List[str]] = None,
    ) -> List[DocsResult]:
        """Generate all documentation sections.

        Args:
            project_name: Name of the project
            user_prompt: The original project description / user prompt
            output_dir: Directory to write markdown files
            sections: Optional list of section IDs to generate (default: all)

        Returns:
            List of DocsResult for each section
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine which sections to generate
        target_sections = DOC_SECTIONS
        if sections:
            section_map = {s.id: s for s in DOC_SECTIONS}
            target_sections = [
                section_map[sid] for sid in sections if sid in section_map
            ]

        results: List[DocsResult] = []

        for section in target_sections:
            logger.info(
                "Generating '%s' (%s) for '%s'...",
                section.id, section.filename, project_name,
            )

            result = await self._generate_section(
                section=section,
                project_name=project_name,
                user_prompt=user_prompt,
                output_path=output_path,
            )
            results.append(result)

            if not result.success:
                logger.warning(
                    "Section '%s' failed: %s", section.id, result.error
                )

        success_count = sum(1 for r in results if r.success)
        logger.info(
            "Docs generation complete: %d/%d sections generated for '%s'",
            success_count, len(target_sections), project_name,
        )
        return results

    async def generate_prd(
        self,
        project_name: str,
        user_prompt: str,
        output_dir: str | Path,
    ) -> DocsResult:
        """Generate only the PRD document (convenience method)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        section = DOC_SECTIONS[0]  # prd
        return await self._generate_section(
            section=section,
            project_name=project_name,
            user_prompt=user_prompt,
            output_path=output_path,
        )

    async def _generate_section(
        self,
        section: DocSection,
        project_name: str,
        user_prompt: str,
        output_path: Path,
    ) -> DocsResult:
        """Generate a single document section and save to disk."""
        system_prompt = (
            f"You are a senior product manager and technical architect. "
            f"You are helping build a project called \"{project_name}\".\n\n"
            f"Your task: {section.prompt_template}\n\n"
            f"Respond with valid Markdown content only. "
            f"Use proper heading hierarchy (##, ###, ####), bullet points, "
            f"and tables where appropriate."
        )

        user_content = (
            f"Project: {project_name}\n\n"
            f"Description: {user_prompt}\n\n"
            f"Generate the {section.title} section."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            result = await self._llm.chat(
                messages=messages,
                temperature=self._temperature,
                max_tokens=4096,
            )

            if result.error:
                return DocsResult(
                    section_id=section.id,
                    title=section.title,
                    filename=section.filename,
                    success=False,
                    error=result.error,
                )

            content = result.content.strip()
            if not content:
                return DocsResult(
                    section_id=section.id,
                    title=section.title,
                    filename=section.filename,
                    success=False,
                    error="LLM returned empty content",
                )

            # Add YAML frontmatter
            full_content = self._format_doc(section, project_name, content)

            # Write to file
            file_path = output_path / section.filename
            file_path.write_text(full_content, encoding="utf-8")

            logger.info(
                "Wrote %s (%d chars)", file_path, len(full_content)
            )
            return DocsResult(
                section_id=section.id,
                title=section.title,
                filename=section.filename,
                success=True,
                file_path=str(file_path),
                char_count=len(full_content),
            )

        except Exception as exc:
            logger.exception(
                "Failed to generate '%s': %s", section.id, exc
            )
            return DocsResult(
                section_id=section.id,
                title=section.title,
                filename=section.filename,
                success=False,
                error=str(exc),
            )

    @staticmethod
    def _format_doc(
        section: DocSection,
        project_name: str,
        content: str,
    ) -> str:
        """Wrap generated content with YAML frontmatter metadata."""
        now = datetime.now(timezone.utc).isoformat()

        # Remove any markdown code fences wrapping the entire content
        clean = content.strip()
        if clean.startswith("```"):
            # Find the second newline after opening fence
            first_nl = clean.find("\n")
            if first_nl != -1:
                clean = clean[first_nl + 1:]
            # Remove trailing fence
            if clean.endswith("```"):
                clean = clean[:-3].strip()
            elif "```" in clean:
                clean = clean.rsplit("```", 1)[0].strip()

        return f"""---
title: {section.title}
project: {project_name}
generated_at: {now}
generator: odysseus/docs_generator
provider: {getattr(section, '_provider', 'unknown')}
---

{clean}
"""


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


async def generate_project_docs(
    project_name: str,
    user_prompt: str,
    output_dir: str | Path,
    provider: str = "hermes",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    profile: str = "swarm1",
    sections: Optional[List[str]] = None,
) -> List[DocsResult]:
    """One-shot convenience function to generate all project docs.

    Args:
        project_name: Name of the project
        user_prompt: Original project description
        output_dir: Directory to write docs
        provider: LLM provider ('hermes', 'openai', 'gemini', etc.)
        model: Model name (required for direct providers)
        api_key: API key (required for cloud providers)
        profile: Hermes profile (only for provider='hermes')
        sections: Optional list of section IDs to generate

    Returns:
        List of DocsResult
    """
    gen = DocsGenerator(
        provider=provider,
        model=model,
        api_key=api_key,
        profile=profile,
    )
    return await gen.generate_all(
        project_name=project_name,
        user_prompt=user_prompt,
        output_dir=output_dir,
        sections=sections,
    )
