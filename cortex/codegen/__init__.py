"""
cortex/codegen/ — Code generation engine for Odysseus.

Generates projects, pages, and code from natural language descriptions
using a two-stage pipeline (Plan to Tool Loop) with pluggable LLM providers.

Capabilities:
  - Project scaffolding (Next.js, Express, minimal)
  - File patching (create/replace/edit/delete with search/replace)
  - Multi-provider LLM support (Hermes Agent, OpenAI, Gemini, Ollama)
  - Intelligent context loading (40KB budget, priority system)
  - Documentation generation (PRD, specs, roadmap)
"""

from . import prompts
from . import patcher
from . import tools
from . import starter
from . import llm
from . import context
from . import engine

__all__ = [
    "prompts",
    "patcher",
    "tools",
    "starter",
    "llm",
    "context",
    "engine",
]
