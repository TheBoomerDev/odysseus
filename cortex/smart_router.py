"""
cortex/smart_router.py — Multi-provider LLM SmartRouter.

Port of ADABA/codeWriter v2 SmartRouter.ts + ModelCostTable.ts.

Task-category-aware model selection with cost optimization and
automatic failover across providers. Integrates with Odysseus as
a recommendation engine for provider/model selection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost table
# ---------------------------------------------------------------------------

@dataclass
class ModelCostEntry:
    model: str
    provider: str
    input_per_1m: float   # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens


MODEL_COST_TABLE: Dict[str, ModelCostEntry] = {
    # DeepSeek
    "deepseek-chat": ModelCostEntry("deepseek-chat", "deepseek", 0.14, 0.28),
    "deepseek-reasoner": ModelCostEntry("deepseek-reasoner", "deepseek", 0.55, 2.19),

    # Anthropic
    "claude-sonnet-4": ModelCostEntry("claude-sonnet-4", "anthropic", 3.00, 15.00),
    "claude-opus-4": ModelCostEntry("claude-opus-4", "anthropic", 15.00, 75.00),
    "claude-sonnet-4-20250514": ModelCostEntry("claude-sonnet-4-20250514", "anthropic", 3.00, 15.00),
    "claude-haiku-3.5": ModelCostEntry("claude-haiku-3.5", "anthropic", 0.80, 4.00),

    # OpenAI
    "gpt-4o": ModelCostEntry("gpt-4o", "openai", 2.50, 10.00),
    "gpt-4o-mini": ModelCostEntry("gpt-4o-mini", "openai", 0.15, 0.60),
    "gpt-4-turbo": ModelCostEntry("gpt-4-turbo", "openai", 10.00, 30.00),
    "o1": ModelCostEntry("o1", "openai", 15.00, 60.00),
    "o1-mini": ModelCostEntry("o1-mini", "openai", 3.00, 12.00),

    # Google
    "gemini-2.0-flash": ModelCostEntry("gemini-2.0-flash", "google", 0.10, 0.40),
    "gemini-2.0-pro": ModelCostEntry("gemini-2.0-pro", "google", 2.00, 8.00),
    "gemini-1.5-flash": ModelCostEntry("gemini-1.5-flash", "google", 0.075, 0.30),

    # OpenRouter
    "openrouter/auto": ModelCostEntry("openrouter/auto", "openrouter", 1.00, 3.00),

    # Ollama (local — free)
    "ollama/llama3": ModelCostEntry("ollama/llama3", "ollama", 0.0, 0.0),
    "ollama/mistral": ModelCostEntry("ollama/mistral", "ollama", 0.0, 0.0),

    # Grok
    "grok-2": ModelCostEntry("grok-2", "grok", 2.00, 10.00),

    # xAI
    "grok-3": ModelCostEntry("grok-3", "xai", 3.00, 15.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a model call."""
    entry = MODEL_COST_TABLE.get(model)
    if not entry:
        return 0.0
    return (
        (input_tokens / 1_000_000) * entry.input_per_1m +
        (output_tokens / 1_000_000) * entry.output_per_1m
    )


# ---------------------------------------------------------------------------
# Task categories
# ---------------------------------------------------------------------------

class TaskCategory(str, Enum):
    ANALYSIS = "analysis"
    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    WRITING = "writing"
    TESTING = "testing"
    DECOMPOSITION = "decomposition"
    CHAT = "chat"


CATEGORY_LABELS = {
    TaskCategory.ANALYSIS: "Analysis",
    TaskCategory.PLANNING: "Planning",
    TaskCategory.CODE_GENERATION: "Code generation",
    TaskCategory.CODE_REVIEW: "Code review",
    TaskCategory.RESEARCH: "Research",
    TaskCategory.WRITING: "Writing",
    TaskCategory.TESTING: "Testing",
    TaskCategory.DECOMPOSITION: "Decomposition",
    TaskCategory.CHAT: "General chat",
}


class RoutingPriority(str, Enum):
    SPEED = "speed"
    QUALITY = "quality"
    COST = "cost"


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class SmartRouterResponse:
    content: str = ""
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    cached: bool = False


@dataclass
class ModelEntry:
    model: str
    reason: str


# ---------------------------------------------------------------------------
# Task → model mapping
# ---------------------------------------------------------------------------

TASK_MODEL_MAP: Dict[TaskCategory, List[ModelEntry]] = {
    # Cheap: analysis, planning, decomposition, research
    TaskCategory.ANALYSIS: [
        ModelEntry("deepseek-chat", "Bajo costo para tareas analíticas"),
    ],
    TaskCategory.PLANNING: [
        ModelEntry("deepseek-chat", "Suficiente para planificación"),
    ],
    TaskCategory.DECOMPOSITION: [
        ModelEntry("deepseek-chat", "Descomposición no requiere modelo caro"),
    ],
    TaskCategory.RESEARCH: [
        ModelEntry("deepseek-chat", "Investigación básica de bajo costo"),
    ],

    # Expensive: code generation & review
    TaskCategory.CODE_GENERATION: [
        ModelEntry("claude-sonnet-4", "Mejor para generación de código"),
        ModelEntry("gpt-4o", "Fallback: buena calidad de código"),
        ModelEntry("deepseek-chat", "Fallback: opción económica"),
    ],
    TaskCategory.CODE_REVIEW: [
        ModelEntry("claude-sonnet-4", "Calidad superior para revisión"),
        ModelEntry("gpt-4o", "Fallback: buena capacidad de revisión"),
    ],

    # Mid-range: writing
    TaskCategory.WRITING: [
        ModelEntry("gpt-4o-mini", "Bueno para escritura, costo moderado"),
    ],

    # Cheap: testing
    TaskCategory.TESTING: [
        ModelEntry("deepseek-chat", "Suficiente para generar tests"),
    ],

    # Cheap default: chat
    TaskCategory.CHAT: [
        ModelEntry("deepseek-chat", "Chat general de bajo costo"),
    ],
}


# ---------------------------------------------------------------------------
# Category detector
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[TaskCategory, List[str]] = {
    TaskCategory.ANALYSIS: ["analy", "assess", "evaluat", "investig", "diagnos",
                            "compar", "metrics", "perform", "impact"],
    TaskCategory.PLANNING: ["plan", "roadmap", "timeline", "schedule", "milestone",
                            "strategy", "sprint", "backlog"],
    TaskCategory.CODE_GENERATION: ["implement", "write code", "create function",
                                    "develop", "build", "program", "feature",
                                    "add endpoint", "script"],
    TaskCategory.CODE_REVIEW: ["review", "audit code", "check", "inspect",
                                "refactor", "quality"],
    TaskCategory.RESEARCH: ["research", "find", "search", "look up", "investigate",
                            "learn about", "what is", "how does"],
    TaskCategory.WRITING: ["write", "document", "draft", "compose", "edit",
                           "readme", "doc", "blog", "post"],
    TaskCategory.TESTING: ["test", "unit test", "integration test", "e2e",
                           "coverage", "assert", "mock"],
    TaskCategory.DECOMPOSITION: ["decompos", "break down", "split", "divide",
                                  "subtask", "task list"],
    TaskCategory.CHAT: ["chat", "talk", "hello", "hi", "help", "what can you"],
}


def detect_category(prompt: str) -> TaskCategory:
    """Detect the best task category from a prompt."""
    prompt_lower = prompt.lower()
    best_match = TaskCategory.CHAT
    best_score = 0

    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > best_score:
            best_score = score
            best_match = cat

    return best_match


# ---------------------------------------------------------------------------
# SmartRouter
# ---------------------------------------------------------------------------

class SmartRouter:
    """Task-category-aware model selection with cost optimization.

    Port of ADABA/codeWriter v2 SmartRouter.ts.

    This is a RECOMMENDATION engine — it suggests the best model/provider
    for a given prompt based on task category, priority, and cost.
    The actual LLM call is made by Odysseus's existing provider system.
    """

    def __init__(self):
        pass

    def select_model(
        self,
        category: TaskCategory,
        priority: RoutingPriority = RoutingPriority.QUALITY,
    ) -> str:
        """Select the best model for a task category based on priority."""
        models = TASK_MODEL_MAP.get(category, [])
        if not models:
            logger.warning("Unknown task category '%s', using default", category)
            return "deepseek-chat"

        if priority == RoutingPriority.QUALITY:
            return models[0].model
        elif priority == RoutingPriority.COST:
            return models[-1].model
        else:  # SPEED or default
            return models[0].model

    def get_models_for_category(self, category: TaskCategory) -> List[str]:
        """Get all models available for a task category."""
        return [e.model for e in TASK_MODEL_MAP.get(category, [])]

    def get_provider_for_model(self, model: str) -> str:
        """Get the provider name for a given model."""
        entry = MODEL_COST_TABLE.get(model)
        return entry.provider if entry else "unknown"

    def route(
        self,
        prompt: str,
        category: Optional[TaskCategory] = None,
        priority: RoutingPriority = RoutingPriority.QUALITY,
    ) -> Dict:
        """Route a prompt to the best model/provider combination.

        Returns a dict with recommendation and cost estimate.
        Does NOT make the actual LLM call — Odysseus handles that.
        """
        if category is None:
            category = detect_category(prompt)

        model = self.select_model(category, priority)
        provider = self.get_provider_for_model(model)

        # Estimate token counts (rough)
        prompt_tokens = max(1, len(prompt) // 4)
        estimated_output = 500  # assume ~500 output tokens
        cost = estimate_cost(model, prompt_tokens, estimated_output)

        fallbacks = [
            e.model for e in TASK_MODEL_MAP.get(category, [])
            if e.model != model
        ]

        return {
            "category": category.value,
            "label": CATEGORY_LABELS.get(category, "Unknown"),
            "recommended_model": model,
            "recommended_provider": provider,
            "estimated_cost_usd": round(cost, 6),
            "estimated_prompt_tokens": prompt_tokens,
            "fallback_models": fallbacks,
            "priority": priority.value,
        }

    def list_categories(self) -> List[Dict]:
        """List all task categories with their recommended models."""
        result = []
        for cat, models in TASK_MODEL_MAP.items():
            result.append({
                "category": cat.value,
                "label": CATEGORY_LABELS.get(cat, "Unknown"),
                "primary_model": models[0].model if models else None,
                "fallback_models": [m.model for m in models[1:]],
                "cost_tier": self._cost_tier(cat),
            })
        return result

    def _cost_tier(self, category: TaskCategory) -> str:
        """Determine cost tier for a category."""
        models = TASK_MODEL_MAP.get(category, [])
        if not models:
            return "unknown"
        primary = models[0].model
        entry = MODEL_COST_TABLE.get(primary)
        if not entry:
            return "unknown"
        total_cost = entry.input_per_1m + entry.output_per_1m
        if total_cost == 0:
            return "free"
        if total_cost < 1.0:
            return "cheap"
        if total_cost < 10.0:
            return "moderate"
        return "expensive"
