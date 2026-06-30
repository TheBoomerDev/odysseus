"""
cortex/csuite.py — C-Suite Executive Agents.

Port of ADABA/codeWriter v2 C-Suite subagents (CEO, CTO, CMO, CFO, QA, R&D).

Each agent has specific capabilities and produces structured output.
They share company context and can be queried independently.
Integrates with Odysseus as tools available via /api/cortex/csuite/*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------


class CSuiteRole(str, Enum):
    CEO = "ceo"
    CTO = "cto"
    CMO = "cmo"
    CFO = "cfo"
    QA = "qa"
    RD = "rd"  # Research & Development


ROLE_INFO = {
    CSuiteRole.CEO: {
        "name": "Chief Executive Officer",
        "description": "Strategic vision and high-level decision making",
        "capabilities": [
            "define_vision",
            "create_strategies",
            "set_priorities",
            "make_decisions",
            "provide_recommendations",
        ],
        "system_prompt": """You are the CEO of a technology company. Your role is to:
- Define and communicate the company's vision and strategy
- Make high-level decisions about product direction and priorities
- Evaluate trade-offs between different approaches
- Provide clear, actionable recommendations
- Consider market position, competitive landscape, and business impact""",
    },
    CSuiteRole.CTO: {
        "name": "Chief Technology Officer",
        "description": "Technical architecture and engineering decisions",
        "capabilities": [
            "design_architecture",
            "evaluate_tech_stack",
            "assess_technical_risk",
            "plan_engineering_timelines",
            "review_code_quality",
        ],
        "system_prompt": """You are the CTO of a technology company. Your role is to:
- Design and evaluate technical architecture
- Make technology stack decisions
- Assess technical risks and mitigation strategies
- Plan engineering timelines and resource allocation
- Ensure code quality, scalability, and maintainability
- Consider technical debt and long-term system health""",
    },
    CSuiteRole.CMO: {
        "name": "Chief Marketing Officer",
        "description": "Marketing strategy and go-to-market planning",
        "capabilities": [
            "define_marketing_strategy",
            "plan_go_to_market",
            "analyze_audience",
            "create_content_strategy",
            "measure_campaign_effectiveness",
        ],
        "system_prompt": """You are the CMO of a technology company. Your role is to:
- Define marketing strategy and positioning
- Plan go-to-market launches
- Analyze target audience and market segments
- Create content and channel strategies
- Measure and optimize campaign effectiveness
- Align marketing with product and sales goals""",
    },
    CSuiteRole.CFO: {
        "name": "Chief Financial Officer",
        "description": "Financial planning and budget management",
        "capabilities": [
            "plan_budget",
            "analyze_unit_economics",
            "forecast_revenue",
            "assess_financial_risk",
            "optimize_costs",
        ],
        "system_prompt": """You are the CFO of a technology company. Your role is to:
- Plan and manage budgets across departments
- Analyze unit economics and profitability
- Forecast revenue and cash flow
- Assess financial risks and opportunities
- Optimize costs while maintaining growth
- Evaluate ROI of strategic initiatives""",
    },
    CSuiteRole.QA: {
        "name": "Quality Assurance",
        "description": "Quality standards and testing strategy",
        "capabilities": [
            "define_quality_standards",
            "plan_testing_strategy",
            "identify_risks",
            "review_test_coverage",
            "assess_release_readiness",
        ],
        "system_prompt": """You are the QA Director of a technology company. Your role is to:
- Define and enforce quality standards
- Plan testing strategy (unit, integration, e2e, manual)
- Identify quality risks and bottlenecks
- Review test coverage and effectiveness
- Assess release readiness and sign-off criteria
- Advocate for quality throughout the development process""",
    },
    CSuiteRole.RD: {
        "name": "Research & Development",
        "description": "Innovation, research, and competitive analysis",
        "capabilities": [
            "analyze_competitors",
            "research_technologies",
            "identify_trends",
            "propose_innovations",
            "evaluate_partnerships",
        ],
        "system_prompt": """You are the R&D Director of a technology company. Your role is to:
- Analyze competitors and their strategies
- Research emerging technologies and trends
- Identify innovation opportunities
- Propose new products, features, or improvements
- Evaluate potential partnerships and acquisitions
- Keep the company at the forefront of technology""",
    },
}


# ---------------------------------------------------------------------------
# Company context
# ---------------------------------------------------------------------------


@dataclass
class CompanyContext:
    name: str = "Technology Company"
    description: str = ""
    industry: str = "technology"
    stage: str = "growth"  # startup, growth, mature
    team_size: int = 10
    tech_stack: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    target_market: str = ""
    revenue_model: str = ""
    goals: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# C-Suite Orchestrator
# ---------------------------------------------------------------------------


class CSuiteOrchestrator:
    """Orchestrates C-Suite executive agents.

    Each agent produces structured output in its domain.
    Agents share company context for consistency.
    """

    def __init__(self, context: Optional[CompanyContext] = None):
        self.context = context or CompanyContext()
        self._history: Dict[CSuiteRole, List[Dict]] = {role: [] for role in CSuiteRole}

    def update_context(self, context: CompanyContext) -> None:
        """Update the shared company context."""
        self.context = context

    def get_role_info(self, role: CSuiteRole) -> Dict:
        """Get information about a specific role."""
        info = ROLE_INFO.get(role)
        if not info:
            raise ValueError(f"Unknown role: {role}")
        return {
            "role": role.value,
            "name": info["name"],
            "description": info["description"],
            "capabilities": info["capabilities"],
            "system_prompt": info["system_prompt"],
        }

    def list_roles(self) -> List[Dict]:
        """List all available C-Suite roles."""
        return [
            {
                "role": role.value,
                "name": info["name"],
                "description": info["description"],
                "capabilities": info["capabilities"],
            }
            for role, info in ROLE_INFO.items()
        ]

    def prepare_prompt(
        self,
        role: CSuiteRole,
        question: str,
    ) -> Dict:
        """Prepare a prompt for a C-Suite agent.

        Returns the system prompt + context for Odysseus to use
        with its existing LLM infrastructure.
        """
        info = ROLE_INFO.get(role)
        if not info:
            raise ValueError(f"Unknown role: {role}")

        context_str = self._render_context()
        system_prompt = info["system_prompt"]
        user_prompt = (
            f"{context_str}\n\n"
            f"## Question\n{question}\n\n"
            f"Please provide your analysis and recommendations based on "
            f"your role as {info['name']}."
        )

        return {
            "role": role.value,
            "role_name": info["name"],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "capabilities": info["capabilities"],
        }

    def record_response(
        self,
        role: CSuiteRole,
        question: str,
        response: str,
    ) -> None:
        """Record a C-Suite response for history/context."""
        self._history[role].append(
            {
                "question": question,
                "response": response,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
        )

    def get_history(self, role: Optional[CSuiteRole] = None) -> Dict:
        """Get C-Suite interaction history."""
        if role:
            return {role.value: self._history.get(role, [])}
        return {r.value: h for r, h in self._history.items()}

    def _render_context(self) -> str:
        """Render company context as text."""
        ctx = self.context
        parts = [
            "## Company Context",
            f"**Company:** {ctx.name}",
            f"**Industry:** {ctx.industry}",
            f"**Stage:** {ctx.stage}",
            f"**Team Size:** {ctx.team_size}",
        ]
        if ctx.description:
            parts.append(f"**Description:** {ctx.description}")
        if ctx.tech_stack:
            parts.append(f"**Tech Stack:** {', '.join(ctx.tech_stack)}")
        if ctx.products:
            parts.append(f"**Products:** {', '.join(ctx.products)}")
        if ctx.target_market:
            parts.append(f"**Target Market:** {ctx.target_market}")
        if ctx.revenue_model:
            parts.append(f"**Revenue Model:** {ctx.revenue_model}")
        if ctx.goals:
            parts.append("**Goals:**")
            for g in ctx.goals:
                parts.append(f"- {g}")
        return "\n".join(parts)
