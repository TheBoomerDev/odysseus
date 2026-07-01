"""
Tests for cortex/codegen/prompts.py — Per-family prompt registry.

Achieves >95% coverage by testing all families, stages, fallback logic,
error paths, and legacy aliases. No mocking needed; all tests use the
real module data.
"""

from __future__ import annotations

import pytest

from cortex.codegen.prompts import (
    CODEGEN_SYSTEM_PROMPT,
    DESIGN_VIBE_PROMPT,
    MVP_SCOPE_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PRD_SYSTEM_PROMPT,
    _DEFAULT_FAMILY,
    _REGISTRY,
    get_prompt,
    known_families,
    known_stages,
)


# =========================================================================
# Families and stages known to the registry
# =========================================================================

EXPECTED_FAMILIES = ["openai-chat", "gemini", "openai-reasoning", "ollama"]

# (family, stages) pairs with expected stage lists
FAMILY_STAGES: dict[str, list[str]] = {
    "openai-chat": ["planner", "codegen", "prd", "mvp_scope", "design_vibe"],
    "gemini": ["planner", "codegen", "prd", "mvp_scope", "design_vibe"],
    "openai-reasoning": ["planner", "codegen"],
    "ollama": ["planner", "codegen", "prd", "mvp_scope"],
}


# =========================================================================
# known_families
# =========================================================================


class TestKnownFamilies:
    """known_families() returns the correct list of registered families."""

    def test_returns_expected_families(self) -> None:
        result = known_families()
        assert result == EXPECTED_FAMILIES

    def test_matches_registry_keys(self) -> None:
        assert known_families() == list(_REGISTRY.keys())

    def test_returns_new_list_each_call(self) -> None:
        result1 = known_families()
        result2 = known_families()
        assert result1 is not result2  # not the same object
        assert result1 == result2


# =========================================================================
# known_stages
# =========================================================================


class TestKnownStages:
    """known_stages(family) returns stages for a family, falling back."""

    @pytest.mark.parametrize("family,expected", FAMILY_STAGES.items())
    def test_known_family_returns_expected_stages(
        self, family: str, expected: list[str]
    ) -> None:
        assert known_stages(family) == expected

    def test_unknown_family_falls_back_to_default(self) -> None:
        """An unrecognised family falls back to the default family's stages."""
        result = known_stages("no-such-family")
        assert result == known_stages(_DEFAULT_FAMILY)
        assert result == FAMILY_STAGES[_DEFAULT_FAMILY]

    def test_empty_string_falls_back_to_default(self) -> None:
        """Empty string is not a registered family, so fallback applies."""
        result = known_stages("")
        assert result == FAMILY_STAGES[_DEFAULT_FAMILY]

    def test_returns_new_list_each_call(self) -> None:
        r1 = known_stages("openai-chat")
        r2 = known_stages("openai-chat")
        assert r1 is not r2
        assert r1 == r2


# =========================================================================
# get_prompt — all valid family+stage combinations
# =========================================================================


class TestGetPromptValidCombinations:
    """get_prompt(family, stage) returns a non-empty string for every
    registered family+stage pair."""

    @pytest.mark.parametrize(
        "family,stage",
        [
            (f, s)
            for f, stages in FAMILY_STAGES.items()
            for s in stages
        ],
    )
    def test_returns_non_empty_string(self, family: str, stage: str) -> None:
        prompt = get_prompt(family, stage)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.parametrize(
        "family,stage",
        [
            (f, s)
            for f, stages in FAMILY_STAGES.items()
            for s in stages
        ],
    )
    def test_prompt_contains_stage_keyword(
        self, family: str, stage: str
    ) -> None:
        """Each prompt is long enough to be meaningful (>= 50 chars) and
        contains at least one keyword related to its stage."""
        prompt = get_prompt(family, stage)
        assert len(prompt) >= 50
        # Check for a stage-relevant keyword rather than the literal key
        keywords = {
            "planner": ["plan"],
            "codegen": ["code", "generator", "typescript", "next.js"],
            "prd": ["product", "prd", "requirements"],
            "mvp_scope": ["mvp", "scope", "features"],
            "design_vibe": ["design", "ui", "visual", "color"],
        }
        assert any(kw in prompt.lower() for kw in keywords.get(stage, [stage]))


# =========================================================================
# get_prompt — fallback for unknown families
# =========================================================================


class TestGetPromptFallback:
    """When family is not in the registry, get_prompt falls back to the
    default family ("openai-chat")."""

    @pytest.mark.parametrize("stage", FAMILY_STAGES[_DEFAULT_FAMILY])
    def test_unknown_family_fallback(self, stage: str) -> None:
        result = get_prompt("completely-unknown-family", stage)
        expected = get_prompt(_DEFAULT_FAMILY, stage)
        assert result == expected

    @pytest.mark.parametrize("stage", FAMILY_STAGES[_DEFAULT_FAMILY])
    def test_empty_string_family_fallback(self, stage: str) -> None:
        result = get_prompt("", stage)
        expected = get_prompt(_DEFAULT_FAMILY, stage)
        assert result == expected

    @pytest.mark.parametrize("stage", FAMILY_STAGES[_DEFAULT_FAMILY])
    def test_numeric_family_fallback(self, stage: str) -> None:
        """A number-as-string that is not a key should also fall back."""
        result = get_prompt("12345", stage)
        expected = get_prompt(_DEFAULT_FAMILY, stage)
        assert result == expected


# =========================================================================
# get_prompt — KeyError for unknown stages
# =========================================================================


class TestGetPromptUnknownStage:
    """Asking for a stage that doesn't exist in a family raises KeyError."""

    @pytest.mark.parametrize("family", EXPECTED_FAMILIES)
    def test_unknown_stage_raises_key_error(self, family: str) -> None:
        with pytest.raises(KeyError):
            get_prompt(family, "stage_that_does_not_exist")

    def test_unknown_stage_in_default_family_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_prompt(_DEFAULT_FAMILY, "bogus_stage")

    def test_unknown_stage_after_fallback_raises_key_error(self) -> None:
        """Fallback family doesn't have the unknown stage either → KeyError."""
        with pytest.raises(KeyError):
            get_prompt("nonexistent-family", "stage_that_does_not_exist")

    def test_empty_string_stage_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_prompt("openai-chat", "")

    def test_error_message_contains_stage_name(self) -> None:
        try:
            get_prompt("openai-chat", "this-stage-does-not-exist")
        except KeyError as exc:
            assert "this-stage-does-not-exist" in str(exc)


# =========================================================================
# get_prompt — structural integrity of returned prompts
# =========================================================================


class TestGetPromptStructure:
    """Basic sanity checks on returned prompt content."""

    def test_planner_contains_plan(self) -> None:
        prompt = get_prompt("openai-chat", "planner")
        assert "plan" in prompt.lower()

    def test_codegen_contains_stack_ref(self) -> None:
        prompt = get_prompt("openai-chat", "codegen")
        assert "TypeScript" in prompt or "React" in prompt

    def test_prd_contains_product(self) -> None:
        prompt = get_prompt("openai-chat", "prd")
        assert "product" in prompt.lower()

    def test_mvp_scope_contains_mvp(self) -> None:
        prompt = get_prompt("openai-chat", "mvp_scope")
        assert "mvp" in prompt.lower()

    def test_design_vibe_contains_ui_or_design(self) -> None:
        prompt = get_prompt("openai-chat", "design_vibe")
        assert "ui" in prompt.lower() or "design" in prompt.lower()

    def test_different_families_same_stage_are_different(self) -> None:
        """Ensure gemini planner differs from openai-chat planner."""
        p1 = get_prompt("openai-chat", "planner")
        p2 = get_prompt("gemini", "planner")
        assert p1 != p2, "Family-specific prompts should differ"

    def test_different_stages_same_family_are_different(self) -> None:
        p1 = get_prompt("openai-chat", "planner")
        p2 = get_prompt("openai-chat", "codegen")
        assert p1 != p2


# =========================================================================
# Legacy aliases
# =========================================================================


class TestLegacyAliases:
    """Backward-compatibility module-level constants."""

    def test_planner_system_prompt_exists(self) -> None:
        assert isinstance(PLANNER_SYSTEM_PROMPT, str)
        assert len(PLANNER_SYSTEM_PROMPT) > 0

    def test_codegen_system_prompt_exists(self) -> None:
        assert isinstance(CODEGEN_SYSTEM_PROMPT, str)
        assert len(CODEGEN_SYSTEM_PROMPT) > 0

    def test_prd_system_prompt_exists(self) -> None:
        assert isinstance(PRD_SYSTEM_PROMPT, str)
        assert len(PRD_SYSTEM_PROMPT) > 0

    def test_mvp_scope_prompt_exists(self) -> None:
        assert isinstance(MVP_SCOPE_PROMPT, str)
        assert len(MVP_SCOPE_PROMPT) > 0

    def test_design_vibe_prompt_exists(self) -> None:
        assert isinstance(DESIGN_VIBE_PROMPT, str)
        assert len(DESIGN_VIBE_PROMPT) > 0

    @pytest.mark.parametrize(
        "alias,stage",
        [
            (PLANNER_SYSTEM_PROMPT, "planner"),
            (CODEGEN_SYSTEM_PROMPT, "codegen"),
            (PRD_SYSTEM_PROMPT, "prd"),
            (MVP_SCOPE_PROMPT, "mvp_scope"),
            (DESIGN_VIBE_PROMPT, "design_vibe"),
        ],
    )
    def test_alias_matches_get_prompt(
        self, alias: str, stage: str
    ) -> None:
        """Each legacy alias equals get_prompt('openai-chat', stage)."""
        assert alias == get_prompt("openai-chat", stage)

    def test_all_aliases_are_distinct(self) -> None:
        aliases = {
            PLANNER_SYSTEM_PROMPT,
            CODEGEN_SYSTEM_PROMPT,
            PRD_SYSTEM_PROMPT,
            MVP_SCOPE_PROMPT,
            DESIGN_VIBE_PROMPT,
        }
        assert len(aliases) == 5


# =========================================================================
# _DEFAULT_FAMILY constant
# =========================================================================


class TestDefaultFamily:
    def test_default_family_is_in_registry(self) -> None:
        assert _DEFAULT_FAMILY in _REGISTRY

    def test_default_family_is_openai_chat(self) -> None:
        assert _DEFAULT_FAMILY == "openai-chat"


# =========================================================================
# _REGISTRY structure
# =========================================================================


class TestRegistryStructure:
    """Verify the internal registry is well-formed."""

    def test_registry_is_dict_of_dicts(self) -> None:
        assert isinstance(_REGISTRY, dict)
        for family, stages in _REGISTRY.items():
            assert isinstance(family, str)
            assert isinstance(stages, dict)
            for stage, prompt in stages.items():
                assert isinstance(stage, str)
                assert isinstance(prompt, str)

    def test_all_prompts_are_non_empty(self) -> None:
        for family, stages in _REGISTRY.items():
            for stage, prompt in stages.items():
                assert len(prompt) > 0, (
                    f"Empty prompt for {family}/{stage}"
                )

    def test_openai_chat_has_all_five_stages(self) -> None:
        assert set(_REGISTRY["openai-chat"].keys()) == {
            "planner", "codegen", "prd", "mvp_scope", "design_vibe"
        }

    def test_gemini_has_all_five_stages(self) -> None:
        assert set(_REGISTRY["gemini"].keys()) == {
            "planner", "codegen", "prd", "mvp_scope", "design_vibe"
        }

    def test_openai_reasoning_has_two_stages(self) -> None:
        assert set(_REGISTRY["openai-reasoning"].keys()) == {
            "planner", "codegen"
        }

    def test_ollama_has_four_stages(self) -> None:
        assert set(_REGISTRY["ollama"].keys()) == {
            "planner", "codegen", "prd", "mvp_scope"
        }

    def test_family_count(self) -> None:
        assert len(_REGISTRY) == 4

    def test_no_extra_families(self) -> None:
        assert set(_REGISTRY.keys()) == set(EXPECTED_FAMILIES)
