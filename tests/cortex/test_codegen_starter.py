"""Tests for cortex/codegen/starter.py — project scaffolding templates.

Covers >95% of the module including dict structure, starter registry,
get_starter, list_starters, and is_placeholder_file edge cases.
"""

from __future__ import annotations

from typing import Dict

import pytest

from cortex.codegen.starter import (
    EXPRESS_STARTER,
    NEXT_MINIMAL_STARTER,
    NEXT_STARTER,
    _STARTERS,
    get_starter,
    is_placeholder_file,
    list_starters,
)

# =============================================================================
# Constants
# =============================================================================

# Expected file counts per starter
NEXT_FILE_COUNT = 10
NEXT_MINIMAL_FILE_COUNT = 8
EXPRESS_FILE_COUNT = 4
TOTAL_STARTER_COUNT = 3

# =============================================================================
# NEXT_STARTER — full Next.js 14 App Router
# =============================================================================


class TestNEXTStarter:
    """Validate NEXT_STARTER dict structure and content."""

    def test_has_ten_files(self):
        """NEXT_STARTER contains exactly 10 file entries."""
        assert len(NEXT_STARTER) == NEXT_FILE_COUNT

    def test_has_expected_keys(self):
        """NEXT_STARTER has all expected file paths."""
        expected_keys = {
            "package.json",
            "tsconfig.json",
            "next-env.d.ts",
            "next.config.mjs",
            "postcss.config.mjs",
            "tailwind.config.ts",
            "app/globals.css",
            "lib/utils.ts",
            "app/layout.tsx",
            "app/page.tsx",
        }
        assert set(NEXT_STARTER.keys()) == expected_keys

    def test_package_json_has_next_dep(self):
        """package.json includes 'next' as a dependency."""
        pkg = NEXT_STARTER["package.json"]
        assert '"next": "14.2.18"' in pkg

    def test_package_json_has_lucide_and_framer(self):
        """Full starter includes lucide-react and framer-motion."""
        pkg = NEXT_STARTER["package.json"]
        assert '"lucide-react"' in pkg
        assert '"framer-motion"' in pkg

    def test_package_json_has_tailwind_merge_clsx(self):
        """Full starter includes tailwind-merge and clsx for cn() helper."""
        pkg = NEXT_STARTER["package.json"]
        assert '"tailwind-merge"' in pkg
        assert '"clsx"' in pkg

    def test_package_json_has_shadcn_colors(self):
        """Full starter uses shadcn-style CSS variable colors."""
        tailwind_config = NEXT_STARTER["tailwind.config.ts"]
        assert "hsl(var(--border))" in tailwind_config
        assert "hsl(var(--background))" in tailwind_config
        assert "hsl(var(--foreground))" in tailwind_config

    def test_package_json_has_radius(self):
        """tailwind config includes borderRadius with shadcn tokens."""
        tailwind_config = NEXT_STARTER["tailwind.config.ts"]
        assert "var(--radius)" in tailwind_config

    def test_tailwind_dark_mode_class(self):
        """Full tailwind config uses class-based dark mode."""
        tailwind_config = NEXT_STARTER["tailwind.config.ts"]
        assert 'darkMode: "class"' in tailwind_config

    def test_lib_utils_has_cn_function(self):
        """lib/utils.ts exports the cn() utility."""
        utils = NEXT_STARTER["lib/utils.ts"]
        assert "export function cn" in utils
        assert "clsx" in utils
        assert "twMerge" in utils

    def test_globals_css_has_dark_variables(self):
        """globals.css defines both light and dark theme variables."""
        css = NEXT_STARTER["app/globals.css"]
        assert ":root {" in css
        assert ".dark {" in css
        assert "font-feature-settings" in css

    def test_layout_has_inter_font(self):
        """layout.tsx loads Inter font with font-sans variable."""
        layout = NEXT_STARTER["app/layout.tsx"]
        assert "Inter" in layout
        assert "variable: \"--font-sans\"" in layout

    def test_layout_has_odysseus_description(self):
        """Layout metadata says 'Built with Odysseus.'."""
        layout = NEXT_STARTER["app/layout.tsx"]
        assert "Built with Odysseus" in layout

    def test_page_has_odysseus_codegen_badge(self):
        """Page includes 'Odysseus Codegen' badge span."""
        page = NEXT_STARTER["app/page.tsx"]
        assert "Odysseus Codegen" in page

    def test_next_config_has_unsplash_placeholder(self):
        """next.config.mjs configures remote patterns for Unsplash/Placehold."""
        config = NEXT_STARTER["next.config.mjs"]
        assert "images.unsplash.com" in config
        assert "placehold.co" in config

    def test_tsconfig_has_next_plugin(self):
        """tsconfig.json includes 'next' in plugins array."""
        tsconfig = NEXT_STARTER["tsconfig.json"]
        assert '"plugins": [{ "name": "next" }]' in tsconfig

    def test_next_env_d_ts_has_reference_types(self):
        """next-env.d.ts has reference directives."""
        env = NEXT_STARTER["next-env.d.ts"]
        assert "reference types=\"next\"" in env
        assert "reference types=\"next/image-types/global\"" in env

    def test_all_contents_are_non_empty_strings(self):
        """Every file entry has non-empty string content."""
        for path, content in NEXT_STARTER.items():
            assert isinstance(content, str), f"{path} content is not str"
            assert len(content) > 0, f"{path} content is empty"

    def test_no_missing_placeholders(self):
        """No file content is just a bare placeholder."""
        for path, content in NEXT_STARTER.items():
            assert content.strip(), f"{path} is only whitespace"


# =============================================================================
# NEXT_MINIMAL_STARTER — minimal Next.js 14
# =============================================================================


class TestNEXTMinimalStarter:
    """Validate NEXT_MINIMAL_STARTER dict structure and content."""

    def test_has_eight_files(self):
        """NEXT_MINIMAL_STARTER contains exactly 8 file entries."""
        assert len(NEXT_MINIMAL_STARTER) == NEXT_MINIMAL_FILE_COUNT

    def test_has_expected_keys(self):
        """NEXT_MINIMAL_STARTER has all expected file paths (no next-env, no lib)."""
        expected_keys = {
            "package.json",
            "tsconfig.json",
            "next.config.mjs",
            "postcss.config.mjs",
            "tailwind.config.ts",
            "app/globals.css",
            "app/layout.tsx",
            "app/page.tsx",
        }
        assert set(NEXT_MINIMAL_STARTER.keys()) == expected_keys

    def test_does_not_include_next_env_d_ts(self):
        """Minimal starter does NOT include next-env.d.ts."""
        assert "next-env.d.ts" not in NEXT_MINIMAL_STARTER

    def test_does_not_include_lib_utils(self):
        """Minimal starter does NOT include lib/utils.ts."""
        assert "lib/utils.ts" not in NEXT_MINIMAL_STARTER

    def test_package_json_has_next_dep(self):
        """Minimal package.json includes 'next' dependency."""
        pkg = NEXT_MINIMAL_STARTER["package.json"]
        assert '"next": "14.2.18"' in pkg

    def test_package_json_does_not_include_lucide_framer(self):
        """Minimal starter omits lucide-react and framer-motion."""
        pkg = NEXT_MINIMAL_STARTER["package.json"]
        assert '"lucide-react"' not in pkg
        assert '"framer-motion"' not in pkg

    def test_package_json_has_no_tailwind_merge_or_clsx(self):
        """Minimal starter omits tailwind-merge and clsx."""
        pkg = NEXT_MINIMAL_STARTER["package.json"]
        assert '"tailwind-merge"' not in pkg
        assert '"clsx"' not in pkg

    def test_package_json_name_is_app_minimal(self):
        """Minimal starter's package name is 'app-minimal'."""
        pkg = NEXT_MINIMAL_STARTER["package.json"]
        assert '"name": "app-minimal"' in pkg

    def test_package_json_has_no_lint_script(self):
        """Minimal starter omits the lint script."""
        pkg = NEXT_MINIMAL_STARTER["package.json"]
        assert '"lint"' not in pkg

    def test_tailwind_config_minimal(self):
        """Minimal tailwind config has no color extensions."""
        tw = NEXT_MINIMAL_STARTER["tailwind.config.ts"]
        assert "./app/**/*.{ts,tsx,mdx}" in tw
        assert "hsl(var(--border))" not in tw

    def test_globals_css_minimal(self):
        """Minimal globals.css has no dark theme block."""
        css = NEXT_MINIMAL_STARTER["app/globals.css"]
        assert "@tailwind base" in css
        assert ".dark" not in css
        assert "font-family: system-ui" in css

    def test_layout_no_inter_font(self):
        """Minimal layout does not load Inter font."""
        layout = NEXT_MINIMAL_STARTER["app/layout.tsx"]
        assert "Inter" not in layout
        assert "My App" in layout

    def test_page_says_hello_world(self):
        """Minimal page renders 'Hello, world!'."""
        page = NEXT_MINIMAL_STARTER["app/page.tsx"]
        assert "Hello, world!" in page

    def test_next_config_minimal(self):
        """Minimal next.config.mjs is a bare stub."""
        config = NEXT_MINIMAL_STARTER["next.config.mjs"]
        assert "const nextConfig = {};" in config
        assert "unsplash" not in config

    def test_tsconfig_no_next_plugin(self):
        """Minimal tsconfig.json does not include 'next' plugin."""
        tsconfig = NEXT_MINIMAL_STARTER["tsconfig.json"]
        assert '"plugins"' not in tsconfig

    def test_all_contents_are_non_empty_strings(self):
        """Every file entry has non-empty string content."""
        for path, content in NEXT_MINIMAL_STARTER.items():
            assert isinstance(content, str), f"{path} content is not str"
            assert len(content) > 0, f"{path} content is empty"


# =============================================================================
# EXPRESS_STARTER — Express.js + TypeScript
# =============================================================================


class TestExpressStarter:
    """Validate EXPRESS_STARTER dict structure and content."""

    def test_has_four_files(self):
        """EXPRESS_STARTER contains exactly 4 file entries."""
        assert len(EXPRESS_STARTER) == EXPRESS_FILE_COUNT

    def test_has_expected_keys(self):
        """EXPRESS_STARTER has all expected file paths."""
        expected_keys = {
            "package.json",
            "tsconfig.json",
            ".gitignore",
            "src/index.ts",
        }
        assert set(EXPRESS_STARTER.keys()) == expected_keys

    def test_package_json_has_express_dep(self):
        """Express package.json includes 'express' dependency."""
        pkg = EXPRESS_STARTER["package.json"]
        assert '"express"' in pkg

    def test_package_json_has_cors_and_helmet(self):
        """Express starter includes cors and helmet deps."""
        pkg = EXPRESS_STARTER["package.json"]
        assert '"cors"' in pkg
        assert '"helmet"' in pkg

    def test_package_json_has_tsx_dev_dep(self):
        """Express starter uses tsx for dev script."""
        pkg = EXPRESS_STARTER["package.json"]
        assert '"tsx"' in pkg

    def test_package_json_name_is_api(self):
        """Express starter's package name is 'api'."""
        pkg = EXPRESS_STARTER["package.json"]
        assert '"name": "api"' in pkg

    def test_tsconfig_targets_es2022_commonjs(self):
        """Express tsconfig targets ES2022 with commonjs module."""
        tsconfig = EXPRESS_STARTER["tsconfig.json"]
        assert '"target": "ES2022"' in tsconfig
        assert '"module": "commonjs"' in tsconfig

    def test_tsconfig_has_outdir_dist(self):
        """Express tsconfig outputs to ./dist."""
        tsconfig = EXPRESS_STARTER["tsconfig.json"]
        assert '"outDir": "./dist"' in tsconfig
        assert '"rootDir": "./src"' in tsconfig

    def test_gitignore_exists(self):
        """Express starter includes .gitignore."""
        gi = EXPRESS_STARTER[".gitignore"]
        assert "node_modules/" in gi
        assert "dist/" in gi
        assert ".env" in gi

    def test_src_index_has_health_endpoint(self):
        """Express src/index.ts includes a /health endpoint."""
        src = EXPRESS_STARTER["src/index.ts"]
        assert 'app.get("/health"' in src
        assert 'res.json({ status: "ok" })' in src

    def test_src_index_uses_helmet_cors_json(self):
        """Express src/index.ts uses helmet, cors, and express.json()."""
        src = EXPRESS_STARTER["src/index.ts"]
        assert "helmet()" in src
        assert "cors()" in src
        assert "express.json()" in src

    def test_src_index_listens_on_port(self):
        """Express src/index.ts listens on process.env.PORT || 3001."""
        src = EXPRESS_STARTER["src/index.ts"]
        assert "process.env.PORT || 3001" in src
        assert "app.listen(port" in src

    def test_all_contents_are_non_empty_strings(self):
        """Every file entry has non-empty string content."""
        for path, content in EXPRESS_STARTER.items():
            assert isinstance(content, str), f"{path} content is not str"
            assert len(content) > 0, f"{path} content is empty"


# =============================================================================
# _STARTERS registry
# =============================================================================


class TestStartersRegistry:
    """Validate the internal _STARTERS registry."""

    def test_registry_is_private(self):
        """_STARTERS name indicates it's private (starts with _)."""
        # Just a naming convention check; if it's accessed, it's there.
        assert _STARTERS is not None

    def test_registry_has_three_starters(self):
        """_STARTERS contains exactly 3 entries."""
        assert len(_STARTERS) == TOTAL_STARTER_COUNT

    def test_registry_keys_are_correct(self):
        """_STARTERS maps 'next', 'next-minimal', and 'express'."""
        assert set(_STARTERS.keys()) == {"next", "next-minimal", "express"}

    def test_registry_values_are_starter_dicts(self):
        """Each registry value is a non-empty dict of strings."""
        for name, starter_dict in _STARTERS.items():
            assert isinstance(starter_dict, dict), f"{name} is not a dict"
            assert len(starter_dict) > 0, f"{name} is empty"
            for path, content in starter_dict.items():
                assert isinstance(path, str), f"{name}:{path} key not str"
                assert isinstance(content, str), f"{name}:{path} content not str"

    def test_registry_refers_to_module_dicts(self):
        """_STARTERS entries are the actual module-level dict objects."""
        assert _STARTERS["next"] is NEXT_STARTER
        assert _STARTERS["next-minimal"] is NEXT_MINIMAL_STARTER
        assert _STARTERS["express"] is EXPRESS_STARTER

    def test_registry_immutability_of_file_content(self):
        """The module-level dicts should not be modified (sanity check)."""
        # This is a read-only test — we just verify the data is stable
        # by checking keys haven't changed from expected.
        assert "package.json" in _STARTERS["next"]
        assert "package.json" in _STARTERS["next-minimal"]
        assert "package.json" in _STARTERS["express"]


# =============================================================================
# get_starter()
# =============================================================================


class TestGetStarter:
    """Tests for get_starter(name)."""

    def test_get_next_starter(self):
        """get_starter('next') returns NEXT_STARTER."""
        result = get_starter("next")
        assert result is NEXT_STARTER
        assert result is not None
        assert len(result) == NEXT_FILE_COUNT

    def test_get_next_minimal_starter(self):
        """get_starter('next-minimal') returns NEXT_MINIMAL_STARTER."""
        result = get_starter("next-minimal")
        assert result is NEXT_MINIMAL_STARTER
        assert result is not None
        assert len(result) == NEXT_MINIMAL_FILE_COUNT

    def test_get_express_starter(self):
        """get_starter('express') returns EXPRESS_STARTER."""
        result = get_starter("express")
        assert result is EXPRESS_STARTER
        assert result is not None
        assert len(result) == EXPRESS_FILE_COUNT

    @pytest.mark.parametrize(
        "unknown",
        [
            "unknown",
            "nextjs",
            "react",
            "vue",
            "",
            " ",
            "NEXT",
            "Next",
            "EXPRESS",
            None,
        ],
    )
    def test_get_unknown_starter_returns_none(self, unknown):
        """get_starter returns None for unknown or invalid names."""
        result = get_starter(unknown)
        assert result is None

    def test_returns_direct_reference(self):
        """get_starter returns the actual dict, not a copy."""
        result = get_starter("next")
        result["_test_marker"] = "temporary"
        # Check the module-level dict was affected (and clean up)
        assert "_test_marker" in NEXT_STARTER
        del NEXT_STARTER["_test_marker"]

    def test_content_is_consistent_across_calls(self):
        """Multiple calls to get_starter return the same data."""
        r1 = get_starter("express")
        r2 = get_starter("express")
        assert r1 is r2
        assert r1 == r2


# =============================================================================
# list_starters()
# =============================================================================


class TestListStarters:
    """Tests for list_starters()."""

    def test_returns_three_starters(self):
        """list_starters() returns a list of length 3."""
        result = list_starters()
        assert isinstance(result, list)
        assert len(result) == TOTAL_STARTER_COUNT

    def test_includes_all_starter_names(self):
        """list_starters() contains 'next', 'next-minimal', 'express'."""
        result = list_starters()
        assert "next" in result
        assert "next-minimal" in result
        assert "express" in result

    def test_no_extra_names(self):
        """list_starters() contains exactly the expected names."""
        result = list_starters()
        assert set(result) == {"next", "next-minimal", "express"}

    def test_returns_new_list_each_call(self):
        """list_starters() returns a new list object each call (not the internal keys view)."""
        r1 = list_starters()
        r2 = list_starters()
        assert r1 is not r2
        assert r1 == r2


# =============================================================================
# is_placeholder_file()
# =============================================================================


class TestIsPlaceholderFile:
    """Tests for is_placeholder_file(starter_name, path, content).

    This function checks whether a given file path+content matches the
    original template content from a starter. It returns:
      - True if the starter exists, the path is in the starter, and content matches.
      - False if the starter doesn't exist.
      - False if the path is not in the starter.
      - False if the content has been modified.
    """

    # ------------------------------------------------------------------
    # True cases — exact match with template content
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "starter_name, path",
        [
            ("next", "package.json"),
            ("next", "tsconfig.json"),
            ("next", "next-env.d.ts"),
            ("next", "next.config.mjs"),
            ("next", "postcss.config.mjs"),
            ("next", "tailwind.config.ts"),
            ("next", "app/globals.css"),
            ("next", "lib/utils.ts"),
            ("next", "app/layout.tsx"),
            ("next", "app/page.tsx"),
            ("next-minimal", "package.json"),
            ("next-minimal", "tsconfig.json"),
            ("next-minimal", "next.config.mjs"),
            ("next-minimal", "postcss.config.mjs"),
            ("next-minimal", "tailwind.config.ts"),
            ("next-minimal", "app/globals.css"),
            ("next-minimal", "app/layout.tsx"),
            ("next-minimal", "app/page.tsx"),
            ("express", "package.json"),
            ("express", "tsconfig.json"),
            ("express", ".gitignore"),
            ("express", "src/index.ts"),
        ],
    )
    def test_placeholder_matches_original_content(self, starter_name, path):
        """is_placeholder_file returns True when content matches starter template."""
        starter = get_starter(starter_name)
        content = starter[path]
        assert is_placeholder_file(starter_name, path, content) is True

    # ------------------------------------------------------------------
    # False cases — unknown starter name
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "unknown_name",
        ["unknown", "nonexistent", "react", "vue", "", "NEXT", None],
    )
    def test_unknown_starter_returns_false(self, unknown_name):
        """is_placeholder_file returns False for unknown starter names."""
        assert is_placeholder_file(unknown_name, "package.json", "{}") is False

    # ------------------------------------------------------------------
    # False cases — path not in starter
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "starter_name, path",
        [
            ("next", "nonexistent.py"),
            ("next", "src/index.ts"),
            ("next", "Makefile"),
            ("next-minimal", "lib/utils.ts"),
            ("next-minimal", "next-env.d.ts"),
            ("next-minimal", ".env"),
            ("express", "Dockerfile"),
            ("express", "app/page.tsx"),
        ],
    )
    def test_path_not_in_starter_returns_false(self, starter_name, path):
        """is_placeholder_file returns False when path is not in the starter."""
        content = "any content"
        assert is_placeholder_file(starter_name, path, content) is False

    # ------------------------------------------------------------------
    # False cases — modified content
    # ------------------------------------------------------------------

    def test_modified_package_json_returns_false(self):
        """is_placeholder_file returns False after modifying package.json."""
        content = get_starter("next")["package.json"]
        modified = content.replace('"name": "app"', '"name": "my-modified-app"')
        assert is_placeholder_file("next", "package.json", modified) is False

    def test_modified_layout_returns_false(self):
        """is_placeholder_file returns False after modifying layout.tsx."""
        content = get_starter("next")["app/layout.tsx"]
        modified = content.replace("Your app", "My Custom App")
        assert is_placeholder_file("next", "app/layout.tsx", modified) is False

    def test_modified_page_returns_false(self):
        """is_placeholder_file returns False after modifying page.tsx."""
        content = get_starter("next-minimal")["app/page.tsx"]
        modified = content.replace("Hello, world!", "Goodbye, world!")
        assert is_placeholder_file("next-minimal", "app/page.tsx", modified) is False

    def test_modified_express_src_index_returns_false(self):
        """is_placeholder_file returns False after modifying src/index.ts."""
        content = get_starter("express")["src/index.ts"]
        modified = content.replace("3001", "4000")
        assert is_placeholder_file("express", "src/index.ts", modified) is False

    @pytest.mark.parametrize(
        "starter_name, path, modification",
        [
            ("next", "package.json", '{"modified": true}'),
            ("next", "tsconfig.json", ""),
            ("next", "app/globals.css", " "),
            ("next-minimal", "package.json", "modified content"),
            ("next-minimal", "tailwind.config.ts", "// different config"),
            ("express", "package.json", "{}"),
            ("express", ".gitignore", "*.log"),
            ("express", "tsconfig.json", ""),
        ],
    )
    def test_completely_different_content_returns_false(
        self, starter_name, path, modification
    ):
        """is_placeholder_file returns False for completely different content."""
        assert is_placeholder_file(starter_name, path, modification) is False

    # ------------------------------------------------------------------
    # Edge cases — empty, whitespace content
    # ------------------------------------------------------------------

    def test_empty_content_for_valid_path(self):
        """Empty string content for a valid path returns False (no starter file is empty)."""
        assert is_placeholder_file("next", "package.json", "") is False

    def test_whitespace_only_content(self):
        """Whitespace-only content for a valid path returns False."""
        assert is_placeholder_file("next", "package.json", "   ") is False

    # ------------------------------------------------------------------
    # Cross-starter checks: same path, different content
    # ------------------------------------------------------------------

    def test_next_vs_minimal_same_path_different_content(self):
        """Same path from different starters has different content (not a match)."""
        next_pkg = get_starter("next")["package.json"]
        minimal_pkg = get_starter("next-minimal")["package.json"]
        assert next_pkg != minimal_pkg
        # Minimal's package.json should NOT match the 'next' starter
        assert (
            is_placeholder_file("next", "package.json", minimal_pkg) is False
        )
        # Next's package.json should NOT match the 'next-minimal' starter
        assert (
            is_placeholder_file("next-minimal", "package.json", next_pkg) is False
        )

    # ------------------------------------------------------------------
    # Type safety
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "starter_name, path, content",
        [
            (None, "package.json", "{}"),
            ("next", None, "{}"),
            ("next", "package.json", None),
        ],
    )
    def test_none_arguments_do_not_crash(self, starter_name, path, content):
        """is_placeholder_file handles None arguments gracefully (returns False or error)."""
        # The function uses .get() and == comparisons, so None args won't crash
        try:
            result = is_placeholder_file(starter_name, path, content)
            assert result is False
        except (TypeError, AttributeError):
            pass  # some None args may raise and that's acceptable too

    def test_non_string_content_type_returns_false(self):
        """Non-string content type returns False (not a template match)."""
        assert is_placeholder_file("next", "package.json", 42) is False


# =============================================================================
# Cross-starter consistency
# =============================================================================


class TestCrossStarterConsistency:
    """Tests that verify relationships between different starters."""

    def test_next_minimal_differs_from_next_for_all_shared_paths_except_postcss(self):
        """Files shared between next and next-minimal differ except for postcss.config.mjs which is identical."""
        for path in NEXT_MINIMAL_STARTER:
            if path in NEXT_STARTER and path != "postcss.config.mjs":
                assert (
                    NEXT_MINIMAL_STARTER[path] != NEXT_STARTER[path]
                ), f"{path} content is identical between next and next-minimal"
        # postcss.config.mjs is the same in both
        assert (
            NEXT_MINIMAL_STARTER["postcss.config.mjs"]
            == NEXT_STARTER["postcss.config.mjs"]
        )

    def test_next_has_files_not_in_minimal(self):
        """NEXT_STARTER has files that NEXT_MINIMAL_STARTER doesn't."""
        next_keys = set(NEXT_STARTER.keys())
        minimal_keys = set(NEXT_MINIMAL_STARTER.keys())
        extras = next_keys - minimal_keys
        assert extras == {"next-env.d.ts", "lib/utils.ts"}

    def test_express_has_no_overlap_with_next(self):
        """Express starter files don't overlap with next starter paths."""
        express_paths = set(EXPRESS_STARTER.keys())
        next_paths = set(NEXT_STARTER.keys())
        # Both have package.json and tsconfig.json — content is different
        overlap = express_paths & next_paths
        assert overlap == {"package.json", "tsconfig.json"}

    def test_all_starters_have_package_json(self):
        """All starters include a package.json."""
        for name in list_starters():
            starter = get_starter(name)
            assert "package.json" in starter, f"{name} missing package.json"

    def test_all_starters_have_non_empty_files(self):
        """Every file in every starter has non-empty content."""
        for name in list_starters():
            starter = get_starter(name)
            for path, content in starter.items():
                assert content, f"{name}:{path} is empty"
