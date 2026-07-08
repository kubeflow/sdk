#!/usr/bin/env python3
"""Unit tests for GitHub Actions scripts.

Run with: uv run pytest .github/scripts/test_scripts.py -v
"""

from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

import pytest

SCRIPTS_DIR = Path(__file__).parent


class TestCompareVersions:
    """Tests for the compare_versions.py script."""

    def run_compare(self, current: str, target: str) -> int:
        """Run compare_versions.py and return its exit code.

        Args:
            current: The current version string to compare.
            target: The target version string to compare against.

        Returns:
            The exit code returned by the compare_versions.py script.
        """
        result = subprocess.run(
            [sys.executable, SCRIPTS_DIR / "compare_versions.py", current, target],
            capture_output=True,
            text=True,
        )
        return result.returncode

    def test_upgrade_needed(self) -> None:
        """Current version below target should return 0 (upgrade needed)."""
        assert self.run_compare("2.0.0", "2.1.0") == 0
        assert self.run_compare("1.0.0", "2.0.0") == 0
        assert self.run_compare("2.0.0", "2.0.1") == 0

    def test_no_upgrade_needed(self) -> None:
        """Current version at or above target should return 1 (no upgrade)."""
        assert self.run_compare("2.1.0", "2.0.0") == 1
        assert self.run_compare("2.0.0", "2.0.0") == 1
        assert self.run_compare("3.0.0", "2.9.9") == 1

    def test_pre_release_versions(self) -> None:
        """Compare PEP 440 pre-release versions."""
        # Pre-release is less than final release
        assert self.run_compare("2.0.0rc1", "2.0.0") == 0
        assert self.run_compare("2.0.0a1", "2.0.0") == 0
        assert self.run_compare("2.0.0b1", "2.0.0") == 0

        # Final release is greater than pre-release
        assert self.run_compare("2.0.0", "2.0.0rc1") == 1

    def test_post_release_versions(self) -> None:
        """Compare PEP 440 post-release versions."""
        # Post-release is greater than base release
        assert self.run_compare("2.0.0", "2.0.0.post1") == 0
        assert self.run_compare("2.0.0.post1", "2.0.0") == 1

    def test_dev_versions(self) -> None:
        """Compare PEP 440 dev versions."""
        # Dev versions are less than base release
        assert self.run_compare("2.0.0.dev1", "2.0.0") == 0
        assert self.run_compare("2.0.0", "2.0.0.dev1") == 1

    def test_complex_version_comparison(self) -> None:
        """Compare complex multi-part version numbers."""
        assert self.run_compare("2.0.0.1", "2.0.0.2") == 0
        assert self.run_compare("2.10.0", "2.9.0") == 1  # 10 > 9, not string comparison
        assert self.run_compare("1.0.0rc1", "1.0.0rc2") == 0

    def test_invalid_version_error(self) -> None:
        """Invalid version should return error code 2."""
        result = subprocess.run(
            [sys.executable, SCRIPTS_DIR / "compare_versions.py", "invalid", "2.0.0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "Error" in result.stderr

    def test_missing_arguments(self) -> None:
        """Missing arguments should return error code 2."""
        result = subprocess.run(
            [sys.executable, SCRIPTS_DIR / "compare_versions.py", "2.0.0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2


class TestExtractVersion:
    """Tests for the extract_version.py script."""

    def run_extract(self, tree_output: str, package: str) -> tuple[int, str, str]:
        """Run extract_version.py and return its result.

        Args:
            tree_output: The ``uv tree`` output to pipe into the script's stdin.
            package: The package name whose version should be extracted.

        Returns:
            A tuple of the exit code, the stripped stdout, and the stripped
            stderr produced by the script.
        """
        result = subprocess.run(
            [sys.executable, SCRIPTS_DIR / "extract_version.py", package],
            input=tree_output,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def test_simple_version_extraction(self) -> None:
        """Extract a version from simple ``uv tree`` output."""
        tree_output = "requests v2.31.0"
        exit_code, version, _ = self.run_extract(tree_output, "requests")
        assert exit_code == 0
        assert version == "2.31.0"

    def test_version_with_metadata(self) -> None:
        """Extract a version from tree output that includes metadata."""
        tree_output = "├── requests v2.31.0 (transitive)"
        exit_code, version, _ = self.run_extract(tree_output, "requests")
        assert exit_code == 0
        assert version == "2.31.0"

    def test_pre_release_version(self) -> None:
        """Extract a pre-release version."""
        tree_output = "package v1.0.0rc1"
        exit_code, version, _ = self.run_extract(tree_output, "package")
        assert exit_code == 0
        assert version == "1.0.0rc1"

    def test_post_release_version(self) -> None:
        """Extract a post-release version."""
        tree_output = "package v1.0.0.post1"
        exit_code, version, _ = self.run_extract(tree_output, "package")
        assert exit_code == 0
        assert version == "1.0.0.post1"

    def test_dev_version(self) -> None:
        """Extract a dev version."""
        tree_output = "package v1.0.0.dev1"
        exit_code, version, _ = self.run_extract(tree_output, "package")
        assert exit_code == 0
        assert version == "1.0.0.dev1"

    def test_package_not_found(self) -> None:
        """Package missing from the tree output should return an error."""
        tree_output = "other-package v1.0.0"
        exit_code, _, stderr = self.run_extract(tree_output, "requests")
        assert exit_code == 1
        assert "Could not find version" in stderr

    def test_multiline_tree_output(self) -> None:
        """Extract a version from realistic multi-line ``uv tree`` output."""
        tree_output = """
        project v1.0.0
        ├── requests v2.31.0
        │   ├── certifi v2023.7.22
        │   └── urllib3 v2.0.4
        └── pytest v7.4.0
        """
        exit_code, version, _ = self.run_extract(tree_output, "requests")
        assert exit_code == 0
        assert version == "2.31.0"


class TestUpdateOverrides:
    """Tests for the update_overrides.py script."""

    def run_update(
        self, pyproject_content: str, package: str, target: str, date: str, advisory: str
    ) -> tuple[int, str]:
        """Run update_overrides.py against a temporary ``pyproject.toml``.

        Args:
            pyproject_content: The initial contents of the ``pyproject.toml``.
            package: The package name to add or update in the overrides.
            target: The full version specifier for the override.
            date: The date recorded alongside the override.
            advisory: The advisory URL recorded alongside the override.

        Returns:
            A tuple of the script's exit code and the updated file contents.
        """
        with NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(pyproject_content)
            temp_path = Path(f.name)

        try:
            # Change to temp directory so script finds pyproject.toml
            import os

            original_dir = os.getcwd()
            os.chdir(temp_path.parent)

            # Create pyproject.toml in current directory
            Path("pyproject.toml").write_text(pyproject_content)

            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPTS_DIR / "update_overrides.py",
                    package,
                    target,
                    date,
                    advisory,
                ],
                capture_output=True,
                text=True,
            )

            updated_content = Path("pyproject.toml").read_text()

            # Cleanup
            Path("pyproject.toml").unlink()
            os.chdir(original_dir)

            return result.returncode, updated_content
        finally:
            temp_path.unlink(missing_ok=True)

    def test_create_tool_uv_section(self) -> None:
        """Create the [tool.uv] section when it does not exist."""
        pyproject = "[project]\nname = 'test'\n"
        exit_code, updated = self.run_update(
            pyproject, "requests", "requests==2.31.0", "2025-01-15", "https://advisory.com"
        )

        assert exit_code == 0
        assert "[tool.uv]" in updated
        assert "override-dependencies = [" in updated
        assert '"requests==2.31.0",' in updated
        assert "# requests==2.31.0 - Added 2025-01-15" in updated

    def test_add_to_existing_tool_uv(self) -> None:
        """Add override-dependencies to an existing [tool.uv] section."""
        pyproject = "[project]\nname = 'test'\n\n[tool.uv]\n"
        exit_code, updated = self.run_update(
            pyproject, "requests", "requests==2.31.0", "2025-01-15", "https://advisory.com"
        )

        assert exit_code == 0
        assert "override-dependencies = [" in updated
        assert '"requests==2.31.0",' in updated

    def test_update_existing_override(self) -> None:
        """Update an existing package override to a new version."""
        pyproject = """[project]
name = 'test'

[tool.uv]
# Security overrides - Review periodically and remove if parent constraints allow natural upgrade
override-dependencies = [
    # requests==2.30.0 - Added 2025-01-01 for security fix - https://old.com
    "requests==2.30.0",
]
"""
        exit_code, updated = self.run_update(
            pyproject, "requests", "requests==2.31.0", "2025-01-15", "https://new.com"
        )

        assert exit_code == 0
        assert '"requests==2.31.0",' in updated
        assert "2.30.0" not in updated  # Old version removed
        assert "2025-01-15" in updated  # New date
        assert "https://new.com" in updated  # New advisory

    def test_add_second_override(self) -> None:
        """Add a second package to existing overrides."""
        pyproject = """[project]
name = 'test'

[tool.uv]
override-dependencies = [
    # requests==2.31.0 - Added 2025-01-15 for security fix - https://advisory.com
    "requests==2.31.0",
]
"""
        exit_code, updated = self.run_update(
            pyproject, "urllib3", "urllib3==2.0.5", "2025-01-16", "https://urllib-advisory.com"
        )

        assert exit_code == 0
        assert '"requests==2.31.0",' in updated
        assert '"urllib3==2.0.5",' in updated
        # Should be alphabetically sorted
        lines = updated.split("\n")
        requests_line = next(i for i, line in enumerate(lines) if "requests==" in line)
        urllib_line = next(i for i, line in enumerate(lines) if "urllib3==" in line)
        assert requests_line < urllib_line  # requests comes before urllib3

    def test_multiline_array_preserved(self) -> None:
        """Ensure the output is always a multi-line array."""
        pyproject = "[project]\nname = 'test'\n"
        exit_code, updated = self.run_update(
            pyproject, "pkg", "pkg==1.0.0", "2025-01-15", "https://advisory.com"
        )

        assert exit_code == 0
        # Check multi-line format
        assert "override-dependencies = [\n" in updated
        assert '    "pkg==1.0.0",\n' in updated
        assert "]\n" in updated

    def test_single_line_array_converted(self) -> None:
        """Single-line array should be parsed and converted to multi-line."""
        pyproject = """[project]
name = 'test'

[tool.uv]
override-dependencies = ["pkg1==1.0.0", "pkg2==2.0.0"]
"""
        exit_code, updated = self.run_update(
            pyproject, "pkg3", "pkg3==3.0.0", "2025-01-15", "https://advisory.com"
        )

        assert exit_code == 0
        # Should preserve existing packages
        assert '"pkg1==1.0.0",' in updated
        assert '"pkg2==2.0.0",' in updated
        assert '"pkg3==3.0.0",' in updated
        # Should be in multi-line format
        assert "override-dependencies = [\n" in updated
        # Should not have duplicate override-dependencies
        assert updated.count("override-dependencies") == 1


if __name__ == "__main__":
    # Allow running with: python test_scripts.py
    pytest.main([__file__, "-v"])
