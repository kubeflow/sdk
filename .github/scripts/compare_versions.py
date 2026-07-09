#!/usr/bin/env python3
"""Compare two PEP 440 versions from the command line.

Parses two version strings and reports, through the process exit code,
whether an upgrade from the current version to the target version is needed.

Typical usage example:

    python compare_versions.py <current_version> <target_version>

Exit codes:
    0: The current version is older than the target (upgrade needed).
    1: The current version is greater than or equal to the target (no upgrade
        needed).
    2: An argument was missing or a version string could not be parsed.
"""

import sys

from packaging.version import InvalidVersion, Version


def main() -> None:
    """Compare two PEP 440 versions supplied as command-line arguments.

    Reads the current and target version strings from ``sys.argv`` and signals
    the result of the comparison through the process exit code rather than a
    return value.

    Raises:
        SystemExit: Always raised to report the outcome. The exit code is 0 when
            the current version is older than the target (upgrade needed), 1
            when the current version is greater than or equal to the target (no
            upgrade needed), and 2 when an argument is missing or a version
            string cannot be parsed.
    """
    if len(sys.argv) != 3:
        print("Usage: compare_versions.py <current_version> <target_version>", file=sys.stderr)
        sys.exit(2)

    current_str = sys.argv[1]
    target_str = sys.argv[2]

    try:
        current = Version(current_str)
        target = Version(target_str)

        # Exit 0 if current < target (upgrade needed)
        # Exit 1 if current >= target (no upgrade needed)
        sys.exit(0 if current < target else 1)
    except InvalidVersion as e:
        print(f"Error: Invalid version format - {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
