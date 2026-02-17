#!/usr/bin/env python3
"""
Compare PEP 440 versions.

Usage:
    python compare_versions.py <current_version> <target_version>

Exit codes:
    0 - current_version < target_version (upgrade needed)
    1 - current_version >= target_version (no upgrade needed)
    2 - error parsing versions
"""

import sys

from packaging.version import InvalidVersion, Version


def main():
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
