#!/usr/bin/env python3
"""
Extract package version from uv tree output.

This script parses uv tree output to extract the full version
(including pre-release, post-release, dev versions) of a package.

Usage:
    uv tree --package <package> | python extract_version.py <package>

Exit codes:
    0 - version found and printed to stdout
    1 - version not found or error
"""

import re
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: uv tree --package <pkg> | extract_version.py <pkg>", file=sys.stderr)
        sys.exit(1)

    package_name = sys.argv[1]

    # Read from stdin (piped from uv tree)
    tree_output = sys.stdin.read()

    # Look for pattern: "package_name vX.Y.Z"
    # Using non-greedy match to get version until whitespace
    pattern = rf"{re.escape(package_name)}\s+v([^\s]+)"
    match = re.search(pattern, tree_output)

    if match:
        version = match.group(1)
        print(version)
        sys.exit(0)
    else:
        print(f"Error: Could not find version for {package_name}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
