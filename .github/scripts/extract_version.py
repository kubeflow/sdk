#!/usr/bin/env python3
"""Extract a package version from ``uv tree`` output.

This script parses ``uv tree`` output read from standard input to extract the
full version (including pre-release, post-release, and dev versions) of a
package and prints it to standard output.

Typical usage example:

    uv tree --package <package> | python extract_version.py <package>

Exit codes:
    0: The version was found and printed to stdout.
    1: The version could not be found, or an argument was missing.
"""

import re
import sys


def main() -> None:
    """Extract a package version from ``uv tree`` output on standard input.

    Reads the target package name from ``sys.argv`` and the tree output from
    standard input, searches for the package's version string, and prints it
    to standard output.

    Raises:
        SystemExit: Always raised to report the outcome. The exit code is 0
            when the version is found and printed, and 1 when the argument is
            missing or no matching version is found.
    """
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
