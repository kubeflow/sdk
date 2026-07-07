#!/usr/bin/env python3
"""Verify repository boilerplate headers."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
YEAR_RE = re.compile(r"Copyright \d{4}(?:-\d{4})? ")
ANY_YEAR_RE = re.compile(r"Copyright \d{4}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify boilerplate headers")
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Base branch used to detect new files (default: main).",
    )
    return parser.parse_args()


def load_template(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def template_for(relpath: str) -> str | None:
    if relpath == "Makefile":
        return "sh"
    if relpath == "docs/source/conf.py":
        return "sh"
    if relpath.startswith("kubeflow/") and relpath.endswith(".py"):
        return "sh"
    if relpath.startswith("examples/spark/") and relpath.endswith(".py"):
        return "sh"
    if relpath.startswith("test/e2e/spark/") and relpath.endswith(".py"):
        return "sh"
    if relpath.startswith("docs/_ext/") and relpath.endswith(".py"):
        return "sh"
    if relpath.startswith(".github/scripts/") and relpath.endswith(".py"):
        return "sh"
    if relpath.startswith("hack/") and relpath.endswith(".sh"):
        return "sh"
    if relpath.startswith("hack/") and relpath.endswith(".go"):
        return "go"
    return None


def collect_files() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT_DIR),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if template_for(line)]


def base_tree_files(base_ref: str) -> set[str] | None:
    for ref in (f"origin/{base_ref}", base_ref):
        try:
            merge_base = subprocess.run(
                ["git", "-C", str(ROOT_DIR), "merge-base", ref, "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            tree = subprocess.run(
                ["git", "-C", str(ROOT_DIR), "ls-tree", "-r", "--name-only", merge_base],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            continue
        return {line for line in tree.stdout.splitlines() if line}
    return None


def strip_shebang(lines: list[str]) -> list[str]:
    if lines and lines[0].startswith("#!"):
        return lines[1:]
    return lines


def file_passes(path: Path, template: list[str], new_file: bool) -> tuple[bool, str | None]:
    try:
        data = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"error reading file: {exc}"

    lines = strip_shebang(data.splitlines())
    if len(lines) < len(template):
        return False, "file is shorter than the expected boilerplate header"

    header = lines[: len(template)]
    normalized = [YEAR_RE.sub("Copyright ", line) for line in header]
    if normalized != template:
        return False, "header does not match the boilerplate template"

    if new_file and any(ANY_YEAR_RE.search(line) for line in header):
        return False, "new files must use the year-less boilerplate header"

    return True, None


def main() -> int:
    args = parse_args()

    templates = {
        "sh": load_template(ROOT_DIR / "hack" / "boilerplate" / "boilerplate.sh.txt"),
        "go": load_template(ROOT_DIR / "hack" / "boilerplate" / "boilerplate.go.txt"),
    }

    files = collect_files()
    base_files = base_tree_files(args.base_ref)
    if base_files is None:
        print(
            f"ERROR: could not resolve base ref {args.base_ref!r}; fetch the base branch first.",
            file=sys.stderr,
        )
        return 1

    failed: list[tuple[str, str]] = []
    for relpath in files:
        stem = template_for(relpath)
        if stem is None:
            continue

        path = ROOT_DIR / relpath
        passes, reason = file_passes(path, templates[stem], relpath not in base_files)
        if not passes:
            failed.append((relpath, reason or "unknown error"))

    if failed:
        print("Boilerplate verification failed:", file=sys.stderr)
        for relpath, reason in failed:
            print(f"  {relpath}: {reason}", file=sys.stderr)
        print(
            "Use the year-less copyright boilerplate from hack/boilerplate/ and rerun make verify-boilerplate.",
            file=sys.stderr,
        )
        return 1

    print("Boilerplate header verification passed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())