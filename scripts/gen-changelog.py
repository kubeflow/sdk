import argparse
import os
import re
import sys

try:
    from github import Github
except ImportError:
    print("Install PyGithub with uv sync")
    sys.exit(1)

REPO_NAME = "kubeflow/sdk"
CHANGELOG_DIR = "CHANGELOG"


def categorize_pr(title: str) -> str:
    title = title.lower().strip()
    if title.startswith("feat"):
        return "feat"
    elif title.startswith("fix"):
        return "fix"
    elif title.startswith("chore"):
        return "chore"
    elif title.startswith("revert"):
        return "revert"
    else:
        return "misc"


def get_initial_commit(github_repo):
    commits = list(github_repo.get_commits())
    return commits[-1].sha


def main():
    parser = argparse.ArgumentParser(description="Generate changelog for Kubeflow SDK")
    parser.add_argument("--token", required=True, help="GitHub Access Token")
    parser.add_argument("--version", required=True, help="Target version (e.g. 0.1.0)")

    args = parser.parse_args()

    current_release = args.version

    if re.search(r"rc\d+$", current_release):
        print("Skipping changelog generation for pre release")
        return
    github_repo = Github(args.token).get_repo(REPO_NAME)

    try:
        tags = list(github_repo.get_tags())
        stable_tags = [t for t in tags if not re.search(r"rc\d+$", t.name)]
        if stable_tags:
            previous_release = stable_tags[0].name
        else:
            print("First release - using full history")
            previous_release = get_initial_commit(github_repo)

        print(f"Generating changelog: {previous_release} → {current_release}")
    except Exception as e:
        print(f"Error finding previous release: {e}")
        sys.exit(1)

    comparison = github_repo.compare(previous_release, "HEAD")
    commits = list(comparison.commits)

    if not commits:
        print("No commits found in range")
        return

    categories = {"feat": [], "fix": [], "chore": [], "revert": [], "misc": []}

    pr_set = set()
    for commit in reversed(commits):
        for pr in commit.get_pulls():
            if pr.number in pr_set:
                continue
            pr_set.add(pr.number)

            category = categorize_pr(pr.title)
            pr_entry = f"- {pr.title} ([#{pr.number}]({pr.html_url})) by @{pr.user.login}"
            categories[category].append(pr_entry)

    if not pr_set:
        print("No PRs found in range")
        return

    release_date = str(commits[-1].commit.author.date).split(" ")[0]
    release_url = f"https://github.com/{REPO_NAME}/releases/tag/{current_release}"

    major_minor_parts = current_release.split(".")[:2]
    major_minor = ".".join(major_minor_parts)
    changelog_file = os.path.join(CHANGELOG_DIR, f"CHANGELOG-{major_minor}.md")

    os.makedirs(CHANGELOG_DIR, exist_ok=True)

    changelog_content = [f"# [{current_release}]({release_url}) ({release_date})\n\n"]

    if categories["feat"]:
        changelog_content.append("## New Features\n\n")
        changelog_content.append("\n".join(categories["feat"]) + "\n\n")

    if categories["fix"]:
        changelog_content.append("## Bug Fixes\n\n")
        changelog_content.append("\n".join(categories["fix"]) + "\n\n")

    if categories["chore"]:
        changelog_content.append("## Maintenance\n\n")
        changelog_content.append("\n".join(categories["chore"]) + "\n\n")

    if categories["revert"]:
        changelog_content.append("## Reverts\n\n")
        changelog_content.append("\n".join(categories["revert"]) + "\n\n")

    if categories["misc"]:
        changelog_content.append("## Other Changes\n\n")
        changelog_content.append("\n".join(categories["misc"]) + "\n\n")

    try:
        with open(changelog_file) as f:
            existing_content = f.read()
    except FileNotFoundError:
        existing_content = ""

    lines = existing_content.split("\n") if existing_content else []

    new_content = "".join(changelog_content)
    if lines:
        new_lines = new_content.rstrip().split("\n") + [""] + lines
    else:
        new_lines = new_content.rstrip().split("\n")

    with open(changelog_file, "w") as f:
        f.write("\n".join(new_lines))

    print(f"Changelog has been updated: {changelog_file}")
    print(f"Found {len(pr_set)} PRs for {current_release}")


if __name__ == "__main__":
    main()
