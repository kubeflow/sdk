# Release Checklist Workflow

Multi-step workflow for preparing and executing a Kubeflow SDK release.
See `RELEASE.md` for the full release process and `Makefile` for targets.

## Prerequisites

- Docker available locally (for `git-cliff` changelog generation)
- GitHub Token for changelog contributor attribution
- Write access to the `kubeflow/sdk` repository

## Workflow

### Step 1: Determine version

Decide the version following PEP 440:
- `X.Y.Z` for stable releases
- `X.Y.ZrcN` for release candidates

### Step 2: Create release branch (if new minor)

For a new minor release (e.g., `0.5.0`):

```bash
git checkout main
git pull upstream main
git checkout -b release-X.Y
git push upstream release-X.Y
```

### Step 3: Update version and generate changelog

```bash
make release VERSION=X.Y.Z GITHUB_TOKEN=<token>
```

This:
- Updates `kubeflow/__init__.py` with the new version
- Generates changelog in `CHANGELOG/CHANGELOG-X.Y.md` using `git-cliff`

### Step 4: Review changelog

Check `CHANGELOG/CHANGELOG-X.Y.md` for accuracy. Edit if needed.

### Step 5: Commit the release

```bash
git add -A
git commit -s -m "Prepare Release X.Y.Z"
```

### Step 6: Open release PR

Create a PR from the release branch targeting the release branch on upstream.

### Step 7: Tag after merge

After the release PR is merged:

```bash
git tag X.Y.Z
git push upstream X.Y.Z
```

### Step 8: Verify

- Check that the GitHub release is created
- Verify the PyPI package is published
- Test installation: `pip install kubeflow==X.Y.Z`

## Agent guidance

If an agent is assisting with a release:
- The agent should NOT push tags or create releases without explicit approval
- The agent can prepare the `make release` commit and draft the PR
- The agent should verify `make verify` and `make test-python` pass before
  proposing the release commit
