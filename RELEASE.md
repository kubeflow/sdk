# Releasing the Kubeflow SDK

## Prerequisites

- Docker available locally (required for changelog generation with
  [`git-cliff`](https://git-cliff.org/)).

- Create a [GitHub Token](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token).

## Versioning Policy

Kubeflow SDK version format follows Python's [PEP 440](https://peps.python.org/pep-0440/).
Kubeflow SDK versions are in the format of `X.Y.Z`, where `X` is the major version, `Y` is
the minor version, and `Z` is the patch version.
The patch version contains only bug fixes.

Additionally, Kubeflow SDK does pre-releases in this format: `X.Y.ZrcN` where `N` is a number
of the `Nth` release candidate (RC) before an upcoming public release named `X.Y.Z`.

## Release Branches and Tags

Kubeflow SDK releases are tagged with tags like `X.Y.Z`, for example `0.1.0`.

Release branches are in the format of `release-X.Y`, where `X.Y` stands for
the minor release.

`X.Y.Z` releases are released from the `release-X.Y` branch. For example,
`0.1.0` release should be on `release-0.1` branch.

If you want to push changes to the `release-X.Y` release branch, you have to
cherry pick your changes from the `main` branch and submit a PR.

## Changelog Structure

Kubeflow SDK uses a directory-based changelog structure under `CHANGELOG/`:

```
CHANGELOG/
├── CHANGELOG-0.1.md    # All 0.1.x releases
├── CHANGELOG-0.2.md    # All 0.2.x releases
└── CHANGELOG-0.3.md    # All 0.3.x releases
```

Each file contains releases for that minor series. The `make release` target
prepends new entries automatically using `git-cliff`.

## Step-by-Step Release Process

### 1. Update Version and Changelog

For **the latest minor release**, run the following command from the `main` branch.

For **an older minor series patch** (for example, `0.3.1` when `main` is on `0.4.x`), checkout
to the corresponding `release-X.Y` branch and run the following command.

```bash
make release VERSION=X.Y.Z GITHUB_TOKEN=<token>
# or for a release candidate:
make release VERSION=X.Y.ZrcN GITHUB_TOKEN=<token>
```

This will:

1. Update `kubeflow/__init__.py` with `__version__ = "X.Y.Z"`.
1. Generate `CHANGELOG/CHANGELOG-X.Y.md` using `git-cliff` (skipped for RC releases).

After reviewing the changes, create a signed commit and open a PR to the appropriate branch
(e.g. `main` or `release-X.Y`):

```bash
git add -A && git commit -s -m 'Prepare Release X.Y.Z'
```

### 2. Automated Release After Merge

When the `kubeflow/__init__.py` change is merged, the
[release workflow](.github/workflows/release.yaml) runs automatically:

1. **Prepare**: Detects the version change in `kubeflow/__init__.py` and creates or updates the `release-X.Y` branch.
2. **Build**: Runs tests and builds the package on the release branch.
3. **Tag**: Creates and pushes the release tag `X.Y.Z`.
4. **Publish**: Publishes the package to [PyPI](https://pypi.org/project/kubeflow/) (requires manual approval).
5. **Release**: Creates a GitHub Release with the generated changelog (requires manual approval).

### 4. Final Verification

1. Verify the release on [PyPI](https://pypi.org/project/kubeflow/).
2. Verify the release on [GitHub Releases](https://github.com/kubeflow/sdk/releases).
3. Test installation: `pip install kubeflow==X.Y.Z`.

## Announcement

Post the release announcement for the new Kubeflow SDK release in:

- [#kubeflow-ml-experience](https://www.kubeflow.org/docs/about/community/#slack-channels) Slack channel
- [kubeflow-discuss](https://www.kubeflow.org/docs/about/community/#kubeflow-mailing-list) mailing list
