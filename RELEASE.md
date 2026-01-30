# Releasing the Kubeflow SDK

## Prerequisites

- [Write](https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-permission-levels-for-an-organization#permission-levels-for-repositories-owned-by-an-organization)
  permission for the Kubeflow SDK repository.

- Create a [GitHub Token](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token) and set it as `GITHUB_TOKEN` environment variable.

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

Each file contains releases for that minor series, with the most recent releases at the top.

## Release Process

### Automated Release Workflow

The Kubeflow SDK uses an automated release process with GitHub Actions:

1. **Local Preparation**: Update version and generate changelog locally
2. **Automated CI**: GitHub Actions handles branch creation, tagging, building, and publishing
3. **Manual Approvals**: PyPI and GitHub releases require manual approval

### Step-by-Step Release Process

#### 1. Update Version and Changelog

1. Generate version and changelog locally (this will sync dependencies automatically):

   ```sh
   export GITHUB_TOKEN=<your_github_token>
   make release VERSION=<X.Y.Z>
   # e.g. make release VERSION=0.3.1
   ```

This updates:
- `kubeflow/__init__.py` with `__version__ = "X.Y.Z"`
- `CHANGELOG/CHANGELOG-X.Y.md` with a new top entry `# [X.Y.Z] (YYYY-MM-DD)`

2. Open a PR:
   - Review `kubeflow/__init__.py` and `CHANGELOG/CHANGELOG-X.Y.md`
   - **For latest minor series**: Open a PR to `main` and get it reviewed and merged
   - **For older minor series patch (e.g. 0.1.1 when main is at 0.2.x)**: Checkout the `release-X.Y` branch, commit changes and then open a PR to the corresponding `release-X.Y` branch

#### 2. Automated Release Process

The `Release` GitHub Action automatically:

1. **Prepare**: Detects the version change in `kubeflow/__init__.py` and creates or updates the `release-X.Y` branch
2. **Build**: Runs tests and builds the package on the release branch
3. **Tag**: Creates and pushes the release tag
4. **Publish**: Publishes to PyPI (requires manual approval)
5. **Release**: Creates GitHub Release (requires manual approval)

**Verification**: Confirm the release branch and tag were created!

#### 3. Manual Approvals

1. **PyPI Publishing**: Go to [GitHub Actions](https://github.com/kubeflow/sdk/actions) → `Release` workflow → Approve "Publish to PyPI"

2. **GitHub Release**: After PyPI approval → Approve "Create GitHub Release"

#### 4. Final Verification

1. Verify the release on [PyPI](https://pypi.org/project/kubeflow/)
2. Verify the release on [GitHub Releases](https://github.com/kubeflow/sdk/releases)
3. Test installation: `pip install kubeflow==X.Y.Z`


## Announcement

**Announce**: Post the announcement for the new Kubeflow SDK release in:
- [#kubeflow-ml-experience](https://www.kubeflow.org/docs/about/community/#slack-channels) Slack channel
- [kubeflow-discuss](https://www.kubeflow.org/docs/about/community/#kubeflow-mailing-list) mailing list
