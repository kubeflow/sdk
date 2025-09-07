# Contributing to the Kubeflow SDK

This guide explains how to contribute to the Kubeflow SDK project.
For the Kubeflow SDK documentation, please check [the official Kubeflow documentation](https://www.kubeflow.org/docs/components/).

## Requirements
- Python 3.9–3.11
- [pre-commit](https://pre-commit.com/)
- uv


## Development

The Kubeflow SDK project includes a Makefile with several helpful commands to streamline your development workflow.

Install uv if not installed [Official Docs](https://docs.astral.sh/uv/getting-started/installation/) or using the following command:
```shell
make uv
```
### Install SDK & Dependencies

Create a virtual environment and install all dependencies (including dev tools):

```sh
uv sync
```

#### API Models

To install development tools and the latest API modules directly from the master branch:

```shell
uv sync --dev
```

### Pre-commit
We use pre-commit to ensure consistent code formatting. To enable pre-commit hooks, run:
```shell
pre-commit install
```
To run all hooks manually:
```sh
pre-commit run --all-files
```

## Testing

The Kubeflow SDK project includes several types of tests to ensure code quality and functionality.

### Unit Testing
To run unit tests locally use the following make command:

```shell
make test-python
```

### E2E Tests
E2E test run in CI on a kind cluster using [Kubeflow Trainer E2E Scripts](https://github.com/kubeflow/trainer/blob/master/CONTRIBUTING.md#e2e-tests).
Clone the `Kubeflow Trainer` repo and run the provided commands against `Trainer` Makefile.
For more details check [the Kubeflow Trainer Contributing Guide](https://github.com/kubeflow/trainer/blob/master/CONTRIBUTING.md#e2e-tests).

## Coding Style
Make sure to install [pre-commit](https://pre-commit.com/) (`uv pip install pre-commit`) and run `pre-commit install` from the root of the repository at least once before creating git commits.

The pre-commit hooks ensure code quality and consistency. They are executed in CI. PRs that fail to comply with the hooks will not be able to pass the corresponding CI gate. The hooks are only executed against staged files unless you run `pre-commit run --all`, in which case, they'll be executed against every file in the repository.

Specific programmatically generated files listed in the `exclude` field in [.pre-commit-config.yaml](.pre-commit-config.yaml) are deliberately excluded from the hooks.

To check formatting:

```shell
make verify 
```

### Using Ruff

You can use ruff directly to check the linting issues before commiting your code

```shell
uv run ruff check --show-fixes
```

To fix all lint issues automatically using ruff append `--fix` to your ruff check command.

```shell
uv run ruff check --fix .
```

To auto-format, lint all files

```shell
uv run ruff format .
```

## Best Practices

### Pull Request Title Conventions

We enforce a pull request (PR) title convention to quickly indicate the type and scope of a PR.
The PR titles are used to generated changelog for releases.

PR titles must:

- Follow the [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/).
- Have an appropriate [type and scope](./.github/workflows/check-pr-title.yaml)

Examples:

- fix: Check empty value for ml_policy
- chore(ci): Remove unused scripts
- feat(docs): Create guide for LLM Fine-Tuning

### Kubeflow Enhancement Proposal (KEP)

For any significant features or enhancement for Kubeflow SDK project we follow the
[Kubeflow Enhancement Proposal process](https://github.com/kubeflow/community/tree/master/proposals).

If you want to submit a significant change to the Kubeflow Trainer, please submit a new KEP under
[./docs/proposals](./docs/proposals/) directory.