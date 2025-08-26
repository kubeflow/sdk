# Contributing to the Kubeflow SDK

Thank you for your interest in contributing to the Kubeflow SDK!

## Getting Started

## Prerequisites
- Python 3.9–3.11
- [pip](https://pip.pypa.io/en/stable/)
- [pre-commit](https://pre-commit.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Development
Clone the repository:
```sh
git clone https://github.com/kubeflow/sdk.git
cd sdk
```

Install uv if not installed [Official Docs](https://docs.astral.sh/uv/getting-started/installation/) or using the following command
```sh
make uv
```
### Install SDK & Dependencies
Use uv to create a virtualenv if not created and install dependencies
```sh
uv sync
```

#### Development Build (Optional)
To install development tools and the latest API modules directly from the master branch:
```sh
uv sync

```

## Development Workflow

### Pre-commit
We use pre-commit to ensure consistent code formatting. To enable pre-commit hooks, run:
```sh
uv run pre-commit install
```
To run all hooks manually:
```sh
uv run pre-commit run --all-files
```

## Testing
To run the unit tests (if present), execute:
```sh
pytest
```

### Code Coverage
To run tests and measure coverage:
```sh
coverage run -m pytest
coverage report -m
```

## Coding Style
To check formatting:
```shell
make verify 
```

### Using Ruff

```shell
uv run ruff check --fix
```

To auto-format, lint all files:

```shell
uv run ruff format .
```

## Continuous Integration
All PRs are automatically checked by CI. Please ensure all checks pass before requesting review.

## Getting Help
For questions, open an issue or contact a maintainer listed in `OWNERS`.

## Resources
- [Kubeflow Trainer Docs](https://www.kubeflow.org/docs/components/trainer/)
- [Source Code](https://github.com/kubeflow/trainer)

---
