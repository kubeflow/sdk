# Contributing to the Kubeflow SDK

Thank you for your interest in contributing to the Kubeflow SDK!

## Getting Started

### Prerequisites
- Python 3.8–3.11
- [pip](https://pip.pypa.io/en/stable/)
- [pre-commit](https://pre-commit.com/)

### Setting Up Your Development Environment
Clone the repository:
```sh
git clone https://github.com/kubeflow/sdk.git
cd sdk
```

Create a virtual environment and activate it:
```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies in editable mode:
```sh
pip install -e .
```

#### Development Build (Optional)
To install development tools and the latest API modules directly from the master branch:
```sh
pip install -e ".[dev]"
```

## Development Workflow

### Pre-commit
We use pre-commit to ensure consistent code formatting. To enable pre-commit hooks, run:
```sh
pre-commit install
```
To run all hooks manually:
```sh
pre-commit run --all-files
```

### Testing
To run the unit tests (if present), execute:
```sh
pytest
```

#### Development Testing
For development and before submitting PRs, run:
```sh
make test-python-dev
```

This tests against latest master branch dependencies to catch integration issues early.

### Code Coverage
To run tests and measure coverage:
```sh
coverage run -m pytest
coverage report -m
```

### Code Formatting
To check formatting:
```sh
black --check .
```
To auto-format all files:
```sh
black .
```
To sort imports:
```sh
isort .
```
To lint:
```sh
flake8 --exclude .venv
```

## Continuous Integration
All PRs are automatically checked by CI. Please ensure all checks pass before requesting review.

## Getting Help
For questions, open an issue or contact a maintainer listed in `OWNERS`.

## Resources
- [Kubeflow Trainer Docs](https://www.kubeflow.org/docs/components/trainer/)
- [Source Code](https://github.com/kubeflow/trainer)

---
