"""Shared test utilities and types for Kubeflow Trainer tests."""

from dataclasses import dataclass, field
from typing import Any

# Common status constants
SUCCESS = "success"
FAILED = "Failed"
DEFAULT_NAMESPACE = "default"
TIMEOUT = "timeout"
RUNTIME = "runtime"


@dataclass
class TestCase:
    """Container describing a single parametrized test case."""

    name: str
    expected_status: str = SUCCESS
    # ``config`` and ``expected_output`` are heterogeneous per-test bags, so they are
    # intentionally typed as ``Any``: the values differ in shape from case to case and
    # are consumed positionally by each test. Keeping them ``Any`` avoids sprinkling
    # casts across every call site while leaving ``expected_error`` meaningfully typed.
    config: Any = field(default_factory=dict)
    expected_output: Any = None
    expected_error: type[Exception] | None = None
    # Prevent pytest from collecting this dataclass as a test
    __test__ = False
