# Shared test utilities and types for Kubeflow Trainer tests.

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
    name: str
    expected_status: str = SUCCESS
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Any | None = None
    expected_error: type[Exception] | None = None
    # Prevent pytest from collecting this dataclass as a test
    __test__ = False
