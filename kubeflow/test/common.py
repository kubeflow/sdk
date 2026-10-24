# Copyright The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared test utilities and types for Kubeflow SDK tests."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Common status constants
SUCCESS = "success"
FAILED = "failed"
TIMEOUT = "timeout"
RUNTIME = "runtime"
DEFAULT_NAMESPACE = "default"


@dataclass
class TestCase:
    """Test case dataclass for parametrized tests."""

    name: str
    expected_status: str = SUCCESS
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Any | None = None
    expected_error: type[Exception] | None = None
    __test__ = False


def write_kubeconfig(path: Path, contexts: dict[str, str], current_context: str) -> str:
    """Write a synthetic kubeconfig for tests and return its path.

    Every context points at an unreachable local server with a dummy token, so tests never
    depend on a real kubeconfig or cluster.

    Args:
        path: File to write.
        contexts: Mapping of context name to the namespace of that context.
        current_context: Name of the context set as current-context.
    """
    context_entries = "".join(
        f"- name: {name}\n"
        f"  context:\n"
        f"    cluster: test-cluster\n"
        f"    user: test-user\n"
        f"    namespace: {namespace}\n"
        for name, namespace in contexts.items()
    )
    path.write_text(
        "apiVersion: v1\n"
        "kind: Config\n"
        "clusters:\n"
        "- name: test-cluster\n"
        "  cluster:\n"
        "    server: https://127.0.0.1:1\n"
        "users:\n"
        "- name: test-user\n"
        "  user:\n"
        "    token: dummy-token\n"
        f"contexts:\n{context_entries}"
        f"current-context: {current_context}\n"
    )
    return str(path)


# Default-namespace resolution cases shared by the Trainer and Optimizer Kubernetes backends.
# config keys: config_file (use the explicit kubeconfig), context, namespace,
# default_kubeconfig (write the default kubeconfig, default True), client_configuration.
NAMESPACE_RESOLUTION_TEST_CASES = [
    TestCase(
        name="config_file sets the namespace from its current context",
        expected_status=SUCCESS,
        config={"config_file": True},
        expected_output="team-b",
    ),
    TestCase(
        name="config_file with context sets the namespace from that context",
        expected_status=SUCCESS,
        config={"config_file": True, "context": "team-c-ctx"},
        expected_output="team-c",
    ),
    TestCase(
        name="config_file is used when no default kubeconfig exists",
        expected_status=SUCCESS,
        config={"config_file": True, "default_kubeconfig": False},
        expected_output="team-b",
    ),
    TestCase(
        name="without config_file the default kubeconfig namespace is used",
        expected_status=SUCCESS,
        config={},
        expected_output="default-ns",
    ),
    TestCase(
        name="explicit namespace takes precedence over config_file",
        expected_status=SUCCESS,
        config={"config_file": True, "namespace": "explicit-ns"},
        expected_output="explicit-ns",
    ),
    TestCase(
        name="client_configuration keeps the default kubeconfig namespace",
        expected_status=SUCCESS,
        config={"config_file": True, "client_configuration": True},
        expected_output="default-ns",
    ),
]


def namespace_test_kubeconfigs(test_case: TestCase, tmp_path: Path) -> tuple[str, str | None]:
    """Write the kubeconfigs for a NAMESPACE_RESOLUTION_TEST_CASES entry.

    Returns:
        The default kubeconfig path (which may not exist) and the explicit config_file path,
        or None when the test case does not use one.
    """
    default_kubeconfig = tmp_path / "default-kubeconfig"
    if test_case.config.get("default_kubeconfig", True):
        write_kubeconfig(default_kubeconfig, {"default-ctx": "default-ns"}, "default-ctx")

    config_file = None
    if test_case.config.get("config_file"):
        config_file = write_kubeconfig(
            tmp_path / "team-kubeconfig",
            {"team-b-ctx": "team-b", "team-c-ctx": "team-c"},
            "team-b-ctx",
        )
    return str(default_kubeconfig), config_file
