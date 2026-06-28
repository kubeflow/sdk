# Copyright 2025 The Kubeflow Authors.
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
from unittest.mock import mock_open, patch

import pytest

from kubeflow.common import constants, utils
from kubeflow.trainer.test.common import SUCCESS, TestCase


def _context(name: str, namespace: str | None) -> dict:
    inner: dict = {"cluster": f"{name}-cluster", "user": f"{name}-user"}
    if namespace is not None:
        inner["namespace"] = namespace
    return {"name": name, "context": inner}


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="explicit context returns its namespace",
            expected_status=SUCCESS,
            config={
                "context": "prod",
                "all_contexts": [_context("dev", "dev-ns"), _context("prod", "prod-ns")],
                "current_context": _context("dev", "dev-ns"),
            },
            expected_output="prod-ns",
        ),
        TestCase(
            name="explicit context without namespace falls back to default",
            expected_status=SUCCESS,
            config={
                "context": "prod",
                "all_contexts": [_context("prod", None)],
                "current_context": _context("dev", "dev-ns"),
            },
            expected_output=constants.DEFAULT_NAMESPACE,
        ),
        TestCase(
            name="explicit context with empty namespace falls back to default",
            expected_status=SUCCESS,
            config={
                "context": "prod",
                "all_contexts": [_context("prod", "")],
                "current_context": _context("dev", "dev-ns"),
            },
            expected_output=constants.DEFAULT_NAMESPACE,
        ),
        TestCase(
            name="unknown context falls back to default instead of current context",
            expected_status=SUCCESS,
            config={
                "context": "missing",
                "all_contexts": [_context("dev", "dev-ns")],
                "current_context": _context("dev", "dev-ns"),
            },
            expected_output=constants.DEFAULT_NAMESPACE,
        ),
        TestCase(
            name="explicit empty context is honored, not the current context",
            expected_status=SUCCESS,
            config={
                "context": "",
                "all_contexts": [_context("dev", "dev-ns")],
                "current_context": _context("dev", "dev-ns"),
            },
            expected_output=constants.DEFAULT_NAMESPACE,
        ),
        TestCase(
            name="no context uses the current context namespace",
            expected_status=SUCCESS,
            config={
                "context": None,
                "all_contexts": [_context("dev", "dev-ns")],
                "current_context": _context("dev", "dev-ns"),
            },
            expected_output="dev-ns",
        ),
        TestCase(
            name="kube config error falls back to default",
            expected_status=SUCCESS,
            config={"context": None, "raises": Exception("no kube config")},
            expected_output=constants.DEFAULT_NAMESPACE,
        ),
    ],
)
def test_get_default_target_namespace_out_of_cluster(test_case: TestCase):
    print("Executing test:", test_case.name)

    if "raises" in test_case.config:
        list_contexts = patch(
            "kubernetes.config.list_kube_config_contexts",
            side_effect=test_case.config["raises"],
        )
    else:
        list_contexts = patch(
            "kubernetes.config.list_kube_config_contexts",
            return_value=(
                test_case.config["all_contexts"],
                test_case.config["current_context"],
            ),
        )

    with (
        patch("kubeflow.common.utils.is_running_in_k8s", return_value=False),
        list_contexts,
    ):
        namespace = utils.get_default_target_namespace(test_case.config["context"])

    assert test_case.expected_status == SUCCESS
    assert namespace == test_case.expected_output
    print("test execution complete")


def test_get_default_target_namespace_in_cluster_strips_newline():
    """In-cluster the namespace is read from the service account file and trimmed."""

    with (
        patch("kubeflow.common.utils.is_running_in_k8s", return_value=True),
        patch("builtins.open", mock_open(read_data="my-namespace\n")),
    ):
        namespace = utils.get_default_target_namespace()

    assert namespace == "my-namespace"


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid polling_interval and timeout",
            expected_status=SUCCESS,
            config={"polling_interval": 2, "timeout": 600},
        ),
        TestCase(
            name="timeout is zero",
            expected_status=SUCCESS,
            config={"polling_interval": 2, "timeout": 0},
            expected_error=ValueError,
            expected_output="Timeout must be a positive number",
        ),
        TestCase(
            name="polling_interval is zero",
            expected_status=SUCCESS,
            config={"polling_interval": 0, "timeout": 600},
            expected_error=ValueError,
            expected_output="Polling interval must be a positive number",
        ),
        TestCase(
            name="polling_interval is negative",
            expected_status=SUCCESS,
            config={"polling_interval": -5, "timeout": 600},
            expected_error=ValueError,
            expected_output="Polling interval must be a positive number",
        ),
        TestCase(
            name="polling_interval equals timeout",
            expected_status=SUCCESS,
            config={"polling_interval": 10, "timeout": 10},
            expected_error=ValueError,
            expected_output="Polling interval must be strictly less than timeout",
        ),
    ],
)
def test_validate_wait_for_job_status(test_case):
    """Test validate_wait_for_job_status across valid and invalid inputs."""
    print("Executing test:", test_case.name)

    polling_interval = test_case.config["polling_interval"]
    timeout = test_case.config["timeout"]

    if test_case.expected_error:
        with pytest.raises(test_case.expected_error, match=test_case.expected_output):
            utils.validate_wait_for_job_status(polling_interval, timeout)
    else:
        utils.validate_wait_for_job_status(polling_interval, timeout)
