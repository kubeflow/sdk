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
import pytest

from kubeflow.common import utils
from kubeflow.trainer.test.common import SUCCESS, TestCase


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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid simple name",
            expected_status=SUCCESS,
            config={"name": "my-train-job"},
        ),
        TestCase(
            name="valid name with uuid suffix",
            expected_status=SUCCESS,
            config={"name": "a1b2c3d4e5f6"},
        ),
        TestCase(
            name="forward slash rejected",
            expected_status=SUCCESS,
            config={"name": "experiment/run_1"},
            expected_error=ValueError,
            expected_output="Invalid job name",
        ),
        TestCase(
            name="backslash rejected",
            expected_status=SUCCESS,
            config={"name": "experiment\\run_1"},
            expected_error=ValueError,
            expected_output="Invalid job name",
        ),
        TestCase(
            name="path traversal rejected",
            expected_status=SUCCESS,
            config={"name": "../run"},
            expected_error=ValueError,
            expected_output="Invalid job name",
        ),
        TestCase(
            name="current directory rejected",
            expected_status=SUCCESS,
            config={"name": "."},
            expected_error=ValueError,
            expected_output="Invalid job name",
        ),
        TestCase(
            name="parent directory rejected",
            expected_status=SUCCESS,
            config={"name": ".."},
            expected_error=ValueError,
            expected_output="Invalid job name",
        ),
    ],
)
def test_validate_filesystem_safe_name(test_case):
    """Test validate_filesystem_safe_name across valid and invalid inputs."""
    print("Executing test:", test_case.name)

    name = test_case.config["name"]

    if test_case.expected_error:
        with pytest.raises(test_case.expected_error, match=test_case.expected_output):
            utils.validate_filesystem_safe_name(name)
    else:
        utils.validate_filesystem_safe_name(name)
