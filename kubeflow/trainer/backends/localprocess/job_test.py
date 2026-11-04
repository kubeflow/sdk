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

from unittest.mock import patch

import pytest

from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.test.common import SUCCESS, TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="return buffered lines when follow is disabled",
            expected_status=SUCCESS,
            config={"follow": False},
            expected_output=["line-1", "line-2"],
        ),
        TestCase(
            name="return streaming lines without printing when follow is enabled",
            expected_status=SUCCESS,
            config={"follow": True},
            expected_output=["line-1", "line-2"],
        ),
    ],
)
def test_logs(test_case: TestCase):
    """Test LocalJob.logs()."""
    job = LocalJob(name="test-job", command=["echo", "unused"])
    job._stdout = "line-1\nline-2\n"

    if test_case.config["follow"]:
        with (
            patch.object(job, "stream_logs", return_value=iter(test_case.expected_output)),
            patch("builtins.print") as mock_print,
        ):
            assert list(job.logs(follow=True)) == test_case.expected_output
        mock_print.assert_not_called()
    else:
        assert job.logs() == test_case.expected_output
