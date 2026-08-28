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
import multiprocessing
from unittest.mock import MagicMock, Mock, patch

import pytest

from kubeflow.common import utils
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


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


# ---------------------------------------------------------------------------
# Helpers for read_pod_logs tests
# ---------------------------------------------------------------------------


def _make_apply_result(return_value=None, side_effect=None):
    """Return a mock shaped like a k8s ApplyResult (has .get(timeout))."""
    thread = Mock()
    if side_effect is not None:
        thread.get.side_effect = side_effect
    else:
        thread.get.return_value = return_value
    return thread


def _real_generator(*lines):
    """Return a real Python generator — NOT a Mock — to prevent .get() being silently absorbed."""
    yield from lines


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="non-follow: yields split log lines",
            expected_status=SUCCESS,
            config={"follow": False, "log_text": "line1\nline2\nline3"},
            expected_output=["line1", "line2", "line3"],
        ),
        TestCase(
            name="non-follow: empty response yields nothing",
            expected_status=SUCCESS,
            config={"follow": False, "log_text": ""},
            expected_output=[],
        ),
        TestCase(
            name="non-follow: timeout raises TimeoutError",
            expected_status=FAILED,
            config={"follow": False, "api_error": multiprocessing.TimeoutError()},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="non-follow: k8s exception raises RuntimeError",
            expected_status=FAILED,
            config={"follow": False, "api_error": Exception("k8s error")},
            expected_error=RuntimeError,
        ),
        TestCase(
            name="follow: yields each streamed line",
            expected_status=SUCCESS,
            config={"follow": True, "stream_lines": ["stream-line-1", "stream-line-2"]},
            expected_output=["stream-line-1", "stream-line-2"],
        ),
        TestCase(
            name="follow: stream exception raises RuntimeError",
            expected_status=FAILED,
            config={"follow": True, "stream_error": Exception("stream broken")},
            expected_error=RuntimeError,
        ),
    ],
)
def test_read_pod_logs(test_case):
    """Test read_pod_logs for both follow and non-follow paths with shape-faithful mocks.

    The follow-mode mock uses a real Python generator (not Mock) so that any regression
    that calls .get() on the stream would raise AttributeError rather than silently
    returning another Mock and masking the bug.
    """
    print("Executing test:", test_case.name)

    core_api = MagicMock()
    follow = test_case.config.get("follow", False)

    if follow:
        stream_error = test_case.config.get("stream_error")
        stream_lines = test_case.config.get("stream_lines", [])

        if stream_error:
            # Simulate the generator raising on iteration.
            def _error_gen():
                raise stream_error
                yield  # make it a generator

            mock_stream = _error_gen()
        else:
            # Use a *real* generator — not Mock() — so .get() would crash if called,
            # catching any future regression that re-introduces .get(timeout) on the stream.
            mock_stream = _real_generator(*stream_lines)

        with patch("kubeflow.common.utils.watch") as mock_watch:
            mock_watch.Watch.return_value.stream.return_value = mock_stream

            if test_case.expected_status == SUCCESS:
                result = list(
                    utils.read_pod_logs(
                        core_api=core_api,
                        pod_name="test-pod",
                        namespace="default",
                        follow=True,
                    )
                )
                assert result == test_case.expected_output
                # Verify async_req was NOT passed to stream() — it doesn't belong there.
                call_kwargs = mock_watch.Watch.return_value.stream.call_args[1]
                assert "async_req" not in call_kwargs
            else:
                with pytest.raises(test_case.expected_error):
                    list(
                        utils.read_pod_logs(
                            core_api=core_api,
                            pod_name="test-pod",
                            namespace="default",
                            follow=True,
                        )
                    )
    else:
        api_error = test_case.config.get("api_error")
        log_text = test_case.config.get("log_text", "")

        if api_error is not None:
            core_api.read_namespaced_pod_log.return_value = _make_apply_result(
                side_effect=api_error
            )
        else:
            core_api.read_namespaced_pod_log.return_value = _make_apply_result(
                return_value=log_text
            )

        if test_case.expected_status == SUCCESS:
            result = list(
                utils.read_pod_logs(
                    core_api=core_api,
                    pod_name="test-pod",
                    namespace="default",
                    follow=False,
                )
            )
            assert result == test_case.expected_output
            # Verify async_req=True was passed to the API call.
            call_kwargs = core_api.read_namespaced_pod_log.call_args[1]
            assert call_kwargs.get("async_req") is True
        else:
            with pytest.raises(test_case.expected_error):
                list(
                    utils.read_pod_logs(
                        core_api=core_api,
                        pod_name="test-pod",
                        namespace="default",
                        follow=False,
                    )
                )

    print("test execution complete")
