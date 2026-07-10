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

"""Unit tests for update_trainjob_status function."""

from dataclasses import dataclass
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
from kubeflow.trainer.backends.kubernetes.utils import update_trainjob_status


@dataclass
class StatusTestCase:
    """Parametrized test case for update_trainjob_status."""

    name: str
    progress_percent: int | None = None
    estimated_remaining_seconds: int | None = None
    metrics: dict | None = None
    force: bool = True
    expected_result: bool = True
    expected_progress: int | None = None
    expected_eta: int | None = None
    expected_metrics_count: int | None = None
    __test__ = False


PAYLOAD_TEST_CASES = [
    StatusTestCase(
        name="basic_progress",
        progress_percent=50,
        expected_progress=50,
    ),
    StatusTestCase(
        name="progress_clamped_above_100",
        progress_percent=150,
        expected_progress=100,
    ),
    StatusTestCase(
        name="progress_clamped_below_0",
        progress_percent=-10,
        expected_progress=0,
    ),
    StatusTestCase(
        name="eta_in_seconds",
        progress_percent=50,
        estimated_remaining_seconds=3600,
        expected_progress=50,
        expected_eta=3600,
    ),
    StatusTestCase(
        name="eta_shorter_duration",
        progress_percent=50,
        estimated_remaining_seconds=1800,
        expected_progress=50,
        expected_eta=1800,
    ),
    StatusTestCase(
        name="negative_eta_clamped_to_0",
        progress_percent=50,
        estimated_remaining_seconds=-30,
        expected_progress=50,
        expected_eta=0,
    ),
    StatusTestCase(
        name="metrics_included",
        progress_percent=25,
        metrics={"loss": 0.5, "step": 100},
        expected_progress=25,
        expected_metrics_count=2,
    ),
    StatusTestCase(
        name="empty_metrics_omitted",
        progress_percent=75,
        metrics={},
        expected_progress=75,
        expected_metrics_count=None,
    ),
    StatusTestCase(
        name="none_metrics_omitted",
        progress_percent=75,
        metrics=None,
        expected_progress=75,
        expected_metrics_count=None,
    ),
]


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module state before each test."""
    k8s_utils._last_update_time = 0.0
    k8s_utils._cached_token = None
    k8s_utils._token_read_time = 0.0
    k8s_utils._http_session = None


@pytest.fixture
def token_file():
    """Create a temp token file, yield its path, and clean up."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".token") as f:
        f.write("test-token")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def mock_env(token_file):
    """Return env dict with server URL and token path."""
    return {
        "KUBEFLOW_TRAINER_SERVER_URL": "https://trainer.example.com/status",
        "KUBEFLOW_TRAINER_SERVER_TOKEN": token_file,
    }


@pytest.fixture
def mock_session():
    """Patch _get_status_session and return a mock with a 200 response."""
    with patch("kubeflow.trainer.backends.kubernetes.utils._get_status_session") as session_fn:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        session_fn.return_value.post.return_value = mock_resp
        yield session_fn


class TestUpdateRuntimeStatus:
    """Tests for update_trainjob_status function."""

    def test_returns_false_when_not_in_kubeflow(self):
        with patch.dict(os.environ, {}, clear=True):
            assert update_trainjob_status(progress_percent=50) is False

    def test_returns_false_when_token_unavailable(self):
        with patch.dict(
            os.environ,
            {
                "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                "KUBEFLOW_TRAINER_SERVER_TOKEN": "/nonexistent/token",
            },
            clear=True,
        ):
            assert update_trainjob_status(progress_percent=50) is False

    def test_returns_false_on_non_200_response(self, mock_env, mock_session):
        mock_session.return_value.post.return_value.status_code = 422
        mock_session.return_value.post.return_value.text = "Unprocessable"
        with patch.dict(os.environ, mock_env, clear=True):
            assert update_trainjob_status(progress_percent=50, force=True) is False

    def test_returns_false_on_network_exception(self, mock_env, mock_session):
        mock_session.return_value.post.side_effect = ConnectionError("timeout")
        with patch.dict(os.environ, mock_env, clear=True):
            assert update_trainjob_status(progress_percent=50, force=True) is False

    def test_never_raises_exceptions(self):
        with patch.dict(
            os.environ,
            {
                "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                "KUBEFLOW_TRAINER_SERVER_TOKEN": "/nonexistent",
            },
            clear=True,
        ):
            assert update_trainjob_status(progress_percent=50) is False


class TestPayload:
    """Parametrized tests for payload construction."""

    @pytest.mark.parametrize("case", PAYLOAD_TEST_CASES, ids=[c.name for c in PAYLOAD_TEST_CASES])
    def test_payload_fields(self, case: StatusTestCase, mock_env, mock_session):
        with patch.dict(os.environ, mock_env, clear=True):
            result = update_trainjob_status(
                progress_percent=case.progress_percent,
                estimated_remaining_seconds=case.estimated_remaining_seconds,
                metrics=case.metrics,
                force=case.force,
            )
            assert result is case.expected_result

            payload = mock_session.return_value.post.call_args.kwargs["json"]
            status = payload["trainerStatus"]

            assert "lastUpdatedTime" in status

            if case.expected_progress is not None:
                assert status["progressPercentage"] == case.expected_progress

            if case.expected_eta is not None:
                assert status["estimatedRemainingSeconds"] == case.expected_eta

            if case.expected_metrics_count is not None:
                assert len(status["metrics"]) == case.expected_metrics_count
            elif case.metrics is None or len(case.metrics) == 0:
                assert "metrics" not in status

    def test_metrics_values_are_strings(self, mock_env, mock_session):
        with patch.dict(os.environ, mock_env, clear=True):
            update_trainjob_status(
                metrics={"loss": 0.234, "step": 100, "accuracy": "0.95"},
                force=True,
            )
            payload = mock_session.return_value.post.call_args.kwargs["json"]
            for metric in payload["trainerStatus"]["metrics"]:
                assert isinstance(metric["value"], str)

    def test_metrics_truncated_at_256(self, mock_env, mock_session):
        with patch.dict(os.environ, mock_env, clear=True):
            oversized = {f"metric_{i}": i for i in range(300)}
            update_trainjob_status(metrics=oversized, force=True)
            payload = mock_session.return_value.post.call_args.kwargs["json"]
            assert len(payload["trainerStatus"]["metrics"]) == 256


class TestAuthAndHeaders:
    """Verify authorization header and request URL."""

    def test_bearer_token_in_header(self, mock_env, mock_session):
        with patch.dict(os.environ, mock_env, clear=True):
            update_trainjob_status(progress_percent=50, force=True)
            call_kwargs = mock_session.return_value.post.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"

    def test_request_url_matches_env(self, mock_env, mock_session):
        with patch.dict(os.environ, mock_env, clear=True):
            update_trainjob_status(progress_percent=50, force=True)
            call_args = mock_session.return_value.post.call_args
            assert call_args.args[0] == "https://trainer.example.com/status"

    def test_uses_ca_cert_for_tls_verification(self, token_file, mock_session):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".crt") as ca:
            ca.write("fake-cert")
            ca_path = ca.name

        try:
            env = {
                "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                "KUBEFLOW_TRAINER_SERVER_TOKEN": token_file,
                "KUBEFLOW_TRAINER_SERVER_CA_CERT": ca_path,
            }
            with patch.dict(os.environ, env, clear=True):
                update_trainjob_status(progress_percent=50, force=True)
                call_kwargs = mock_session.return_value.post.call_args.kwargs
                assert call_kwargs["verify"] == ca_path
        finally:
            os.unlink(ca_path)


class TestThrottling:
    """Tests for throttle behavior."""

    def test_throttled_call_makes_no_http_post(self, mock_env, mock_session):
        with patch.dict(os.environ, mock_env, clear=True):
            update_trainjob_status(progress_percent=10, force=True)
            assert mock_session.return_value.post.call_count == 1

            result = update_trainjob_status(progress_percent=20)
            assert result is False
            assert mock_session.return_value.post.call_count == 1

    def test_force_bypasses_throttling(self, mock_env, mock_session):
        with patch.dict(os.environ, mock_env, clear=True):
            update_trainjob_status(progress_percent=10, force=True)
            result = update_trainjob_status(progress_percent=30, force=True)
            assert result is True
            assert mock_session.return_value.post.call_count == 2

    def test_failed_send_does_not_consume_throttle_window(self, mock_env, mock_session):
        """A failed HTTP call should not block the next retry."""
        mock_session.return_value.post.return_value.status_code = 500
        mock_session.return_value.post.return_value.text = "error"

        with patch.dict(os.environ, mock_env, clear=True):
            result1 = update_trainjob_status(progress_percent=10, force=True)
            assert result1 is False

            mock_session.return_value.post.return_value.status_code = 200
            result2 = update_trainjob_status(progress_percent=20)
            assert result2 is True


class TestTokenCaching:
    """Tests for _get_cached_token behavior."""

    def test_token_cache_hit_avoids_reread(self, token_file):
        token1 = k8s_utils._get_cached_token(token_file)
        assert token1 == "test-token"

        with open(token_file, "w") as f:
            f.write("new-token")

        token2 = k8s_utils._get_cached_token(token_file)
        assert token2 == "test-token"

    def test_token_cache_expires_after_ttl(self, token_file):
        k8s_utils._get_cached_token(token_file)

        with open(token_file, "w") as f:
            f.write("refreshed-token")

        k8s_utils._token_read_time = time.monotonic() - k8s_utils._TOKEN_CACHE_TTL_SECONDS - 1

        token = k8s_utils._get_cached_token(token_file)
        assert token == "refreshed-token"

    def test_oserror_on_token_read_returns_none(self):
        result = k8s_utils._get_cached_token("/nonexistent/path/token")
        assert result is None

    def test_oserror_on_unreadable_file_returns_none(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            os.chmod(path, 0o000)
            k8s_utils._cached_token = None
            k8s_utils._token_read_time = 0.0
            result = k8s_utils._get_cached_token(path)
            assert result is None
        finally:
            os.chmod(path, 0o644)
            os.unlink(path)
