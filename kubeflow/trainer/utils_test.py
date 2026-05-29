# Copyright 2024 The Kubeflow Authors.
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

"""Unit tests for update_runtime_status function."""

from datetime import timedelta
import os
import tempfile
from unittest.mock import MagicMock, patch

from kubeflow.trainer import utils
from kubeflow.trainer.utils import update_runtime_status


class TestUpdateRuntimeStatus:
    """Tests for update_runtime_status function."""

    def setup_method(self):
        """Reset module state before each test."""
        utils._last_update_time = 0.0
        utils._cached_token = None
        utils._token_read_time = 0.0
        utils._session = None

    def _make_token_file(self, content: str = "test-token") -> str:
        """Create a temp file with a token and return its path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".token") as f:
            f.write(content)
            return f.name

    def _mock_env(self, token_path: str) -> dict:
        return {
            "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
            "KUBEFLOW_TRAINER_SERVER_TOKEN": token_path,
        }

    def test_returns_false_when_not_in_kubeflow(self):
        """Should return False when KUBEFLOW_TRAINER_SERVER_URL is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = update_runtime_status(progress_percent=50)
            assert result is False

    def test_returns_false_when_token_unavailable(self):
        """Should return False when token file doesn't exist."""
        with patch.dict(
            os.environ,
            {
                "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                "KUBEFLOW_TRAINER_SERVER_TOKEN": "/nonexistent/token",
            },
            clear=True,
        ):
            result = update_runtime_status(progress_percent=50)
            assert result is False

    def test_throttling_skips_frequent_updates(self):
        """Should skip updates that are too frequent (within 5 seconds)."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test-token")
            token_path = f.name

        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                        "KUBEFLOW_TRAINER_SERVER_TOKEN": token_path,
                    },
                    clear=True,
                ),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_session.return_value.post.return_value = mock_response

                # First call should succeed
                result1 = update_runtime_status(progress_percent=10, force=True)
                assert result1 is True

                # Second call immediately after should be throttled
                result2 = update_runtime_status(progress_percent=20)
                assert result2 is False

                # Force should bypass throttling
                result3 = update_runtime_status(progress_percent=30, force=True)
                assert result3 is True
        finally:
            os.unlink(token_path)

    def test_accepts_timedelta_for_eta(self):
        """Should accept timedelta for estimated_time_remaining."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test-token")
            token_path = f.name

        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                        "KUBEFLOW_TRAINER_SERVER_TOKEN": token_path,
                    },
                    clear=True,
                ),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_session.return_value.post.return_value = mock_response

                result = update_runtime_status(
                    progress_percent=50,
                    estimated_time_remaining=timedelta(hours=1),
                    force=True,
                )
                assert result is True

                call_args = mock_session.return_value.post.call_args
                payload = call_args.kwargs["json"]
                assert payload["trainerStatus"]["estimatedRemainingSeconds"] == 3600
        finally:
            os.unlink(token_path)

    def test_metrics_converted_to_strings(self):
        """Should convert all metric values to strings."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test-token")
            token_path = f.name

        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                        "KUBEFLOW_TRAINER_SERVER_TOKEN": token_path,
                    },
                    clear=True,
                ),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_session.return_value.post.return_value = mock_response

                result = update_runtime_status(
                    progress_percent=50,
                    metrics={"loss": 0.234, "step": 100, "accuracy": "0.95"},
                    force=True,
                )
                assert result is True

                call_args = mock_session.return_value.post.call_args
                payload = call_args.kwargs["json"]
                metrics = payload["trainerStatus"]["metrics"]

                for metric in metrics:
                    assert isinstance(metric["value"], str)
        finally:
            os.unlink(token_path)

    def test_progress_clamped_to_0_100(self):
        """Should clamp progress to 0-100 range."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test-token")
            token_path = f.name

        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                        "KUBEFLOW_TRAINER_SERVER_TOKEN": token_path,
                    },
                    clear=True,
                ),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_session.return_value.post.return_value = mock_response

                # Test clamping above 100
                update_runtime_status(progress_percent=150, force=True)
                call_args = mock_session.return_value.post.call_args
                payload = call_args.kwargs["json"]
                assert payload["trainerStatus"]["progressPercentage"] == 100

                # Reset for next test
                utils._last_update_time = 0.0

                # Test clamping below 0
                update_runtime_status(progress_percent=-10, force=True)
                call_args = mock_session.return_value.post.call_args
                payload = call_args.kwargs["json"]
                assert payload["trainerStatus"]["progressPercentage"] == 0
        finally:
            os.unlink(token_path)

    def test_never_raises_exceptions(self):
        """Should never raise exceptions, only return False."""
        with patch.dict(
            os.environ,
            {
                "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                "KUBEFLOW_TRAINER_SERVER_TOKEN": "/nonexistent",
            },
            clear=True,
        ):
            result = update_runtime_status(progress_percent=50)
            assert result is False

    def test_payload_includes_last_updated_time(self):
        """Should always include lastUpdatedTime in payload."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test-token")
            token_path = f.name

        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "KUBEFLOW_TRAINER_SERVER_URL": "https://test",
                        "KUBEFLOW_TRAINER_SERVER_TOKEN": token_path,
                    },
                    clear=True,
                ),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_session.return_value.post.return_value = mock_response

                update_runtime_status(progress_percent=50, force=True)

                call_args = mock_session.return_value.post.call_args
                payload = call_args.kwargs["json"]
                assert "lastUpdatedTime" in payload["trainerStatus"]
        finally:
            os.unlink(token_path)

    def test_returns_false_on_non_200_response(self):
        """Should return False when the controller responds with a non-200 status."""
        token_path = self._make_token_file()
        try:
            with (
                patch.dict(os.environ, self._mock_env(token_path), clear=True),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 422
                mock_response.text = "Unprocessable Entity"
                mock_session.return_value.post.return_value = mock_response

                result = update_runtime_status(progress_percent=50, force=True)
                assert result is False
        finally:
            os.unlink(token_path)

    def test_returns_false_on_network_exception(self):
        """Should return False (never raise) when the HTTP call itself fails."""
        token_path = self._make_token_file()
        try:
            with (
                patch.dict(os.environ, self._mock_env(token_path), clear=True),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_session.return_value.post.side_effect = ConnectionError("timeout")

                result = update_runtime_status(progress_percent=50, force=True)
                assert result is False
        finally:
            os.unlink(token_path)

    def test_accepts_int_seconds_for_eta(self):
        """Should accept a plain int for estimated_time_remaining."""
        token_path = self._make_token_file()
        try:
            with (
                patch.dict(os.environ, self._mock_env(token_path), clear=True),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_session.return_value.post.return_value = mock_response

                result = update_runtime_status(
                    progress_percent=50,
                    estimated_time_remaining=1800,
                    force=True,
                )
                assert result is True

                payload = mock_session.return_value.post.call_args.kwargs["json"]
                assert payload["trainerStatus"]["estimatedRemainingSeconds"] == 1800
        finally:
            os.unlink(token_path)

    def test_uses_ca_cert_for_tls_verification(self):
        """Should pass CA cert path to requests when the file exists."""
        token_path = self._make_token_file()
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".crt") as ca:
                ca.write("fake-cert")
                ca_path = ca.name

            with (
                patch.dict(
                    os.environ,
                    {
                        **self._mock_env(token_path),
                        "KUBEFLOW_TRAINER_SERVER_CA_CERT": ca_path,
                    },
                    clear=True,
                ),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_session.return_value.post.return_value = mock_response

                update_runtime_status(progress_percent=50, force=True)

                call_kwargs = mock_session.return_value.post.call_args.kwargs
                assert call_kwargs["verify"] == ca_path
        finally:
            os.unlink(token_path)
            os.unlink(ca_path)

    def test_metrics_truncated_at_256(self):
        """Should truncate metrics dict to 256 entries with a warning."""
        token_path = self._make_token_file()
        try:
            with (
                patch.dict(os.environ, self._mock_env(token_path), clear=True),
                patch("kubeflow.trainer.utils._get_session") as mock_session,
            ):
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_session.return_value.post.return_value = mock_response

                oversized = {f"metric_{i}": i for i in range(300)}
                update_runtime_status(metrics=oversized, force=True)

                payload = mock_session.return_value.post.call_args.kwargs["json"]
                assert len(payload["trainerStatus"]["metrics"]) == 256
        finally:
            os.unlink(token_path)
