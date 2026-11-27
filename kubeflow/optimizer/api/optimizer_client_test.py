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

"""Unit tests for OptimizerClient API."""

from unittest.mock import patch

import pytest

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.optimizer.api.optimizer_client import OptimizerClient

BACKEND_PATH = "kubeflow.optimizer.api.optimizer_client.KubernetesBackend"


def test_init_default_backend():
    """OptimizerClient without a config builds a KubernetesBackend."""

    with patch(BACKEND_PATH) as mock_backend:
        client = OptimizerClient()

        assert client.backend is not None
        mock_backend.assert_called_once()


def test_init_custom_namespace():
    """A KubernetesBackendConfig is passed through to the backend."""

    with patch(BACKEND_PATH) as mock_backend:
        OptimizerClient(backend_config=KubernetesBackendConfig(namespace="optimizer"))

        mock_backend.assert_called_once()


def test_init_invalid_backend_config_raises():
    """A non-KubernetesBackendConfig backend_config raises ValueError."""

    with pytest.raises(ValueError):
        OptimizerClient(backend_config="invalid")


def test_optimize_delegates_to_backend():
    """optimize forwards its arguments to the backend and returns the job name."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value
        backend.optimize.return_value = "opt-12345"

        template = object()
        trial_config = object()
        objectives = object()
        algorithm = object()
        search_space = {"lr": object()}

        client = OptimizerClient()
        name = client.optimize(
            trial_template=template,
            trial_config=trial_config,
            search_space=search_space,
            objectives=objectives,
            algorithm=algorithm,
        )

        assert name == "opt-12345"
        backend.optimize.assert_called_once_with(
            trial_template=template,
            trial_config=trial_config,
            objectives=objectives,
            search_space=search_space,
            algorithm=algorithm,
        )


def test_list_jobs_delegates_to_backend():
    """list_jobs returns whatever the backend returns."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value
        sentinel = [object(), object()]
        backend.list_jobs.return_value = sentinel

        client = OptimizerClient()
        result = client.list_jobs()

        assert result is sentinel
        backend.list_jobs.assert_called_once_with()


def test_get_job_delegates_to_backend():
    """get_job forwards the name and returns the backend result."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value
        sentinel = object()
        backend.get_job.return_value = sentinel

        client = OptimizerClient()
        result = client.get_job("opt-12345")

        assert result is sentinel
        backend.get_job.assert_called_once_with(name="opt-12345")


def test_get_job_logs_delegates_to_backend():
    """get_job_logs forwards name, trial_name, and follow to the backend."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value
        sentinel = iter(["line-1", "line-2"])
        backend.get_job_logs.return_value = sentinel

        client = OptimizerClient()
        result = client.get_job_logs(name="opt-12345", trial_name="trial-1", follow=True)

        assert result is sentinel
        backend.get_job_logs.assert_called_once_with(
            name="opt-12345",
            trial_name="trial-1",
            follow=True,
        )


def test_get_best_results_delegates_to_backend():
    """get_best_results forwards the name and returns the backend result."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value
        sentinel = object()
        backend.get_best_results.return_value = sentinel

        client = OptimizerClient()
        result = client.get_best_results("opt-12345")

        assert result is sentinel
        backend.get_best_results.assert_called_once_with(name="opt-12345")


def test_wait_for_job_status_delegates_to_backend():
    """wait_for_job_status forwards all arguments once validation passes."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value
        sentinel = object()
        backend.wait_for_job_status.return_value = sentinel

        client = OptimizerClient()
        result = client.wait_for_job_status(
            name="opt-12345",
            status={"Complete"},
            timeout=100,
            polling_interval=2,
            callbacks=None,
        )

        assert result is sentinel
        backend.wait_for_job_status.assert_called_once_with(
            name="opt-12345",
            status={"Complete"},
            timeout=100,
            polling_interval=2,
            callbacks=None,
        )


@pytest.mark.parametrize(
    "polling_interval,timeout",
    [
        (2, 0),
        (0, 100),
        (100, 100),
    ],
)
def test_wait_for_job_status_invalid_values_raise(polling_interval, timeout):
    """Invalid polling_interval/timeout fail before the backend is called."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value

        client = OptimizerClient()
        with pytest.raises(ValueError):
            client.wait_for_job_status(
                name="opt-12345",
                timeout=timeout,
                polling_interval=polling_interval,
            )

        backend.wait_for_job_status.assert_not_called()


def test_delete_job_delegates_to_backend():
    """delete_job forwards the name to the backend."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value

        client = OptimizerClient()
        client.delete_job("opt-12345")

        backend.delete_job.assert_called_once_with(name="opt-12345")


def test_get_job_events_delegates_to_backend():
    """get_job_events forwards the name and returns the backend result."""

    with patch(BACKEND_PATH) as mock_backend:
        backend = mock_backend.return_value
        sentinel = [object()]
        backend.get_job_events.return_value = sentinel

        client = OptimizerClient()
        result = client.get_job_events("opt-12345")

        assert result is sentinel
        backend.get_job_events.assert_called_once_with(name="opt-12345")
