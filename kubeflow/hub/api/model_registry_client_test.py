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

"""Tests for ModelRegistryClient."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


@pytest.fixture(autouse=True)
def skip_if_no_model_registry():
    """Skip tests if model-registry not installed."""
    pytest.importorskip("model_registry")


@pytest.fixture
def mock_registry():
    """Create a mock ModelRegistry with all methods we wrap."""
    registry = MagicMock()
    # Set up return values for list methods to be iterable
    registry.get_registered_models.return_value = iter([])
    registry.get_model_versions.return_value = iter([])
    return registry


@pytest.fixture
def client(mock_registry, monkeypatch):
    """Create ModelRegistryClient with mock registry."""
    from kubeflow.hub.api.model_registry_client import ModelRegistryClient

    # Patch ModelRegistry so __init__ uses the mock
    monkeypatch.setattr("model_registry.ModelRegistry", lambda **kwargs: mock_registry)

    return ModelRegistryClient(base_url="http://localhost", port=8080, author="test")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="raises helpful ImportError when model-registry not installed",
            expected_status=FAILED,
            config={
                "base_url": "http://localhost",
                "port": 9080,
                "author": "test-author",
            },
            expected_error=ImportError,
        ),
    ],
)
def test_init_import_error(test_case, monkeypatch):
    """Test that __init__ raises helpful ImportError when model-registry missing."""

    from kubeflow.hub.api.model_registry_client import ModelRegistryClient

    # Simulate missing model_registry by making import fail
    def mock_import(name, *args, **kwargs):
        if name == "model_registry":
            raise ImportError("No module named 'model_registry'")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    try:
        ModelRegistryClient(**test_case.config)
        assert test_case.expected_status == SUCCESS
    except ImportError as e:
        assert test_case.expected_status == FAILED
        assert "pip install 'kubeflow[hub]'" in str(e)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="http URL infers port=8080 and is_secure=False",
            expected_status=SUCCESS,
            config={
                "base_url": "http://localhost",
                "author": "test",
            },
            expected_output={
                "port": 8080,
                "is_secure": False,
            },
        ),
        TestCase(
            name="https URL infers port=443 and is_secure=True",
            expected_status=SUCCESS,
            config={
                "base_url": "https://registry.example.com",
                "author": "test",
            },
            expected_output={
                "port": 443,
                "is_secure": True,
            },
        ),
        TestCase(
            name="explicit port overrides inference",
            expected_status=SUCCESS,
            config={
                "base_url": "http://localhost",
                "port": 9080,
                "author": "test-author",
            },
            expected_output={
                "port": 9080,
                "is_secure": False,
            },
        ),
    ],
)
def test_init(test_case, monkeypatch):
    """Test ModelRegistryClient initialization with different URL schemes."""

    from kubeflow.hub.api.model_registry_client import ModelRegistryClient

    mock_registry_class = MagicMock()
    mock_registry_instance = MagicMock()
    mock_registry_class.return_value = mock_registry_instance

    monkeypatch.setattr("model_registry.ModelRegistry", mock_registry_class)

    try:
        client = ModelRegistryClient(**test_case.config)

        assert test_case.expected_status == SUCCESS
        mock_registry_class.assert_called_once()
        call_kwargs = mock_registry_class.call_args[1]
        assert call_kwargs["port"] == test_case.expected_output["port"]
        assert call_kwargs["is_secure"] is test_case.expected_output["is_secure"]
        assert client._registry == mock_registry_instance

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="register_model delegates to ModelRegistry.register_model",
            expected_status=SUCCESS,
            config={
                "name": "test",
                "uri": "s3://test",
                "model_format_name": "pytorch",
                "model_format_version": "1.0",
                "version": "v1",
            },
        ),
    ],
)
def test_register_model(test_case, client, mock_registry):
    """Test register_model delegates to ModelRegistry.register_model."""

    try:
        client.register_model(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert mock_registry.register_model.called

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_model delegates to get_registered_model",
            expected_status=SUCCESS,
            config={
                "name": "test-model",
            },
        ),
    ],
)
def test_get_model(test_case, client, mock_registry):
    """Test get_model delegates to get_registered_model."""

    try:
        client.get_model(test_case.config["name"])

        assert test_case.expected_status == SUCCESS
        mock_registry.get_registered_model.assert_called_once_with(test_case.config["name"])

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_model_version delegates to get_model_version",
            expected_status=SUCCESS,
            config={
                "name": "test-model",
                "version": "v1",
            },
        ),
    ],
)
def test_get_model_version(test_case, client, mock_registry):
    """Test get_model_version delegates to ModelRegistry.get_model_version."""

    try:
        client.get_model_version(test_case.config["name"], test_case.config["version"])

        assert test_case.expected_status == SUCCESS
        mock_registry.get_model_version.assert_called_once_with(
            test_case.config["name"], test_case.config["version"]
        )

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get_model_artifact delegates to get_model_artifact",
            expected_status=SUCCESS,
            config={
                "name": "test-model",
                "version": "v1",
            },
        ),
    ],
)
def test_get_model_artifact(test_case, client, mock_registry):
    """Test get_model_artifact delegates to ModelRegistry.get_model_artifact."""

    try:
        client.get_model_artifact(test_case.config["name"], test_case.config["version"])

        assert test_case.expected_status == SUCCESS
        mock_registry.get_model_artifact.assert_called_once_with(
            test_case.config["name"], test_case.config["version"]
        )

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list_models returns iterator that yields from pager",
            expected_status=SUCCESS,
            config={
                "mock_models_count": 2,
            },
            expected_output=2,
        ),
        TestCase(
            name="list_models returns empty iterator when no models",
            expected_status=SUCCESS,
            config={
                "mock_models_count": 0,
            },
            expected_output=0,
        ),
    ],
)
def test_list_models(test_case, client, mock_registry):
    """Test list_models returns an iterator that yields from pager."""

    mock_models = [Mock() for _ in range(test_case.config["mock_models_count"])]
    mock_registry.get_registered_models.return_value = iter(mock_models)

    try:
        result = client.list_models()
        items = list(result)

        assert test_case.expected_status == SUCCESS
        assert len(items) == test_case.expected_output
        assert mock_registry.get_registered_models.called

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="update_model delegates to ModelRegistry.update",
            expected_status=SUCCESS,
            config={
                "model_name": "test-model",
            },
        ),
        TestCase(
            name="update_model raises TypeError for ModelVersion",
            expected_status=FAILED,
            config={
                "wrong_type": "ModelVersion",
                "name": "v1",
            },
            expected_error=TypeError,
        ),
        TestCase(
            name="update_model raises TypeError for ModelArtifact",
            expected_status=FAILED,
            config={
                "wrong_type": "ModelArtifact",
                "name": "artifact",
            },
            expected_error=TypeError,
        ),
    ],
)
def test_update_model(test_case, client, mock_registry):
    """Test update_model delegates to ModelRegistry.update and validates types."""

    from model_registry.types import ModelArtifact, ModelVersion, RegisteredModel

    try:
        if test_case.expected_status == SUCCESS:
            model = RegisteredModel(name=test_case.config["model_name"])
            client.update_model(model)
            mock_registry.update.assert_called_once_with(model)
        else:
            # Test type checking
            if test_case.config["wrong_type"] == "ModelVersion":
                wrong_type = ModelVersion(name=test_case.config["name"])
            else:
                wrong_type = ModelArtifact(name=test_case.config["name"], uri="s3://bucket/model")
            client.update_model(wrong_type)

        assert test_case.expected_status == SUCCESS

    except TypeError as e:
        assert test_case.expected_status == FAILED
        assert "Expected RegisteredModel" in str(e)
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="update_model_version delegates to ModelRegistry.update",
            expected_status=SUCCESS,
            config={
                "version_name": "v1.0",
            },
        ),
        TestCase(
            name="update_model_version raises TypeError for RegisteredModel",
            expected_status=FAILED,
            config={
                "wrong_type": "RegisteredModel",
                "name": "model",
            },
            expected_error=TypeError,
        ),
    ],
)
def test_update_model_version(test_case, client, mock_registry):
    """Test update_model_version delegates to ModelRegistry.update and validates types."""

    from model_registry.types import ModelVersion, RegisteredModel

    try:
        if test_case.expected_status == SUCCESS:
            version = ModelVersion(name=test_case.config["version_name"])
            client.update_model_version(version)
            mock_registry.update.assert_called_once_with(version)
        else:
            # Test type checking
            wrong_type = RegisteredModel(name=test_case.config["name"])
            client.update_model_version(wrong_type)

        assert test_case.expected_status == SUCCESS

    except TypeError as e:
        assert test_case.expected_status == FAILED
        assert "Expected ModelVersion" in str(e)
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="update_model_artifact delegates to ModelRegistry.update",
            expected_status=SUCCESS,
            config={
                "artifact_name": "model-artifact",
                "uri": "s3://bucket/model",
            },
        ),
        TestCase(
            name="update_model_artifact raises TypeError for RegisteredModel",
            expected_status=FAILED,
            config={
                "wrong_type": "RegisteredModel",
                "name": "model",
            },
            expected_error=TypeError,
        ),
    ],
)
def test_update_model_artifact(test_case, client, mock_registry):
    """Test update_model_artifact delegates to ModelRegistry.update and validates types."""

    from model_registry.types import ModelArtifact, RegisteredModel

    try:
        if test_case.expected_status == SUCCESS:
            artifact = ModelArtifact(
                name=test_case.config["artifact_name"],
                uri=test_case.config["uri"],
            )
            client.update_model_artifact(artifact)
            mock_registry.update.assert_called_once_with(artifact)
        else:
            # Test type checking
            wrong_type = RegisteredModel(name=test_case.config["name"])
            client.update_model_artifact(wrong_type)

        assert test_case.expected_status == SUCCESS

    except TypeError as e:
        assert test_case.expected_status == FAILED
        assert "Expected ModelArtifact" in str(e)
    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
