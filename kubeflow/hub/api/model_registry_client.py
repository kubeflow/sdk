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

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

from kubeflow.hub.types.types import StorageConfig

if TYPE_CHECKING:
    from model_registry.types import (
        ModelArtifact,
        ModelVersion,
        RegisteredModel,
        SupportedTypes,
    )


class ModelRegistryClient:
    """Client for Kubeflow Model Registry operations.

    Requires the model-registry package to be installed. Install it with:

        pip install 'kubeflow[hub]'

    """

    def __init__(
        self,
        base_url: str,
        port: int | None = None,
        *,
        author: str | None = None,
        is_secure: bool | None = None,
        user_token: str | None = None,
        custom_ca: str | None = None,
    ):
        """Initialize the ModelRegistryClient.

        Args:
            base_url: Base URL of the model registry server including scheme.
                Examples: "https://registry.example.com", "http://localhost".
                The scheme is used to infer is_secure and port if not explicitly provided.

        Keyword Args:
            port: Server port. If not provided, inferred from base_url scheme:
                - https:// defaults to 443
                - http:// defaults to 8080
                - no scheme defaults to 443
            author: Name of the author. Defaults to None.
            is_secure: Whether to use a secure connection. If not provided, inferred from base_url:
                - https:// sets is_secure=True
                - http:// sets is_secure=False
                - no scheme defaults to True
            user_token: The PEM-encoded user token as a string. Defaults to None.
            custom_ca: Path to the PEM-encoded root certificates as a string. Defaults to None.

        Raises:
            ImportError: If model-registry is not installed.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org", port=456)
            >>> client = ModelRegistryClient("https://example.org:456")
            >>> client = ModelRegistryClient("https://example.org")
        """
        try:
            from model_registry import ModelRegistry
        except ImportError as e:
            raise ImportError(
                "model-registry is not installed. Install it with:\n\n"  # fmt: skip
                "  pip install 'kubeflow[hub]'\n"
            ) from e

        is_http = base_url.startswith("http://")
        if is_secure is None:
            is_secure = not is_http
        if port is None:
            port = 8080 if is_http else 443

        self._registry = ModelRegistry(
            server_address=base_url,
            port=port,
            author=author,  # ty: ignore[invalid-argument-type]
            is_secure=is_secure,
            user_token=user_token,
            custom_ca=custom_ca,
        )

    def register_model(
        self,
        name: str,
        uri: str,
        *,
        version: str,
        model_format_name: str | None = None,
        model_format_version: str | None = None,
        author: str | None = None,
        owner: str | None = None,
        version_description: str | None = None,
        metadata: Mapping[str, SupportedTypes] | None = None,
        storage_config: StorageConfig | None = None,
    ) -> RegisteredModel:
        """Register a model in the model registry.

        This registers a model version and its artifact in the registry. The model data
        must be stored in remote storage (e.g., S3, GCS) before registration.

        Most models can be registered using their URI, along with an optional
        `storage_config` describing how KServe should fetch the model at inference time.
        URI builder utilities are recommended when referring to specialized storage; for
        example `utils.s3_uri_from` when using S3 object storage data connections.

        Args:
            name: Name of the model.
            uri: URI of the model artifact in remote storage.

        Keyword Args:
            version: Unique version string for the model (e.g., "v1.0.0").
            model_format_name: Name of the model format (e.g., "pytorch", "onnx").
                Used by KServe to select the appropriate serving runtime. Defaults to None.
            model_format_version: Version of the model format (e.g., "2.0"). Defaults to None.
            author: Author of the model. Defaults to the client author.
            owner: Owner of the model. Defaults to the client author.
            version_description: Description of the model version. Defaults to None.
            metadata: Additional model version metadata. Defaults to None.
            storage_config: Storage credentials configuration for the model artifact.
                See `StorageConfig` for details. Defaults to None.

        Returns:
            The registered model object.

        Raises:
            model_registry.exceptions.StoreError: If registration fails in the underlying store.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> model = client.register_model(
            ...     name="my-model",
            ...     uri="s3://my-bucket/models/my-model",
            ...     version="v1.0.0",
            ...     model_format_name="pytorch",
            ...     model_format_version="2.0",
            ...     version_description="Initial model registration",
            ... )
        """
        storage = storage_config or StorageConfig()
        return self._registry.register_model(
            name=name,
            uri=uri,
            model_format_name=model_format_name,  # ty: ignore[invalid-argument-type]
            model_format_version=model_format_version,  # ty: ignore[invalid-argument-type]
            version=version,
            author=author,
            owner=owner,
            description=version_description,
            metadata=metadata,
            storage_key=storage.storage_key,
            storage_path=storage.storage_path,
            service_account_name=storage.service_account_name,
        )

    def update_model(self, model: RegisteredModel) -> RegisteredModel:
        """Update a registered model metadata or attributes.

        Args:
            model: The registered model object to update. Must contain an ID.

        Returns:
            The updated registered model object.

        Raises:
            TypeError: If the input is not a RegisteredModel instance.
            model_registry.exceptions.StoreError: If the model does not have an ID or store fails.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> model = client.get_model("my-model")
            >>> model.description = "Updated description"
            >>> updated_model = client.update_model(model)
        """
        from model_registry.types import RegisteredModel

        if not isinstance(model, RegisteredModel):
            raise TypeError(f"Expected RegisteredModel, got {type(model).__name__}. ")
        return self._registry.update(model)

    def update_model_version(self, model_version: ModelVersion) -> ModelVersion:
        """Update metadata or attributes of a model version.

        Args:
            model_version: The model version object to update. Must contain an ID.

        Returns:
            The updated model version object.

        Raises:
            TypeError: If the input is not a ModelVersion instance.
            model_registry.exceptions.StoreError: If the model version does not have an ID
                or the store operation fails.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> version = client.get_model_version("my-model", "v1.0.0")
            >>> version.description = "Updated version description"
            >>> updated_version = client.update_model_version(version)
        """
        from model_registry.types import ModelVersion

        if not isinstance(model_version, ModelVersion):
            raise TypeError(f"Expected ModelVersion, got {type(model_version).__name__}. ")
        return self._registry.update(model_version)

    def update_model_artifact(self, model_artifact: ModelArtifact) -> ModelArtifact:
        """Update metadata or attributes of a model artifact.

        Args:
            model_artifact: The model artifact object to update. Must contain an ID.

        Returns:
            The updated model artifact object.

        Raises:
            TypeError: If the input is not a ModelArtifact instance.
            model_registry.exceptions.StoreError: If the model artifact does not have an ID
                or the store operation fails.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> artifact = client.get_model_artifact("my-model", "v1.0.0")
            >>> artifact.uri = "s3://my-bucket/new-path"
            >>> updated_artifact = client.update_model_artifact(artifact)
        """
        from model_registry.types import ModelArtifact

        if not isinstance(model_artifact, ModelArtifact):
            raise TypeError(f"Expected ModelArtifact, got {type(model_artifact).__name__}. ")
        return self._registry.update(model_artifact)

    def get_model(self, name: str) -> RegisteredModel:
        """Retrieve a registered model by name.

        Args:
            name: The name of the registered model.

        Returns:
            The retrieved registered model object.

        Raises:
            ValueError: If the model name is not found.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> model = client.get_model("my-model")
            >>> print(model.id)
        """
        model = self._registry.get_registered_model(name)
        if model is None:
            raise ValueError(f"Model {name!r} not found")
        return model

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """Retrieve a specific model version.

        Args:
            name: The name of the registered model.
            version: The model version string to retrieve.

        Returns:
            The retrieved model version object.

        Raises:
            model_registry.exceptions.StoreError: If the model does not exist.
            ValueError: If the version is not found.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> version = client.get_model_version("my-model", "v1.0.0")
            >>> print(version.id)
        """
        model_version = self._registry.get_model_version(name, version)
        if model_version is None:
            raise ValueError(f"Model version {version!r} not found for model {name!r}")
        return model_version

    def get_model_artifact(self, name: str, version: str) -> ModelArtifact:
        """Retrieve a model artifact.

        Args:
            name: The name of the registered model.
            version: The version string of the model.

        Returns:
            The retrieved model artifact object.

        Raises:
            model_registry.exceptions.StoreError: If either the model or the version does not exist.
            ValueError: If the artifact is not found.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> artifact = client.get_model_artifact("my-model", "v1.0.0")
            >>> print(artifact.uri)
        """
        artifact = self._registry.get_model_artifact(name, version)
        if artifact is None:
            raise ValueError(f"Model artifact not found for model {name!r} version {version!r}")
        return artifact

    def list_models(self) -> Iterator[RegisteredModel]:
        """List registered models.

        Yields:
            Registered model objects in the registry.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> for model in client.list_models():
            ...     print(model.name)
        """
        yield from self._registry.get_registered_models()

    def list_model_versions(self, name: str) -> Iterator[ModelVersion]:
        """List model versions for a registered model.

        Args:
            name: The name of the registered model.

        Yields:
            Model version objects associated with the model.

        Raises:
            model_registry.exceptions.StoreError: If the model does not exist.

        Examples:
            >>> from kubeflow.hub import ModelRegistryClient
            >>> client = ModelRegistryClient("https://example.org")
            >>> for version in client.list_model_versions("my-model"):
            ...     print(version.name)
        """
        yield from self._registry.get_model_versions(name)
