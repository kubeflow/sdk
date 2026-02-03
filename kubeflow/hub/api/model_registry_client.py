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
                     Examples: "https://registry.example.com", "http://localhost"
                     The scheme is used to infer is_secure and port if not explicitly provided.

        Keyword Args:
            port: Server port. If not provided, inferred from base_url scheme:
                 - https:// defaults to 443
                 - http:// defaults to 8080
                 - no scheme defaults to 443
            author: Name of the author.
            is_secure: Whether to use a secure connection. If not provided, inferred from base_url:
                      - https:// sets is_secure=True
                      - http:// sets is_secure=False
                      - no scheme defaults to True
            user_token: The PEM-encoded user token as a string.
            custom_ca: Path to the PEM-encoded root certificates as a string.

        Raises:
            ImportError: If model-registry is not installed.
        Examples:
            ModelRegistryClient("https://example.org", port=456)  # port kwarg
            ModelRegistryClient("https://example.org:456")        # base_url (including port)
            ModelRegistryClient("https://example.org")            # default port (`443` for https, `8080` for http)
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
            author=author,  # type: ignore[arg-type]
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
    ) -> RegisteredModel:
        """Register a model.

        This registers a model in the model registry. The model is not downloaded,
        and has to be stored prior to registration.

        Most models can be registered using their URI, along with optional
        connection-specific parameters, `storage_key` and `storage_path` or,
        simply a `service_account_name`. URI builder utilities are recommended
        when referring to specialized storage; for example `utils.s3_uri_from`
        helper when using S3 object storage data connections.

        Args:
            name: Name of the model.
            uri: URI of the model.

        Keyword Args:
            version: Version of the model. Has to be unique.
            model_format_name: Name of the model format (e.g., "pytorch", "tensorflow", "onnx").
                               Used by KServe to select the appropriate serving runtime.
            model_format_version: Version of the model format (e.g., "2.0", "1.15").
            author: Author of the model. Defaults to the client author.
            owner: Owner of the model. Defaults to the client author.
            version_description: Description of the model version.
            metadata: Additional version metadata.

        Returns:
            Registered model.
        """
        return self._registry.register_model(
            name=name,
            uri=uri,
            model_format_name=model_format_name,  # type: ignore[arg-type]
            model_format_version=model_format_version,  # type: ignore[arg-type]
            version=version,
            author=author,
            owner=owner,
            description=version_description,
            metadata=metadata,
        )

    def update_model(self, model: RegisteredModel) -> RegisteredModel:
        """Update a registered model.

        Args:
            model: The registered model to update. Must have an ID.

        Returns:
            Updated registered model.

        Raises:
            TypeError: If model is not a RegisteredModel instance.
            model_registry.exceptions.StoreError: If model does not have an ID.
        """
        from model_registry.types import RegisteredModel

        if not isinstance(model, RegisteredModel):
            raise TypeError(f"Expected RegisteredModel, got {type(model).__name__}. ")
        return self._registry.update(model)

    def update_model_version(self, model_version: ModelVersion) -> ModelVersion:
        """Update a model version.

        Args:
            model_version: The model version to update. Must have an ID.

        Returns:
            Updated model version.

        Raises:
            TypeError: If model_version is not a ModelVersion instance.
            model_registry.exceptions.StoreError: If model version does not have an ID.
        """
        from model_registry.types import ModelVersion

        if not isinstance(model_version, ModelVersion):
            raise TypeError(f"Expected ModelVersion, got {type(model_version).__name__}. ")
        return self._registry.update(model_version)

    def update_model_artifact(self, model_artifact: ModelArtifact) -> ModelArtifact:
        """Update a model artifact.

        Args:
            model_artifact: The model artifact to update. Must have an ID.

        Returns:
            Updated model artifact.

        Raises:
            TypeError: If model_artifact is not a ModelArtifact instance.
            model_registry.exceptions.StoreError: If model artifact does not have an ID.
        """
        from model_registry.types import ModelArtifact

        if not isinstance(model_artifact, ModelArtifact):
            raise TypeError(f"Expected ModelArtifact, got {type(model_artifact).__name__}. ")
        return self._registry.update(model_artifact)

    def get_model(self, name: str) -> RegisteredModel:
        """Get a registered model.

        Args:
            name: Name of the model.

        Returns:
            Registered model.

        Raises:
            ValueError: If the model does not exist.
        """
        model = self._registry.get_registered_model(name)
        if model is None:
            raise ValueError(f"Model {name!r} not found")
        return model

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """Get a model version.

        Args:
            name: Name of the model.
            version: Version of the model.

        Returns:
            Model version.

        Raises:
            model_registry.exceptions.StoreError: If the model does not exist.
            ValueError: If the version does not exist.
        """
        model_version = self._registry.get_model_version(name, version)
        if model_version is None:
            raise ValueError(f"Model version {version!r} not found for model {name!r}")
        return model_version

    def get_model_artifact(self, name: str, version: str) -> ModelArtifact:
        """Get a model artifact.

        Args:
            name: Name of the model.
            version: Version of the model.

        Returns:
            Model artifact.

        Raises:
            model_registry.exceptions.StoreError: If either the model or the version don't exist.
            ValueError: If the artifact does not exist.
        """
        artifact = self._registry.get_model_artifact(name, version)
        if artifact is None:
            raise ValueError(f"Model artifact not found for model {name!r} version {version!r}")
        return artifact

    def list_models(self) -> Iterator[RegisteredModel]:
        """Get an iterator for registered models.

        Yields:
            Registered models.
        """
        yield from self._registry.get_registered_models()

    def list_model_versions(self, name: str) -> Iterator[ModelVersion]:
        """Get an iterator for model versions.

        Args:
            name: Name of the model.

        Yields:
            Model versions.

        Raises:
            model_registry.exceptions.StoreError: If the model does not exist.
        """
        yield from self._registry.get_model_versions(name)
