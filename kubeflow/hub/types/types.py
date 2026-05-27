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

from dataclasses import dataclass


@dataclass
class StorageConfig:
    """Storage credentials for a model artifact.

    Groups the storage-related fields that KServe's StorageInitializer uses
    to pull a model at inference time. Pass an instance to
    ``ModelRegistryClient.register_model`` via the ``storage_config`` keyword
    argument.

    All fields default to ``None``; populate only the fields relevant to your
    storage backend:

    - Secret-based (S3 / MinIO data connections): set ``storage_key`` and
      optionally ``storage_path``.
    - IRSA / Workload Identity: set ``service_account_name``.

    Args:
        storage_key: Name of the Kubernetes Secret containing storage
            credentials (e.g., an S3 access-key/secret pair for MinIO or AWS).
        storage_path: Subpath within the storage bucket where the model
            resides.
        service_account_name: Kubernetes ServiceAccount name annotated for
            IRSA or Workload Identity (cloud-native auth without a Secret).

    Example:
        from kubeflow.hub import ModelRegistryClient, StorageConfig

        client = ModelRegistryClient("https://registry.example.com")
        client.register_model(
            "my-model",
            uri="s3://my-bucket/models/v1",
            version="1.0",
            storage_config=StorageConfig(
                storage_key="my-s3-secret",
                storage_path="models/v1",
            ),
        )
    """

    storage_key: str | None = None
    storage_path: str | None = None
    service_account_name: str | None = None
