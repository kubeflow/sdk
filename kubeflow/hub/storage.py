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

"""Standalone storage utilities for uploading model artifacts."""

from __future__ import annotations

from kubeflow.hub.types.types import OCIUploadParams, S3UploadParams


def upload_artifact(
    model_files_path: str,
    *,
    upload_params: S3UploadParams | OCIUploadParams,
) -> str:
    """Upload a model artifact to object storage and return its URI.

    This is a decoupled upload utility - it does not register anything in a
    model registry. The returned URI can be passed to any registry client's
    ``register_model`` call.

    Args:
        model_files_path: Local path to the model file or directory to upload.
            Directories are uploaded recursively.

    Keyword Args:
        upload_params: An :class:`S3UploadParams` or :class:`OCIUploadParams`
            instance describing the target storage location and credentials.

    Returns:
        The URI of the uploaded artifact, e.g.
        ``"s3://my-bucket/fraud-detector/v1"`` or
        ``"oci://ghcr.io/my-org/models/fraud-detector:v1"``.

    Raises:
        ImportError: If the required storage backend library is not installed.
            Install the hub extra with ``pip install 'kubeflow[hub]'``.
        TypeError: If ``upload_params`` is not a recognised params type.

    Example::

        from kubeflow.hub import upload_artifact, S3UploadParams, ModelRegistryClient

        uri = upload_artifact(
            "/path/to/model",
            upload_params=S3UploadParams(
                bucket_name="ml-models",
                s3_prefix="fraud-detector/v1",
            ),
        )

        client = ModelRegistryClient("https://registry.example.com")
        client.register_model(
            name="fraud-detector",
            uri=uri,
            version="v1.0.0",
            model_format_name="sklearn",
        )
    """
    if isinstance(upload_params, S3UploadParams):
        return _upload_to_s3(model_files_path, upload_params)
    if isinstance(upload_params, OCIUploadParams):
        return _upload_to_oci(model_files_path, upload_params)
    raise TypeError(
        f"upload_params must be S3UploadParams or OCIUploadParams, "
        f"got {type(upload_params).__name__}"
    )


def _upload_to_s3(path: str, params: S3UploadParams) -> str:
    """Upload to S3-compatible storage."""
    try:
        from model_registry.utils import (
            _connect_to_s3,
            _s3_creds,
            _upload_to_s3 as _mr_upload_to_s3,
        )
    except ImportError as e:
        raise ImportError(
            "S3 upload requires the model-registry package with S3 support.\n"
            "Install it with:  pip install 'kubeflow[hub]'"
        ) from e

    endpoint_url, access_key_id, secret_access_key, region = _s3_creds(
        params.endpoint_url,
        params.access_key_id,
        params.secret_access_key,
        params.region,
    )
    s3, transfer_config = _connect_to_s3(
        endpoint_url=endpoint_url,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        multipart_threshold=params.multipart_threshold,
        multipart_chunksize=params.multipart_chunksize,
        max_pool_connections=params.max_pool_connections,
    )
    return _mr_upload_to_s3(
        path=path,
        path_prefix=params.s3_prefix,
        bucket=params.bucket_name,
        s3=s3,
        endpoint_url=endpoint_url,
        region=region,
        transfer_config=transfer_config,
    )


def _upload_to_oci(path: str, params: OCIUploadParams) -> str:
    """Upload to an OCI registry."""
    try:
        from model_registry.utils import save_to_oci_registry
    except ImportError as e:
        raise ImportError(
            "OCI upload requires the model-registry package with OCI support.\n"
            "Install it with:  pip install 'kubeflow[hub]'"
        ) from e

    return save_to_oci_registry(
        base_image=params.base_image,
        oci_ref=params.oci_ref,
        model_files_path=path,
    )
