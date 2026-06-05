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

"""Tests for standalone storage utilities."""

from __future__ import annotations

from unittest.mock import Mock

from pydantic import ValidationError
import pytest

from kubeflow.hub.storage import upload_artifact
from kubeflow.hub.types import OCIUploadParams, S3UploadParams


def test_upload_artifact_delegates_to_s3(monkeypatch):
    """Test upload_artifact delegates S3 uploads to the S3 implementation."""
    mock_upload = Mock(return_value="s3://bucket/prefix")
    monkeypatch.setattr("kubeflow.hub.storage._upload_to_s3", mock_upload)

    params = S3UploadParams(bucket_name="bucket", s3_prefix="prefix")
    result = upload_artifact("/tmp/model", upload_params=params)

    assert result == "s3://bucket/prefix"
    mock_upload.assert_called_once_with("/tmp/model", params)


def test_upload_artifact_delegates_to_oci(monkeypatch):
    """Test upload_artifact delegates OCI uploads to the OCI implementation."""
    mock_upload = Mock(return_value="oci://registry.example.com/model:v1")
    monkeypatch.setattr("kubeflow.hub.storage._upload_to_oci", mock_upload)

    params = OCIUploadParams(base_image="python:3.11", oci_ref="registry.example.com/model:v1")
    result = upload_artifact("/tmp/model", upload_params=params)

    assert result == "oci://registry.example.com/model:v1"
    mock_upload.assert_called_once_with("/tmp/model", params)


def test_upload_artifact_raises_type_error_for_unknown_upload_params():
    """Test upload_artifact rejects unsupported upload parameter types."""
    with pytest.raises(TypeError, match="upload_params must be"):
        upload_artifact("/tmp/model", upload_params=object())  # type: ignore[arg-type]


def test_s3_upload_params_requires_bucket_name():
    """Test S3UploadParams validates required fields."""
    with pytest.raises(ValidationError):
        S3UploadParams(s3_prefix="prefix")  # type: ignore[call-arg]


def test_oci_upload_params_requires_base_image_and_oci_ref():
    """Test OCIUploadParams validates required fields."""
    with pytest.raises(ValidationError):
        OCIUploadParams()  # type: ignore[call-arg]
