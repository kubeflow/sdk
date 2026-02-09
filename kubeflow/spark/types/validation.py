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

"""Validation utilities for Spark SDK inputs.

This module provides validation functions to ensure user inputs are correct
before they reach the backend. Clear validation errors improve developer experience
and prevent obscure failures deep in the stack.
"""

import re
from typing import Optional


class ValidationError(ValueError):
    """Raised when user input validation fails.

    This exception provides clear, actionable error messages to help users
    fix their configuration quickly.
    """


def validate_resource_dict(
    resources: Optional[dict[str, str]], param_name: str = "resources"
) -> None:
    """Validate a Kubernetes resource requirements dictionary.

    Args:
        resources: Dictionary of resource requirements (cpu, memory, etc).
        param_name: Parameter name for error messages.

    Raises:
        ValidationError: If validation fails with clear explanation.

    Example:
        validate_resource_dict({"cpu": "4", "memory": "8Gi"})
    """
    if resources is None:
        return

    if not isinstance(resources, dict):
        raise ValidationError(
            f"{param_name} must be a dict, got {type(resources).__name__}. "
            f'Example: {{"cpu": "4", "memory": "8Gi"}}'
        )

    if not resources:
        raise ValidationError(f"{param_name} cannot be an empty dict")

    for key, value in resources.items():
        if not isinstance(key, str):
            raise ValidationError(
                f"{param_name} keys must be strings, got {type(key).__name__} for key {key!r}"
            )

        if not isinstance(value, str):
            raise ValidationError(
                f"{param_name} values must be strings, got {type(value).__name__} for value {value!r}"
            )

        # Validate memory format if present
        if (
            "memory" in key.lower()
            and value
            and not re.match(r"^\d+(\.\d+)?(Ki|Mi|Gi|Ti|Pi|Ei|k|M|G|T|P|E)?$", value)
        ):
            raise ValidationError(
                f"Invalid memory format in {param_name}: {value!r}. "
                f'Use Kubernetes format like "4Gi", "512Mi", "1Ti"'
            )

        # Validate CPU format if present (e.g. "4", "0.5", "100m")
        if key == "cpu" and value and not re.match(r"^\d+(\.\d+)?m?$", value):
            raise ValidationError(
                f"Invalid CPU format in {param_name}: {value!r}. "
                f'Use integer cores like "4" or millicores like "500m"'
            )


def validate_spark_conf(conf: Optional[dict[str, str]]) -> None:
    """Validate Spark configuration dictionary.

    Args:
        conf: Dictionary of Spark configuration properties.

    Raises:
        ValidationError: If validation fails.
    """
    if conf is None:
        return

    if not isinstance(conf, dict):
        raise ValidationError(
            f"spark_conf must be a dict, got {type(conf).__name__}. "
            f'Example: {{"spark.sql.adaptive.enabled": "true"}}'
        )

    for key, value in conf.items():
        if not isinstance(key, str):
            raise ValidationError(
                f"spark_conf keys must be strings, got {type(key).__name__} for key {key!r}"
            )

        if not isinstance(value, str):
            raise ValidationError(
                f"spark_conf values must be strings, got {type(value).__name__} for value {value!r}"
            )


def validate_num_instances(num: Optional[int], param_name: str = "num_executors") -> None:
    """Validate number of instances.

    Args:
        num: Number of instances.
        param_name: Parameter name for error messages.

    Raises:
        ValidationError: If validation fails.
    """
    if num is None:
        return

    if not isinstance(num, int):
        raise ValidationError(f"{param_name} must be an integer, got {type(num).__name__}")

    if num <= 0:
        raise ValidationError(f"{param_name} must be positive, got {num}")

    if num > 10000:
        raise ValidationError(
            f"{param_name} seems very large ({num}). "
            f"Please verify this is intentional to avoid resource exhaustion."
        )


def validate_image_name(image: Optional[str]) -> None:
    """Validate Docker image name.

    Args:
        image: Docker image name.

    Raises:
        ValidationError: If validation fails.
    """
    if image is None:
        return

    if not isinstance(image, str):
        raise ValidationError(f"image must be a string, got {type(image).__name__}")

    if not image.strip():
        raise ValidationError("image cannot be empty or whitespace")

    # Basic Docker image name validation
    # Format: [registry/]repository[:tag|@digest]
    if not re.match(r"^[a-z0-9._/-]+(?::[a-z0-9._-]+)?$", image, re.IGNORECASE):
        raise ValidationError(
            f"Invalid Docker image name: {image!r}. "
            f'Use format like "spark:3.4.1" or "gcr.io/my-project/spark:latest"'
        )


def validate_service_account(sa: Optional[str]) -> None:
    """Validate Kubernetes service account name.

    Args:
        sa: Service account name.

    Raises:
        ValidationError: If validation fails.
    """
    if sa is None:
        return

    if not isinstance(sa, str):
        raise ValidationError(f"service_account must be a string, got {type(sa).__name__}")

    if not sa.strip():
        raise ValidationError("service_account cannot be empty or whitespace")

    # Kubernetes name validation (DNS-1123 subdomain)
    if not re.match(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", sa):
        raise ValidationError(
            f"Invalid service account name: {sa!r}. "
            f"Must be lowercase alphanumeric with hyphens, no spaces."
        )

    if len(sa) > 253:
        raise ValidationError(f"service_account name too long ({len(sa)} chars, max 253)")
