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

"""Unit tests for validation utilities."""

import pytest

from kubeflow.spark.types.types import Driver, Executor
from kubeflow.spark.types.validation import (
    ValidationError,
    validate_driver,
    validate_executor,
    validate_image_name,
    validate_num_instances,
    validate_resource_dict,
    validate_service_account,
    validate_spark_conf,
)


class TestValidateResourceDict:
    """Tests for validate_resource_dict."""

    def test_none_passes(self):
        validate_resource_dict(None)

    def test_valid_dict(self):
        validate_resource_dict({"cpu": "4", "memory": "8Gi"})

    def test_wrong_type(self):
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_resource_dict("bad")

    def test_empty_dict(self):
        with pytest.raises(ValidationError, match="cannot be an empty dict"):
            validate_resource_dict({})

    def test_non_string_key(self):
        with pytest.raises(ValidationError, match="keys must be strings"):
            validate_resource_dict({123: "4"})

    def test_non_string_value(self):
        with pytest.raises(ValidationError, match="values must be strings"):
            validate_resource_dict({"cpu": 4})

    def test_invalid_memory_format(self):
        with pytest.raises(ValidationError, match="Invalid memory format"):
            validate_resource_dict({"memory": "lots"})

    def test_valid_memory_formats(self):
        for mem in ["512Mi", "4Gi", "1Ti", "100Ki", "8"]:
            validate_resource_dict({"memory": mem})

    def test_invalid_cpu_format(self):
        with pytest.raises(ValidationError, match="Invalid CPU format"):
            validate_resource_dict({"cpu": "fast"})

    def test_valid_cpu_formats(self):
        for cpu in ["4", "0.5", "500m", "1"]:
            validate_resource_dict({"cpu": cpu})


class TestValidateSparkConf:
    """Tests for validate_spark_conf."""

    def test_none_passes(self):
        validate_spark_conf(None)

    def test_valid_conf(self):
        validate_spark_conf({"spark.sql.adaptive.enabled": "true"})

    def test_wrong_type(self):
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_spark_conf("bad")

    def test_non_string_key(self):
        with pytest.raises(ValidationError, match="keys must be strings"):
            validate_spark_conf({42: "value"})

    def test_non_string_value(self):
        with pytest.raises(ValidationError, match="values must be strings"):
            validate_spark_conf({"spark.key": 42})


class TestValidateNumInstances:
    """Tests for validate_num_instances."""

    def test_none_passes(self):
        validate_num_instances(None)

    def test_valid_num(self):
        validate_num_instances(5)

    def test_wrong_type(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_num_instances("3")

    def test_zero(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_num_instances(0)

    def test_negative(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_num_instances(-1)

    def test_very_large(self):
        with pytest.raises(ValidationError, match="seems very large"):
            validate_num_instances(20000)


class TestValidateImageName:
    """Tests for validate_image_name."""

    def test_none_passes(self):
        validate_image_name(None)

    def test_valid_images(self):
        for img in ["spark:3.4.1", "gcr.io/my-project/spark:latest", "my-repo/image"]:
            validate_image_name(img)

    def test_wrong_type(self):
        with pytest.raises(ValidationError, match="must be a string"):
            validate_image_name(123)

    def test_empty(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_image_name("")

    def test_whitespace(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_image_name("   ")

    def test_invalid_format(self):
        with pytest.raises(ValidationError, match="Invalid Docker image name"):
            validate_image_name("image with spaces")


class TestValidateServiceAccount:
    """Tests for validate_service_account."""

    def test_none_passes(self):
        validate_service_account(None)

    def test_valid_sa(self):
        validate_service_account("spark-sa")

    def test_wrong_type(self):
        with pytest.raises(ValidationError, match="must be a string"):
            validate_service_account(123)

    def test_empty(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_service_account("")

    def test_invalid_format(self):
        with pytest.raises(ValidationError, match="Invalid service account name"):
            validate_service_account("UPPER_CASE")

    def test_too_long(self):
        with pytest.raises(ValidationError, match="too long"):
            validate_service_account("a" * 254)


class TestValidateDriver:
    """Tests for validate_driver."""

    def test_none_passes(self):
        validate_driver(None)

    def test_valid_driver(self):
        validate_driver(Driver(image="spark:3.4", service_account="spark-sa"))

    def test_valid_driver_all_none(self):
        validate_driver(Driver())

    def test_wrong_type(self):
        with pytest.raises(ValidationError, match="must be a Driver instance"):
            validate_driver({"image": "spark:3.4"})

    def test_invalid_image(self):
        with pytest.raises(ValidationError, match="Invalid Docker image name"):
            validate_driver(Driver(image="image with spaces"))

    def test_invalid_resources(self):
        with pytest.raises(ValidationError, match="Invalid memory format"):
            validate_driver(Driver(resources={"memory": "bad"}))

    def test_invalid_service_account(self):
        with pytest.raises(ValidationError, match="Invalid service account name"):
            validate_driver(Driver(service_account="BAD ACCOUNT"))


class TestValidateExecutor:
    """Tests for validate_executor."""

    def test_none_passes(self):
        validate_executor(None)

    def test_valid_executor(self):
        validate_executor(Executor(num_instances=3, resources_per_executor={"cpu": "4"}))

    def test_valid_executor_all_none(self):
        validate_executor(Executor())

    def test_wrong_type(self):
        with pytest.raises(ValidationError, match="must be an Executor instance"):
            validate_executor({"num_instances": 2})

    def test_invalid_num_instances(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_executor(Executor(num_instances=-1))

    def test_invalid_resources(self):
        with pytest.raises(ValidationError, match="Invalid CPU format"):
            validate_executor(Executor(resources_per_executor={"cpu": "bad"}))
