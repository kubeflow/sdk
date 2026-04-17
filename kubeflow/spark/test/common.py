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

"""Shared test utilities and types for Kubeflow Spark tests."""

from dataclasses import dataclass, field
from typing import Any

from kubeflow_spark_api import models

from kubeflow.spark.backends.kubernetes import constants

# Common status constants
SUCCESS = "success"
FAILED = "failed"
TIMEOUT = "timeout"
RUNTIME = "runtime"
DEFAULT_NAMESPACE = "default"

# SparkConnect states for mocking
SPARK_CONNECT_READY = "spark-connect-ready"
SPARK_CONNECT_PROVISIONING = "spark-connect-provisioning"
SPARK_CONNECT_FAILED = "spark-connect-failed"


@dataclass
class TestCase:
    """Test case dataclass for parametrized tests."""

    name: str
    expected_status: str = SUCCESS
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Any | None = None
    expected_error: type[Exception] | None = None
    __test__ = False


def get_spark_application(
    name: str,
    namespace: str = DEFAULT_NAMESPACE,
    state: str | None = "SUBMITTED",
) -> models.SparkV1beta2SparkApplication:
    """Create a mock SparkApplication model for testing."""
    return models.SparkV1beta2SparkApplication(
        api_version=f"{constants.SPARK_APPLICATION_GROUP}/{constants.SPARK_APPLICATION_VERSION}",
        kind=constants.SPARK_APPLICATION_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
        ),
        spec=models.SparkV1beta2SparkApplicationSpec(
            type="Python",
            mode="cluster",
            spark_version="4.0.1",
            image="spark:latest",
            main_application_file="s3://job.py",
            driver=models.SparkV1beta2DriverSpec(
                cores=1,
                memory="1g",
            ),
            executor=models.SparkV1beta2ExecutorSpec(
                cores=1,
                memory="1g",
                instances=1,
            ),
        ),
        status=(
            models.SparkV1beta2SparkApplicationStatus(
                application_state=models.SparkV1beta2ApplicationState(
                    state=state,
                ),
                driver_info=models.SparkV1beta2DriverInfo(
                    pod_name=f"{name}-driver",
                ),
            )
            if state is not None
            else None
        ),
    )


def get_spark_application_list(
    items: list[models.SparkV1beta2SparkApplication],
) -> models.SparkV1beta2SparkApplicationList:
    """Create a SparkApplicationList for testing."""

    return models.SparkV1beta2SparkApplicationList(
        items=items,
    )
