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

"""Unit tests for Kubeflow Spark options."""

from unittest.mock import MagicMock

from kubeflow_spark_api import models
import pytest

from kubeflow.spark.backends.kubernetes import constants
from kubeflow.spark.backends.kubernetes.backend import KubernetesBackend
from kubeflow.spark.options import (
    Annotations,
    Labels,
    Name,
    NodeSelector,
    PodTemplateOverride,
    Toleration,
)
from kubeflow.spark.test.common import FAILED, SUCCESS, TestCase


@pytest.fixture
def mock_k8s_backend():
    """Create a mock KubernetesBackend for testing."""
    backend = MagicMock(spec=KubernetesBackend)
    # Make isinstance check work
    backend.__class__ = KubernetesBackend
    return backend


@pytest.fixture
def mock_non_k8s_backend():
    """Create a mock non-Kubernetes backend for testing."""
    backend = MagicMock()
    backend.__class__ = MagicMock
    return backend


@pytest.fixture
def spark_connect_model():
    """Create a minimal SparkConnect model for testing."""
    return models.SparkV1alpha1SparkConnect(
        api_version=f"{constants.SPARK_CONNECT_GROUP}/{constants.SPARK_CONNECT_VERSION}",
        kind=constants.SPARK_CONNECT_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name="test-session",
            namespace="default",
        ),
        spec=models.SparkV1alpha1SparkConnectSpec(
            spark_version=constants.DEFAULT_SPARK_VERSION,
            image=constants.DEFAULT_SPARK_IMAGE,
            server=models.SparkV1alpha1ServerSpec(
                cores=constants.DEFAULT_DRIVER_CPU,
                memory=constants.DEFAULT_DRIVER_MEMORY,
            ),
            executor=models.SparkV1alpha1ExecutorSpec(
                instances=2,
                cores=constants.DEFAULT_EXECUTOR_CPU,
                memory=constants.DEFAULT_EXECUTOR_MEMORY,
            ),
        ),
    )


@pytest.fixture
def spark_application_model():
    """Create a minimal SparkApplication model for testing."""
    return models.SparkV1beta2SparkApplication(
        api_version=f"{constants.SPARK_APPLICATION_GROUP}/{constants.SPARK_APPLICATION_VERSION}",
        kind=constants.SPARK_APPLICATION_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name="test-job",
            namespace="default",
        ),
        spec=models.SparkV1beta2SparkApplicationSpec(
            spark_version=constants.DEFAULT_SPARK_VERSION,
            type="Python",
            mode="cluster",
            image=constants.DEFAULT_SPARK_IMAGE,
            mainApplicationFile="local:///opt/spark/examples/pi.py",
            driver=models.SparkV1beta2DriverSpec(),
            executor=models.SparkV1beta2ExecutorSpec(),
        ),
    )


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="apply labels",
            expected_status=SUCCESS,
            config={
                "labels": {
                    "app": "spark",
                    "team": "data-eng",
                },
            },
        ),
    ],
)
def test_labels_apply_to_crd(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Labels option."""

    print("Executing test:", test_case.name)

    option = Labels(test_case.config["labels"])

    for resource in [spark_connect_model, spark_application_model]:
        option(resource, mock_k8s_backend)

        for key, value in test_case.config["labels"].items():
            assert resource.metadata.labels[key] == value

    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="merge labels",
            expected_status=SUCCESS,
            config={
                "existing_labels": {
                    "existing": "label",
                },
                "new_labels": {
                    "new-label": "value",
                },
            },
        ),
    ],
)
def test_labels_merge_with_existing(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Labels option merging."""

    print("Executing test:", test_case.name)

    for resource in [spark_connect_model, spark_application_model]:
        resource.metadata.labels = test_case.config["existing_labels"].copy()

        option = Labels(test_case.config["new_labels"])
        option(resource, mock_k8s_backend)

        assert resource.metadata.labels["existing"] == "label"
        assert resource.metadata.labels["new-label"] == "value"

    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="incompatible backend",
            expected_status=FAILED,
            expected_error=ValueError,
            expected_output="not compatible",
            config={
                "labels": {
                    "app": "spark",
                },
            },
        ),
    ],
)
def test_labels_incompatible_backend(
    test_case: TestCase,
    mock_non_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Labels option with incompatible backend."""

    print("Executing test:", test_case.name)

    option = Labels(test_case.config["labels"])

    for resource in [spark_connect_model, spark_application_model]:
        with pytest.raises(
            test_case.expected_error,
            match=test_case.expected_output,
        ):
            option(resource, mock_non_k8s_backend)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="apply annotations",
            expected_status=SUCCESS,
            config={
                "annotations": {
                    "description": "ETL pipeline",
                    "owner": "data-team",
                },
            },
        ),
    ],
)
def test_annotations_apply_to_crd(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Annotations option."""

    print("Executing test:", test_case.name)

    option = Annotations(test_case.config["annotations"])

    for resource in [spark_connect_model, spark_application_model]:
        option(resource, mock_k8s_backend)

        for key, value in test_case.config["annotations"].items():
            assert resource.metadata.annotations[key] == value

    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="incompatible backend",
            expected_status=FAILED,
            expected_error=ValueError,
            expected_output="not compatible",
            config={
                "annotations": {
                    "description": "test",
                },
            },
        ),
    ],
)
def test_annotations_incompatible_backend(
    test_case: TestCase,
    mock_non_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Annotations option with incompatible backend."""

    print("Executing test:", test_case.name)

    option = Annotations(test_case.config["annotations"])

    for resource in [spark_connect_model, spark_application_model]:
        with pytest.raises(
            test_case.expected_error,
            match=test_case.expected_output,
        ):
            option(resource, mock_non_k8s_backend)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="apply node selector",
            expected_status=SUCCESS,
            config={
                "node_selector": {
                    "node-type": "spark",
                    "gpu": "true",
                },
            },
        ),
    ],
)
def test_node_selector_applies_to_both_roles(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test NodeSelector option."""

    print("Executing test:", test_case.name)

    option = NodeSelector(test_case.config["node_selector"])

    option(spark_connect_model, mock_k8s_backend)
    option(spark_application_model, mock_k8s_backend)

    connect_server = spark_connect_model.spec.server.template.spec.node_selector
    connect_executor = spark_connect_model.spec.executor.template.spec.node_selector

    app_driver = spark_application_model.spec.driver.node_selector
    app_executor = spark_application_model.spec.executor.node_selector

    for node_selector in [
        connect_server,
        connect_executor,
        app_driver,
        app_executor,
    ]:
        for key, value in test_case.config["node_selector"].items():
            assert node_selector[key] == value

    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="incompatible backend",
            expected_status=FAILED,
            expected_error=ValueError,
            expected_output="not compatible",
            config={
                "node_selector": {
                    "node-type": "spark",
                },
                "resource": "connect",
                "backend": "non-k8s",
            },
        ),
        TestCase(
            name="unsupported resource type",
            expected_status=FAILED,
            expected_error=TypeError,
            expected_output="Unsupported Spark resource type",
            config={
                "node_selector": {
                    "node-type": "spark",
                },
                "resource": "unsupported",
                "backend": "k8s",
            },
        ),
    ],
)
def test_node_selector_incompatible_backend(
    test_case: TestCase,
    mock_k8s_backend,
    mock_non_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test NodeSelector option with incompatible backend."""

    print("Executing test:", test_case.name)

    option = NodeSelector(test_case.config["node_selector"])

    backend = mock_k8s_backend if test_case.config["backend"] == "k8s" else mock_non_k8s_backend

    if test_case.config["resource"] == "connect":
        resource = spark_connect_model
    elif test_case.config["resource"] == "application":
        resource = spark_application_model
    else:
        resource = object()

    with pytest.raises(
        test_case.expected_error,
        match=test_case.expected_output,
    ):
        option(resource, backend)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="toleration with value",
            expected_status=SUCCESS,
            config={
                "key": "spark-workload",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
            },
        ),
    ],
)
def test_toleration_with_value(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Toleration option with value."""

    print("Executing test:", test_case.name)

    option = Toleration(
        key=test_case.config["key"],
        operator=test_case.config["operator"],
        value=test_case.config["value"],
        effect=test_case.config["effect"],
    )

    option(spark_connect_model, mock_k8s_backend)
    option(spark_application_model, mock_k8s_backend)

    connect_server_tolerations = spark_connect_model.spec.server.template.spec.tolerations
    connect_executor_tolerations = spark_connect_model.spec.executor.template.spec.tolerations

    app_driver_tolerations = spark_application_model.spec.driver.tolerations
    app_executor_tolerations = spark_application_model.spec.executor.tolerations

    for tolerations in [
        connect_server_tolerations,
        connect_executor_tolerations,
        app_driver_tolerations,
        app_executor_tolerations,
    ]:
        assert len(tolerations) == 1
        assert tolerations[0].key == test_case.config["key"]
        assert tolerations[0].operator == test_case.config["operator"]
        assert tolerations[0].value == test_case.config["value"]
        assert tolerations[0].effect == test_case.config["effect"]

    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="toleration without value",
            expected_status=SUCCESS,
            config={
                "key": "dedicated",
                "operator": "Exists",
                "effect": "NoSchedule",
            },
        ),
    ],
)
def test_toleration_without_value(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Toleration option without value."""

    print("Executing test:", test_case.name)

    option = Toleration(
        key=test_case.config["key"],
        operator=test_case.config["operator"],
        effect=test_case.config["effect"],
    )

    option(spark_connect_model, mock_k8s_backend)
    option(spark_application_model, mock_k8s_backend)

    connect_tolerations = spark_connect_model.spec.server.template.spec.tolerations
    app_tolerations = spark_application_model.spec.driver.tolerations

    for tolerations in [connect_tolerations, app_tolerations]:
        assert len(tolerations) == 1
        assert tolerations[0].key == test_case.config["key"]
        assert tolerations[0].operator == test_case.config["operator"]
        assert tolerations[0].value is None
        assert tolerations[0].effect == test_case.config["effect"]

    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="incompatible backend",
            expected_status=FAILED,
            expected_error=ValueError,
            expected_output="not compatible",
            config={
                "key": "test",
                "operator": "Exists",
                "resource": "connect",
                "backend": "non-k8s",
            },
        ),
        TestCase(
            name="unsupported resource type",
            expected_status=FAILED,
            expected_error=TypeError,
            expected_output="Unsupported Spark resource type",
            config={
                "key": "test",
                "operator": "Exists",
                "resource": "unsupported",
                "backend": "k8s",
            },
        ),
    ],
)
def test_toleration_incompatible_backend(
    test_case: TestCase,
    mock_k8s_backend,
    mock_non_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Toleration option with incompatible backend."""

    print("Executing test:", test_case.name)

    option = Toleration(
        key=test_case.config["key"],
        operator=test_case.config["operator"],
    )

    backend = mock_k8s_backend if test_case.config["backend"] == "k8s" else mock_non_k8s_backend

    if test_case.config["resource"] == "connect":
        resource = spark_connect_model
    elif test_case.config["resource"] == "application":
        resource = spark_application_model
    else:
        resource = object()

    with pytest.raises(
        test_case.expected_error,
        match=test_case.expected_output,
    ):
        option(resource, backend)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="driver pod template",
            expected_status=SUCCESS,
            config={
                "role": "driver",
                "template": {
                    "spec": {
                        "securityContext": {
                            "runAsUser": 1000,
                            "fsGroup": 1000,
                        }
                    }
                },
            },
        ),
        TestCase(
            name="executor pod template",
            expected_status=SUCCESS,
            config={
                "role": "executor",
                "template": {
                    "spec": {
                        "securityContext": {
                            "runAsUser": 1000,
                        }
                    }
                },
            },
        ),
    ],
)
def test_pod_template_override(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test PodTemplateOverride option."""

    print("Executing test:", test_case.name)

    option = PodTemplateOverride(
        role=test_case.config["role"],
        template=test_case.config["template"],
    )

    option(spark_connect_model, mock_k8s_backend)

    connect_crd = spark_connect_model.to_dict()

    if test_case.config["role"] == "driver":
        security_context = connect_crd["spec"]["server"]["template"]["spec"]["securityContext"]
    else:
        security_context = connect_crd["spec"]["executor"]["template"]["spec"]["securityContext"]

    for key, value in test_case.config["template"]["spec"]["securityContext"].items():
        assert security_context[key] == value

    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="invalid role",
            expected_status=FAILED,
            expected_error=ValueError,
            expected_output="Invalid role",
            config={
                "role": "invalid",
                "template": {"spec": {}},
                "resource": "connect",
            },
        ),
        TestCase(
            name="spark application not supported",
            expected_status=FAILED,
            expected_error=ValueError,
            expected_output="currently supported only for SparkConnect",
            config={
                "role": "driver",
                "template": {"spec": {}},
                "resource": "application",
            },
        ),
    ],
)
def test_pod_template_invalid_role(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test PodTemplateOverride option with invalid role."""

    print("Executing test:", test_case.name)

    option = PodTemplateOverride(
        role=test_case.config["role"],
        template=test_case.config["template"],
    )

    resource = (
        spark_connect_model
        if test_case.config["resource"] == "connect"
        else spark_application_model
    )

    with pytest.raises(
        test_case.expected_error,
        match=test_case.expected_output,
    ):
        option(resource, mock_k8s_backend)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="incompatible backend",
            expected_status=FAILED,
            expected_error=ValueError,
            expected_output="not compatible",
            config={
                "role": "driver",
                "template": {"spec": {}},
            },
        ),
    ],
)
def test_pod_template_incompatible_backend(
    test_case: TestCase,
    mock_non_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test PodTemplateOverride option with incompatible backend."""

    print("Executing test:", test_case.name)

    option = PodTemplateOverride(
        role=test_case.config["role"],
        template=test_case.config["template"],
    )

    for resource in [spark_connect_model, spark_application_model]:
        with pytest.raises(
            test_case.expected_error,
            match=test_case.expected_output,
        ):
            option(resource, mock_non_k8s_backend)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="basic name option",
            expected_status=SUCCESS,
            config={
                "name": "my-custom-session",
            },
        ),
    ],
)
def test_name_option_basic(test_case: TestCase):
    """Test Name option creation."""

    print("Executing test:", test_case.name)

    option = Name(test_case.config["name"])

    assert option.name == test_case.config["name"]
    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="apply name",
            expected_status=SUCCESS,
            config={
                "name": "new-session-name",
            },
        ),
    ],
)
def test_name_option_apply_to_crd(
    test_case: TestCase,
    mock_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Name option."""

    print("Executing test:", test_case.name)

    option = Name(test_case.config["name"])

    for resource in [spark_connect_model, spark_application_model]:
        option(resource, mock_k8s_backend)
        assert resource.metadata.name == test_case.config["name"]

    assert test_case.expected_status == SUCCESS

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="incompatible backend",
            expected_status=FAILED,
            expected_error=ValueError,
            expected_output="not compatible",
            config={
                "name": "test-session",
            },
        ),
    ],
)
def test_name_option_incompatible_backend(
    test_case: TestCase,
    mock_non_k8s_backend,
    spark_connect_model,
    spark_application_model,
):
    """Test Name option with incompatible backend."""

    print("Executing test:", test_case.name)

    option = Name(test_case.config["name"])

    for resource in [spark_connect_model, spark_application_model]:
        with pytest.raises(
            test_case.expected_error,
            match=test_case.expected_output,
        ):
            option(resource, mock_non_k8s_backend)

    print("test execution complete")
