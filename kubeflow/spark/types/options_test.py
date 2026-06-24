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
from kubeflow.spark.types.options import (
    Annotations,
    Labels,
    Name,
    NodeSelector,
    PodTemplateOverride,
    Toleration,
)


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


class TestLabels:
    """Tests for Labels option."""

    def test_labels_apply_to_crd(self, mock_k8s_backend, spark_connect_model):
        """Labels option adds labels to CRD metadata."""
        option = Labels({"app": "spark", "team": "data-eng"})

        option(spark_connect_model, mock_k8s_backend)

        assert spark_connect_model.metadata.labels["app"] == "spark"
        assert spark_connect_model.metadata.labels["team"] == "data-eng"

    def test_labels_merge_with_existing(self, mock_k8s_backend, spark_connect_model):
        """Labels option merges with existing labels."""
        spark_connect_model.metadata.labels = {"existing": "label"}
        option = Labels({"new-label": "value"})

        option(spark_connect_model, mock_k8s_backend)

        assert spark_connect_model.metadata.labels["existing"] == "label"
        assert spark_connect_model.metadata.labels["new-label"] == "value"

    def test_labels_incompatible_backend(self, mock_non_k8s_backend, spark_connect_model):
        """Labels option raises error for incompatible backend."""
        option = Labels({"app": "spark"})

        with pytest.raises(ValueError, match="not compatible"):
            option(spark_connect_model, mock_non_k8s_backend)


class TestAnnotations:
    """Tests for Annotations option."""

    def test_annotations_apply_to_crd(self, mock_k8s_backend, spark_connect_model):
        """Annotations option adds annotations to CRD metadata."""
        option = Annotations({"description": "ETL pipeline", "owner": "data-team"})

        option(spark_connect_model, mock_k8s_backend)

        assert spark_connect_model.metadata.annotations["description"] == "ETL pipeline"
        assert spark_connect_model.metadata.annotations["owner"] == "data-team"

    def test_annotations_incompatible_backend(self, mock_non_k8s_backend, spark_connect_model):
        """Annotations option raises error for incompatible backend."""
        option = Annotations({"description": "test"})

        with pytest.raises(ValueError, match="not compatible"):
            option(spark_connect_model, mock_non_k8s_backend)


class TestNodeSelector:
    """Tests for NodeSelector option."""

    def test_node_selector_applies_to_both_roles(self, mock_k8s_backend, spark_connect_model):
        """NodeSelector option adds selectors to both driver and executor."""
        option = NodeSelector({"node-type": "spark", "gpu": "true"})

        option(spark_connect_model, mock_k8s_backend)

        # Check server (driver)
        server_node_selector = spark_connect_model.spec.server.template.spec.node_selector
        assert server_node_selector["node-type"] == "spark"
        assert server_node_selector["gpu"] == "true"
        # Check executor
        executor_node_selector = spark_connect_model.spec.executor.template.spec.node_selector
        assert executor_node_selector["node-type"] == "spark"
        assert executor_node_selector["gpu"] == "true"

    def test_node_selector_incompatible_backend(self, mock_non_k8s_backend, spark_connect_model):
        """NodeSelector option raises error for incompatible backend."""
        option = NodeSelector({"node-type": "spark"})

        with pytest.raises(ValueError, match="not compatible"):
            option(spark_connect_model, mock_non_k8s_backend)


class TestToleration:
    """Tests for Toleration option."""

    def test_toleration_with_value(self, mock_k8s_backend, spark_connect_model):
        """Toleration option with value."""
        option = Toleration(
            key="spark-workload",
            operator="Equal",
            value="true",
            effect="NoSchedule",
        )

        option(spark_connect_model, mock_k8s_backend)

        tolerations = spark_connect_model.spec.server.template.spec.tolerations
        assert len(tolerations) == 1
        assert tolerations[0].key == "spark-workload"
        assert tolerations[0].operator == "Equal"
        assert tolerations[0].value == "true"
        assert tolerations[0].effect == "NoSchedule"

    def test_toleration_without_value(self, mock_k8s_backend, spark_connect_model):
        """Toleration option without value (operator=Exists)."""
        option = Toleration(
            key="dedicated",
            operator="Exists",
            effect="NoSchedule",
        )

        option(spark_connect_model, mock_k8s_backend)

        tolerations = spark_connect_model.spec.server.template.spec.tolerations
        assert len(tolerations) == 1
        assert tolerations[0].key == "dedicated"
        assert tolerations[0].operator == "Exists"
        assert tolerations[0].value is None  # Value is None when empty
        assert tolerations[0].effect == "NoSchedule"

    def test_toleration_incompatible_backend(self, mock_non_k8s_backend, spark_connect_model):
        """Toleration option raises error for incompatible backend."""
        option = Toleration(key="test", operator="Exists")

        with pytest.raises(ValueError, match="not compatible"):
            option(spark_connect_model, mock_non_k8s_backend)


class TestPodTemplateOverride:
    """Tests for PodTemplateOverride option."""

    def test_pod_template_driver(self, mock_k8s_backend, spark_connect_model):
        """PodTemplateOverride applies to driver."""
        option = PodTemplateOverride(
            role="driver",
            template={
                "spec": {
                    "securityContext": {
                        "runAsUser": 1000,
                        "fsGroup": 1000,
                    }
                }
            },
        )

        option(spark_connect_model, mock_k8s_backend)

        # Convert to dict to verify merged template
        crd = spark_connect_model.to_dict()
        assert crd["spec"]["server"]["template"]["spec"]["securityContext"]["runAsUser"] == 1000
        assert crd["spec"]["server"]["template"]["spec"]["securityContext"]["fsGroup"] == 1000

    def test_pod_template_executor(self, mock_k8s_backend, spark_connect_model):
        """PodTemplateOverride applies to executor."""
        option = PodTemplateOverride(
            role="executor",
            template={
                "spec": {
                    "securityContext": {
                        "runAsUser": 1000,
                    }
                }
            },
        )

        option(spark_connect_model, mock_k8s_backend)

        # Convert to dict to verify merged template
        crd = spark_connect_model.to_dict()
        assert crd["spec"]["executor"]["template"]["spec"]["securityContext"]["runAsUser"] == 1000

    def test_pod_template_invalid_role(self, mock_k8s_backend, spark_connect_model):
        """PodTemplateOverride raises error for invalid role."""
        option = PodTemplateOverride(role="invalid", template={"spec": {}})

        with pytest.raises(ValueError, match="Invalid role"):
            option(spark_connect_model, mock_k8s_backend)

    def test_pod_template_incompatible_backend(self, mock_non_k8s_backend, spark_connect_model):
        """PodTemplateOverride option raises error for incompatible backend."""
        option = PodTemplateOverride(role="driver", template={"spec": {}})

        with pytest.raises(ValueError, match="not compatible"):
            option(spark_connect_model, mock_non_k8s_backend)


class TestNameOption:
    """Tests for Name option."""

    def test_name_option_basic(self):
        """Create Name option with valid name."""
        option = Name("my-custom-session")
        assert option.name == "my-custom-session"

    def test_name_option_apply_to_crd(self, mock_k8s_backend, spark_connect_model):
        """Apply Name option to CRD."""
        option = Name("new-session-name")

        option(spark_connect_model, mock_k8s_backend)

        assert spark_connect_model.metadata.name == "new-session-name"

    def test_name_option_incompatible_backend(self, mock_non_k8s_backend, spark_connect_model):
        """Name option raises error for incompatible backend."""
        option = Name("test-session")

        with pytest.raises(ValueError, match="not compatible"):
            option(spark_connect_model, mock_non_k8s_backend)
