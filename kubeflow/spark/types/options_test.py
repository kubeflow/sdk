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

import pytest

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


class TestLabels:
    """Tests for Labels option."""

    def test_labels_apply_to_crd(self, mock_k8s_backend):
        """Labels option adds labels to CRD metadata."""
        option = Labels({"app": "spark", "team": "data-eng"})
        crd = {}

        option(crd, mock_k8s_backend)

        assert crd["metadata"]["labels"]["app"] == "spark"
        assert crd["metadata"]["labels"]["team"] == "data-eng"

    def test_labels_merge_with_existing(self, mock_k8s_backend):
        """Labels option merges with existing labels."""
        option = Labels({"new-label": "value"})
        crd = {"metadata": {"labels": {"existing": "label"}}}

        option(crd, mock_k8s_backend)

        assert crd["metadata"]["labels"]["existing"] == "label"
        assert crd["metadata"]["labels"]["new-label"] == "value"

    def test_labels_incompatible_backend(self, mock_non_k8s_backend):
        """Labels option raises error for incompatible backend."""
        option = Labels({"app": "spark"})
        crd = {}

        with pytest.raises(ValueError, match="not compatible"):
            option(crd, mock_non_k8s_backend)


class TestAnnotations:
    """Tests for Annotations option."""

    def test_annotations_apply_to_crd(self, mock_k8s_backend):
        """Annotations option adds annotations to CRD metadata."""
        option = Annotations({"description": "ETL pipeline", "owner": "data-team"})
        crd = {}

        option(crd, mock_k8s_backend)

        assert crd["metadata"]["annotations"]["description"] == "ETL pipeline"
        assert crd["metadata"]["annotations"]["owner"] == "data-team"

    def test_annotations_incompatible_backend(self, mock_non_k8s_backend):
        """Annotations option raises error for incompatible backend."""
        option = Annotations({"description": "test"})
        crd = {}

        with pytest.raises(ValueError, match="not compatible"):
            option(crd, mock_non_k8s_backend)


class TestNodeSelector:
    """Tests for NodeSelector option."""

    def test_node_selector_applies_to_both_roles(self, mock_k8s_backend):
        """NodeSelector option adds selectors to both driver and executor."""
        option = NodeSelector({"node-type": "spark", "gpu": "true"})
        crd = {}

        option(crd, mock_k8s_backend)

        assert crd["spec"]["server"]["template"]["spec"]["nodeSelector"]["node-type"] == "spark"
        assert crd["spec"]["server"]["template"]["spec"]["nodeSelector"]["gpu"] == "true"
        assert crd["spec"]["executor"]["template"]["spec"]["nodeSelector"]["node-type"] == "spark"
        assert crd["spec"]["executor"]["template"]["spec"]["nodeSelector"]["gpu"] == "true"

    def test_node_selector_incompatible_backend(self, mock_non_k8s_backend):
        """NodeSelector option raises error for incompatible backend."""
        option = NodeSelector({"node-type": "spark"})
        crd = {}

        with pytest.raises(ValueError, match="not compatible"):
            option(crd, mock_non_k8s_backend)


class TestToleration:
    """Tests for Toleration option."""

    def test_toleration_with_value(self, mock_k8s_backend):
        """Toleration option with value."""
        option = Toleration(
            key="spark-workload",
            operator="Equal",
            value="true",
            effect="NoSchedule",
        )
        crd = {}

        option(crd, mock_k8s_backend)

        tolerations = crd["spec"]["server"]["template"]["spec"]["tolerations"]
        assert len(tolerations) == 1
        assert tolerations[0]["key"] == "spark-workload"
        assert tolerations[0]["operator"] == "Equal"
        assert tolerations[0]["value"] == "true"
        assert tolerations[0]["effect"] == "NoSchedule"

    def test_toleration_without_value(self, mock_k8s_backend):
        """Toleration option without value (operator=Exists)."""
        option = Toleration(
            key="dedicated",
            operator="Exists",
            effect="NoSchedule",
        )
        crd = {}

        option(crd, mock_k8s_backend)

        tolerations = crd["spec"]["server"]["template"]["spec"]["tolerations"]
        assert len(tolerations) == 1
        assert tolerations[0]["key"] == "dedicated"
        assert tolerations[0]["operator"] == "Exists"
        assert "value" not in tolerations[0]  # Value not included when empty
        assert tolerations[0]["effect"] == "NoSchedule"

    def test_toleration_incompatible_backend(self, mock_non_k8s_backend):
        """Toleration option raises error for incompatible backend."""
        option = Toleration(key="test", operator="Exists")
        crd = {}

        with pytest.raises(ValueError, match="not compatible"):
            option(crd, mock_non_k8s_backend)


class TestPodTemplateOverride:
    """Tests for PodTemplateOverride option."""

    def test_pod_template_driver(self, mock_k8s_backend):
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
        crd = {}

        option(crd, mock_k8s_backend)

        assert crd["spec"]["server"]["template"]["spec"]["securityContext"]["runAsUser"] == 1000
        assert crd["spec"]["server"]["template"]["spec"]["securityContext"]["fsGroup"] == 1000

    def test_pod_template_executor(self, mock_k8s_backend):
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
        crd = {}

        option(crd, mock_k8s_backend)

        assert crd["spec"]["executor"]["template"]["spec"]["securityContext"]["runAsUser"] == 1000

    def test_pod_template_invalid_role(self, mock_k8s_backend):
        """PodTemplateOverride raises error for invalid role."""
        option = PodTemplateOverride(role="invalid", template={"spec": {}})
        crd = {}

        with pytest.raises(ValueError, match="Invalid role"):
            option(crd, mock_k8s_backend)

    def test_pod_template_incompatible_backend(self, mock_non_k8s_backend):
        """PodTemplateOverride option raises error for incompatible backend."""
        option = PodTemplateOverride(role="driver", template={"spec": {}})
        crd = {}

        with pytest.raises(ValueError, match="not compatible"):
            option(crd, mock_non_k8s_backend)


class TestNameOption:
    """Tests for Name option."""

    def test_name_option_basic(self):
        """Create Name option with valid name."""
        option = Name("my-custom-session")
        assert option.name == "my-custom-session"

    def test_name_option_apply_to_crd(self, mock_k8s_backend):
        """Apply Name option to CRD."""
        option = Name("test-session")
        crd = {"metadata": {"name": "old-name", "namespace": "default"}}

        option(crd, mock_k8s_backend)

        assert crd["metadata"]["name"] == "test-session"

    def test_name_option_creates_metadata(self, mock_k8s_backend):
        """Name option creates metadata if missing."""
        option = Name("test-session")
        crd = {}

        option(crd, mock_k8s_backend)

        assert crd["metadata"]["name"] == "test-session"

    def test_name_option_incompatible_backend(self, mock_non_k8s_backend):
        """Name option raises error for incompatible backend."""
        option = Name("test-session")
        crd = {}

        with pytest.raises(ValueError, match="not compatible"):
            option(crd, mock_non_k8s_backend)
