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

"""
Unit tests for runtime_loader module.

Tests runtime loading from various sources including GitHub, HTTP, and filesystem.
"""

from unittest.mock import MagicMock, patch

import pytest

from kubeflow.trainer.backends.container import runtime_loader
from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types as base_types

# Sample runtime YAML data for testing
SAMPLE_RUNTIME_YAML = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "ClusterTrainingRuntime",
    "metadata": {
        "name": "torch-distributed",
        "labels": {"trainer.kubeflow.org/framework": "torch"},
    },
    "spec": {
        "mlPolicy": {"numNodes": 1},
        "template": {
            "spec": {
                "replicatedJobs": [
                    {
                        "name": "node",
                        "template": {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "containers": [
                                            {
                                                "name": "trainer",
                                                "image": "pytorch/pytorch:2.0.0",
                                            }
                                        ]
                                    }
                                }
                            }
                        },
                    }
                ]
            }
        },
    },
}


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="parse github url",
            expected_status=SUCCESS,
            config={
                "url": "github://kubeflow/trainer",
                "expected_type": "github",
                "expected_path": "kubeflow/trainer",
            },
        ),
        TestCase(
            name="parse github url with path",
            expected_status=SUCCESS,
            config={
                "url": "github://myorg/myrepo/custom/path",
                "expected_type": "github",
                "expected_path": "myorg/myrepo/custom/path",
            },
        ),
        TestCase(
            name="parse https url",
            expected_status=SUCCESS,
            config={
                "url": "https://example.com/runtime.yaml",
                "expected_type": "https",
                "expected_path": "https://example.com/runtime.yaml",
            },
        ),
        TestCase(
            name="parse http url",
            expected_status=SUCCESS,
            config={
                "url": "http://example.com/runtime.yaml",
                "expected_type": "http",
                "expected_path": "http://example.com/runtime.yaml",
            },
        ),
        TestCase(
            name="parse file url",
            expected_status=SUCCESS,
            config={
                "url": "file:///path/to/runtime.yaml",
                "expected_type": "file",
                "expected_path": "/path/to/runtime.yaml",
            },
        ),
        TestCase(
            name="parse absolute path",
            expected_status=SUCCESS,
            config={
                "url": "/absolute/path/to/runtime.yaml",
                "expected_type": "file",
                "expected_path": "/absolute/path/to/runtime.yaml",
            },
        ),
        TestCase(
            name="parse unsupported scheme",
            expected_status=FAILED,
            config={"url": "ftp://example.com/runtime.yaml"},
            expected_error=ValueError,
        ),
    ],
)
def test_parse_source_url(test_case):
    """Test parsing various source URL formats."""
    print("Executing test:", test_case.name)
    try:
        source_type, path = runtime_loader._parse_source_url(test_case.config["url"])

        assert test_case.expected_status == SUCCESS
        assert source_type == test_case.config["expected_type"]
        assert path == test_case.config["expected_path"]

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="load from default github",
            expected_status=SUCCESS,
            config={
                "github_path": "kubeflow/trainer",
                "discovered_files": ["torch_distributed.yaml"],
                "expected_runtime_name": "torch-distributed",
                "expected_framework": "torch",
            },
        ),
        TestCase(
            name="load from custom github",
            expected_status=SUCCESS,
            config={
                "github_path": "myorg/myrepo",
                "discovered_files": ["custom_runtime.yaml"],
                "expected_runtime_name": "custom-runtime",
                "expected_framework": "custom",
            },
        ),
        TestCase(
            name="load from github no files",
            expected_status=SUCCESS,
            config={
                "github_path": "kubeflow/trainer",
                "discovered_files": [],
                "expected_count": 0,
            },
        ),
        TestCase(
            name="load from github invalid path",
            expected_status=SUCCESS,
            config={
                "github_path": "invalid",
                "expected_count": 0,
            },
        ),
    ],
)
def test_load_from_github_url(test_case):
    """Test loading runtimes from GitHub URLs."""
    print("Executing test:", test_case.name)
    try:
        with (
            patch(
                "kubeflow.trainer.backends.container.runtime_loader._discover_github_runtime_files"
            ) as mock_discover,
            patch(
                "kubeflow.trainer.backends.container.runtime_loader._fetch_runtime_from_github"
            ) as mock_fetch,
        ):
            if test_case.name == "load from github invalid path":
                # Don't set up mocks for invalid path test
                runtimes = runtime_loader._load_from_github_url(test_case.config["github_path"])
                assert len(runtimes) == test_case.config["expected_count"]
            else:
                mock_discover.return_value = test_case.config.get("discovered_files", [])

                # Create runtime YAML with custom name/framework if specified
                runtime_yaml = SAMPLE_RUNTIME_YAML.copy()
                if "expected_runtime_name" in test_case.config:
                    runtime_yaml["metadata"]["name"] = test_case.config["expected_runtime_name"]
                    runtime_yaml["metadata"]["labels"]["trainer.kubeflow.org/framework"] = (
                        test_case.config["expected_framework"]
                    )
                mock_fetch.return_value = runtime_yaml

                runtimes = runtime_loader._load_from_github_url(test_case.config["github_path"])

                if "expected_count" in test_case.config:
                    assert len(runtimes) == test_case.config["expected_count"]
                else:
                    assert len(runtimes) == 1
                    assert runtimes[0].name == test_case.config["expected_runtime_name"]
                    assert runtimes[0].trainer.framework == test_case.config["expected_framework"]

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="priority order github sources",
            expected_status=SUCCESS,
            config={
                "sources": ["github://myorg/myrepo", "github://kubeflow/trainer"],
                "expected_count": 2,
                "expected_names": ["torch-distributed", "deepspeed-distributed"],
            },
        ),
        TestCase(
            name="duplicate runtime names skipped",
            expected_status=SUCCESS,
            config={
                "sources": ["github://myorg/myrepo", "github://kubeflow/trainer"],
                "duplicate_names": True,
                "expected_count": 1,
                "expected_names": ["torch-distributed"],
            },
        ),
        TestCase(
            name="fallback to defaults",
            expected_status=SUCCESS,
            config={
                "sources": ["github://myorg/myrepo"],
                "no_github_runtimes": True,
                "expected_count": 1,
                "expected_names": ["torch-distributed"],
            },
        ),
    ],
)
def test_list_training_runtimes_from_sources(test_case):
    """Test listing runtimes from multiple sources."""
    print("Executing test:", test_case.name)
    try:
        with (
            patch(
                "kubeflow.trainer.backends.container.runtime_loader._load_from_github_url"
            ) as mock_github,
            patch(
                "kubeflow.trainer.backends.container.runtime_loader._create_default_runtimes"
            ) as mock_defaults,
        ):
            if test_case.name == "priority order github sources":
                torch_runtime = base_types.Runtime(
                    name="torch-distributed",
                    trainer=base_types.RuntimeTrainer(
                        trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image="example.com/container",
                    ),
                )
                deepspeed_runtime = base_types.Runtime(
                    name="deepspeed-distributed",
                    trainer=base_types.RuntimeTrainer(
                        trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
                        framework="deepspeed",
                        num_nodes=1,
                        image="example.com/container",
                    ),
                )
                mock_github.side_effect = [[torch_runtime], [deepspeed_runtime]]
                mock_defaults.return_value = []

            elif test_case.name == "duplicate runtime names skipped":
                torch_runtime_1 = base_types.Runtime(
                    name="torch-distributed",
                    trainer=base_types.RuntimeTrainer(
                        trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image="example.com/container",
                    ),
                )
                torch_runtime_2 = base_types.Runtime(
                    name="torch-distributed",
                    trainer=base_types.RuntimeTrainer(
                        trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=2,
                        image="example.com/container",
                    ),
                )
                mock_github.side_effect = [[torch_runtime_1], [torch_runtime_2]]
                mock_defaults.return_value = []

            elif test_case.name == "fallback to defaults":
                mock_github.return_value = []
                default_runtime = base_types.Runtime(
                    name="torch-distributed",
                    trainer=base_types.RuntimeTrainer(
                        trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
                        framework="torch",
                        num_nodes=1,
                        image="example.com/container",
                    ),
                )
                mock_defaults.return_value = [default_runtime]

            runtimes = runtime_loader.list_training_runtimes_from_sources(
                test_case.config["sources"]
            )

            assert len(runtimes) == test_case.config["expected_count"]
            runtime_names = [r.name for r in runtimes]
            for expected_name in test_case.config["expected_names"]:
                assert expected_name in runtime_names

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_create_default_runtimes():
    """Test creating default runtimes from constants."""
    print("Executing test: create default runtimes")
    runtimes = runtime_loader._create_default_runtimes()

    assert len(runtimes) == len(constants.DEFAULT_FRAMEWORK_IMAGES)

    # Check torch runtime
    torch_runtimes = [r for r in runtimes if r.trainer.framework == "torch"]
    assert len(torch_runtimes) == 1
    assert torch_runtimes[0].name == "torch-distributed"
    assert torch_runtimes[0].trainer.trainer_type == base_types.TrainerType.CUSTOM_TRAINER
    assert torch_runtimes[0].trainer.num_nodes == 1
    # Verify default image is set
    assert torch_runtimes[0].trainer.image == constants.DEFAULT_FRAMEWORK_IMAGES["torch"]
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="discover runtime files",
            expected_status=SUCCESS,
            config={
                "html_content": """
                <html>
                    <a>torch_distributed.yaml</a>
                    <a>deepspeed_distributed.yaml</a>
                    <a>kustomization.yaml</a>
                </html>
                """,
                "expected_files": ["torch_distributed.yaml", "deepspeed_distributed.yaml"],
                "excluded_files": ["kustomization.yaml"],
            },
        ),
        TestCase(
            name="discover runtime files custom repo",
            expected_status=SUCCESS,
            config={
                "html_content": """
                <html>
                    <a>custom_runtime.yaml</a>
                </html>
                """,
                "expected_files": ["custom_runtime.yaml"],
                "owner": "myorg",
                "repo": "myrepo",
                "path": "custom/path",
            },
        ),
        TestCase(
            name="discover runtime files network error",
            expected_status=SUCCESS,
            config={
                "network_error": True,
                "expected_files": [],
            },
        ),
    ],
)
def test_discover_github_runtime_files(test_case):
    """Test discovering runtime files from GitHub."""
    print("Executing test:", test_case.name)
    try:
        with patch("urllib.request.urlopen") as mock_urlopen:
            if test_case.config.get("network_error"):
                mock_urlopen.side_effect = Exception("Network error")
            else:
                mock_response = MagicMock()
                mock_response.read.return_value = test_case.config["html_content"].encode("utf-8")
                mock_response.__enter__.return_value = mock_response
                mock_urlopen.return_value = mock_response

            kwargs = {}
            if "owner" in test_case.config:
                kwargs["owner"] = test_case.config["owner"]
                kwargs["repo"] = test_case.config["repo"]
                kwargs["path"] = test_case.config["path"]

            files = runtime_loader._discover_github_runtime_files(**kwargs)

            for expected_file in test_case.config["expected_files"]:
                assert expected_file in files

            for excluded_file in test_case.config.get("excluded_files", []):
                assert excluded_file not in files

            if "owner" in test_case.config and not test_case.config.get("network_error"):
                called_url = mock_urlopen.call_args[0][0]
                assert f"{kwargs['owner']}/{kwargs['repo']}" in called_url
                assert kwargs["path"] in called_url

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="fetch runtime success",
            expected_status=SUCCESS,
            config={
                "yaml_content": """
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: torch-distributed
""",
                "expected_name": "torch-distributed",
            },
        ),
        TestCase(
            name="fetch runtime custom repo",
            expected_status=SUCCESS,
            config={
                "yaml_content": """
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: custom-runtime
""",
                "expected_name": "custom-runtime",
                "runtime_file": "custom.yaml",
                "owner": "myorg",
                "repo": "myrepo",
                "path": "custom/path",
            },
        ),
        TestCase(
            name="fetch runtime network error",
            expected_status=SUCCESS,
            config={
                "network_error": True,
                "expected_none": True,
            },
        ),
    ],
)
def test_fetch_runtime_from_github(test_case):
    """Test fetching runtime YAML from GitHub."""
    print("Executing test:", test_case.name)
    try:
        with patch("urllib.request.urlopen") as mock_urlopen:
            if test_case.config.get("network_error"):
                mock_urlopen.side_effect = Exception("Network error")
            else:
                mock_response = MagicMock()
                mock_response.read.return_value = test_case.config["yaml_content"].encode("utf-8")
                mock_response.__enter__.return_value = mock_response
                mock_urlopen.return_value = mock_response

            default_runtime_file = "torch_distributed.yaml"
            kwargs = {"runtime_file": test_case.config.get("runtime_file", default_runtime_file)}
            if "owner" in test_case.config:
                kwargs["owner"] = test_case.config["owner"]
                kwargs["repo"] = test_case.config["repo"]
                kwargs["path"] = test_case.config["path"]

            data = runtime_loader._fetch_runtime_from_github(**kwargs)

            if test_case.config.get("expected_none"):
                assert data is None
            else:
                assert data is not None
                assert data["metadata"]["name"] == test_case.config["expected_name"]

                if "owner" in test_case.config:
                    called_url = mock_urlopen.call_args[0][0]
                    assert "raw.githubusercontent.com" in called_url
                    assert f"{kwargs['owner']}/{kwargs['repo']}" in called_url
                    assert f"{kwargs['path']}/{kwargs['runtime_file']}" in called_url

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="parse runtime yaml with custom image",
            expected_status=SUCCESS,
            config={
                "custom_image": "quay.io/custom/pytorch-arm:v1.0",
                "runtime_name": "torch-arm",
                "framework": "torch",
                "num_nodes": 2,
            },
        ),
        TestCase(
            name="parse runtime yaml with different custom image",
            expected_status=SUCCESS,
            config={
                "custom_image": "my-registry.io/pytorch:gpu-arm64",
                "runtime_name": "torch-gpu-arm",
                "framework": "torch",
                "num_nodes": 4,
            },
        ),
        TestCase(
            name="parse runtime yaml prefers container named node",
            expected_status=SUCCESS,
            config={
                "custom_image": "correct-node-image:v1.0",
                "runtime_name": "multi-container-runtime",
                "framework": "torch",
                "num_nodes": 1,
                "multiple_containers": True,
            },
        ),
    ],
)
def test_parse_runtime_yaml_extracts_image(test_case):
    """
    Test that _parse_runtime_yaml correctly extracts and stores the container image.
    This prevents regression of bugs where custom images are ignored.
    """
    print("Executing test:", test_case.name)
    try:
        # Create container list based on test case
        if test_case.config.get("multiple_containers"):
            # Test case with multiple containers - should prefer 'node' container
            containers = [
                {
                    "name": "sidecar",
                    "image": "wrong-sidecar-image:v1.0",
                },
                {
                    "name": "node",
                    "image": test_case.config["custom_image"],
                },
            ]
        else:
            # Single container test case
            containers = [
                {
                    "name": "trainer",
                    "image": test_case.config["custom_image"],
                }
            ]

        # Create runtime YAML with custom image
        runtime_yaml = {
            "kind": "ClusterTrainingRuntime",
            "metadata": {
                "name": test_case.config["runtime_name"],
                "labels": {"trainer.kubeflow.org/framework": test_case.config["framework"]},
            },
            "spec": {
                "mlPolicy": {"numNodes": test_case.config["num_nodes"]},
                "template": {
                    "spec": {
                        "replicatedJobs": [
                            {
                                "name": "node",
                                "template": {
                                    "spec": {"template": {"spec": {"containers": containers}}}
                                },
                            }
                        ]
                    }
                },
            },
        }

        runtime = runtime_loader._parse_runtime_yaml(runtime_yaml, "test")

        # Verify image is extracted and stored
        assert runtime.name == test_case.config["runtime_name"]
        assert runtime.trainer.framework == test_case.config["framework"]
        assert runtime.trainer.num_nodes == test_case.config["num_nodes"]
        assert runtime.trainer.image == test_case.config["custom_image"]

        assert test_case.expected_status == SUCCESS

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
