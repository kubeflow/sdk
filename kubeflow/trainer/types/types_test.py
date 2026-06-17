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

from unittest.mock import MagicMock, patch

import pytest

from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid datacacheinitializer creation",
            expected_status=SUCCESS,
            config={
                "storage_uri": "cache://test_schema/test_table",
                "num_data_nodes": 3,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_output=None,
        ),
        TestCase(
            name="invalid num_data_nodes raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema/test_table",
                "num_data_nodes": 1,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="zero num_data_nodes raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema/test_table",
                "num_data_nodes": 0,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="negative num_data_nodes raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema/test_table",
                "num_data_nodes": -1,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri without cache:// prefix raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "invalid://test_schema/test_table",
                "num_data_nodes": 3,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri format raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema",
                "num_data_nodes": 3,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri with too many parts raises ValueError",
            expected_status=FAILED,
            config={
                "storage_uri": "cache://test_schema/test_table/extra",
                "num_data_nodes": 3,
                "metadata_loc": "gs://my-bucket/metadata",
            },
            expected_error=ValueError,
        ),
    ],
)
def test_data_cache_initializer(test_case: TestCase):
    """Test DataCacheInitializer creation and validation."""
    print("Executing test:", test_case.name)

    try:
        initializer = types.DataCacheInitializer(
            storage_uri=test_case.config["storage_uri"],
            num_data_nodes=test_case.config["num_data_nodes"],
            metadata_loc=test_case.config["metadata_loc"],
        )

        assert test_case.expected_status == SUCCESS
        # Only check the fields that were passed in config, not auto-generated ones
        for key in test_case.config:
            assert getattr(initializer, key) == test_case.config[key]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_trainjob_template_train():
    """Test TrainJobTemplate.train() delegates to TrainerClient."""
    template = types.TrainJobTemplate(
        trainer=types.CustomTrainer(func=lambda: None),
        runtime="torch-distributed",
        initializer=None,
    )

    with patch("kubeflow.trainer.api.trainer_client.TrainerClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.train.return_value = "test-job-name"
        mock_client_cls.return_value = mock_client

        result = template.train()

        mock_client.train.assert_called_once_with(
            runtime="torch-distributed",
            initializer=None,
            trainer=template.trainer,
            options=None,
        )
        assert result == "test-job-name"


def test_trainjob_template_optimize():
    """Test TrainJobTemplate.optimize() delegates to OptimizerClient."""
    template = types.TrainJobTemplate(
        trainer=types.CustomTrainer(func=lambda: None),
        runtime="torch-distributed",
        initializer=None,
    )
    search_space = {"lr": 0.01}

    with patch("kubeflow.optimizer.api.optimizer_client.OptimizerClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.optimize.return_value = "test-experiment-name"
        mock_client_cls.return_value = mock_client

        result = template.optimize(search_space=search_space)

        mock_client.optimize.assert_called_once_with(
            trial_template=template,
            trial_config=None,
            search_space=search_space,
            objectives=None,
            algorithm=None,
        )
        assert result == "test-experiment-name"
