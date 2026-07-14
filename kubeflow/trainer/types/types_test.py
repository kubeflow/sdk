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


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid storage_uri with user and model",
            expected_status=SUCCESS,
            config={"storage_uri": "hf://user/model"},
        ),
        TestCase(
            name="invalid storage_uri without hf prefix raises ValueError",
            expected_status=FAILED,
            config={"storage_uri": "user/model"},
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri without repo path raises ValueError",
            expected_status=FAILED,
            config={"storage_uri": "hf://model"},
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri with user but no repo raises ValueError",
            expected_status=FAILED,
            config={"storage_uri": "hf://user/"},
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri with empty user raises ValueError",
            expected_status=FAILED,
            config={"storage_uri": "hf:///model"},
            expected_error=ValueError,
        ),
    ],
)
def test_hugging_face_model_initializer(test_case: TestCase):
    """Test HuggingFaceModelInitializer creation and validation."""
    print("Executing test:", test_case.name)

    try:
        initializer = types.HuggingFaceModelInitializer(
            storage_uri=test_case.config["storage_uri"],
        )

        assert test_case.expected_status == SUCCESS
        assert initializer.storage_uri == test_case.config["storage_uri"]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid storage_uri with user and dataset",
            expected_status=SUCCESS,
            config={"storage_uri": "hf://user/dataset"},
        ),
        TestCase(
            name="invalid storage_uri without hf prefix raises ValueError",
            expected_status=FAILED,
            config={"storage_uri": "user/dataset"},
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri without repo path raises ValueError",
            expected_status=FAILED,
            config={"storage_uri": "hf://dataset"},
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri with user but no repo raises ValueError",
            expected_status=FAILED,
            config={"storage_uri": "hf://user/"},
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid storage_uri with empty user raises ValueError",
            expected_status=FAILED,
            config={"storage_uri": "hf:///dataset"},
            expected_error=ValueError,
        ),
    ],
)
def test_hugging_face_dataset_initializer(test_case: TestCase):
    """Test HuggingFaceDatasetInitializer creation and validation."""
    print("Executing test:", test_case.name)

    try:
        initializer = types.HuggingFaceDatasetInitializer(
            storage_uri=test_case.config["storage_uri"],
        )

        assert test_case.expected_status == SUCCESS
        assert initializer.storage_uri == test_case.config["storage_uri"]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid lora config with positive values",
            expected_status=SUCCESS,
            config={
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.5,
            },
        ),
        TestCase(
            name="valid lora config with boundary dropout=0.0",
            expected_status=SUCCESS,
            config={
                "lora_dropout": 0.0,
            },
        ),
        TestCase(
            name="valid lora config with boundary dropout=1.0",
            expected_status=SUCCESS,
            config={
                "lora_dropout": 1.0,
            },
        ),
        TestCase(
            name="invalid lora_rank negative raises ValueError",
            expected_status=FAILED,
            config={
                "lora_rank": -8,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid lora_rank zero raises ValueError",
            expected_status=FAILED,
            config={
                "lora_rank": 0,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid lora_alpha negative raises ValueError",
            expected_status=FAILED,
            config={
                "lora_alpha": -1,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid lora_alpha zero raises ValueError",
            expected_status=FAILED,
            config={
                "lora_alpha": 0,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid lora_dropout above 1.0 raises ValueError",
            expected_status=FAILED,
            config={
                "lora_dropout": 1.5,
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="invalid lora_dropout negative raises ValueError",
            expected_status=FAILED,
            config={
                "lora_dropout": -0.1,
            },
            expected_error=ValueError,
        ),
    ],
)
def test_lora_config_validation(test_case: TestCase):
    """Test LoraConfig creation and validation."""
    print("Executing test:", test_case.name)

    try:
        config = types.LoraConfig(**test_case.config)

        assert test_case.expected_status == SUCCESS
        for key in test_case.config:
            assert getattr(config, key) == test_case.config[key]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
    print("test execution complete")
