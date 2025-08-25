# Copyright 2024 The Kubeflow Authors.
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


from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, Optional

from kubeflow.trainer.constants import constants


# Configuration for the Custom Trainer.
@dataclass
class CustomTrainer:
    """Custom Trainer configuration. Configure the self-contained function
        that encapsulates the entire model training process.

    Args:
        func (`Callable`): The function that encapsulates the entire model training process.
        func_args (`Optional[Dict]`): The arguments to pass to the function.
        packages_to_install (`Optional[List[str]]`):
            A list of Python packages to install before running the function.
        pip_index_url (`Optional[str]`): The PyPI URL from which to install Python packages.
        num_nodes (`Optional[int]`): The number of nodes to use for training.
        resources_per_node (`Optional[Dict]`): The computing resources to allocate per node.
        env (`Optional[Dict[str, str]]`): The environment variables to set in the training nodes.
    """

    func: Callable
    func_args: Optional[Dict] = None
    packages_to_install: Optional[list[str]] = None
    pip_index_url: str = constants.DEFAULT_PIP_INDEX_URL
    num_nodes: Optional[int] = None
    resources_per_node: Optional[Dict] = None
    env: Optional[Dict[str, str]] = None


# TODO(Electronic-Waste): Add more loss functions.
# Loss function for the TorchTune LLM Trainer.
class Loss(Enum):
    """Loss function for the TorchTune LLM Trainer."""

    CEWithChunkedOutputLoss = "torchtune.modules.loss.CEWithChunkedOutputLoss"


# Data type for the TorchTune LLM Trainer.
class DataType(Enum):
    """Data type for the TorchTune LLM Trainer."""

    BF16 = "bf16"
    FP32 = "fp32"


# Data file type for the TorchTune LLM Trainer.
class DataFormat(Enum):
    """Data file type for the TorchTune LLM Trainer."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    ARROW = "arrow"
    TEXT = "text"
    XML = "xml"


# Configuration for the TorchTune Instruct dataset.
@dataclass
class TorchTuneInstructDataset:
    """
    Configuration for the custom dataset with user instruction prompts and model responses.
    REF: https://pytorch.org/torchtune/main/generated/torchtune.datasets.instruct_dataset.html

    Args:
        source (`Optional[DataFormat]`): Data file type.
        split (`Optional[str]`):
            The split of the dataset to use.  You can use this argument to load a subset of
            a given split, e.g. split="train[:10%]". Default is `train`.
        train_on_input (`Optional[bool]`):
            Whether the model is trained on the user prompt or not. Default is False.
        new_system_prompt (`Optional[str]`):
            The new system prompt to use. If specified, prepend a system message.
            This can serve as instructions to guide the model response. Default is None.
        column_map (`Optional[Dict[str, str]]`):
            A mapping to change the expected "input" and "output" column names to the actual
            column names in the dataset. Keys should be "input" and "output" and values should
            be the actual column names. Default is None, keeping the default "input" and
            "output" column names.
    """

    source: Optional[DataFormat] = None
    split: Optional[str] = None
    train_on_input: Optional[bool] = None
    new_system_prompt: Optional[str] = None
    column_map: Optional[Dict[str, str]] = None


# Configuration for the TorchTune LLM Trainer.
@dataclass
class TorchTuneConfig:
    """TorchTune LLM Trainer configuration. Configure the parameters in
        the TorchTune LLM Trainer that already includes the fine-tuning logic.

    Args:
        dtype (`Optional[Dtype]`):
            The underlying data type used to represent the model and optimizer parameters.
            Currently, we only support `bf16` and `fp32`.
        batch_size (`Optional[int]`):
            The number of samples processed before updating model weights.
        epochs (`Optional[int]`):
            The number of samples processed before updating model weights.
        loss (`Optional[Loss]`): The loss algorithm we use to fine-tune the LLM,
            e.g. `torchtune.modules.loss.CEWithChunkedOutputLoss`.
        num_nodes (`Optional[int]`): The number of nodes to use for training.
        dataset_preprocess_config (`Optional[TorchTuneInstructDataset]`):
            Configuration for the dataset preprocessing.
        resources_per_node (`Optional[Dict]`): The computing resources to allocate per node.
    """

    dtype: Optional[DataType] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    loss: Optional[Loss] = None
    num_nodes: Optional[int] = None
    dataset_preprocess_config: Optional[TorchTuneInstructDataset] = None
    resources_per_node: Optional[Dict] = None


# Configuration for the Builtin Trainer.
@dataclass
class BuiltinTrainer:
    """
    Builtin Trainer configuration. Configure the builtin trainer that already includes
        the fine-tuning logic, requiring only parameter adjustments.

    Args:
        config (`TorchTuneConfig`): The configuration for the builtin trainer.
    """

    config: TorchTuneConfig


# Change it to list: BUILTIN_CONFIGS, once we support more Builtin Trainer configs.
TORCH_TUNE = BuiltinTrainer.__annotations__["config"].__name__.lower().replace("config", "")


class TrainerType(Enum):
    CUSTOM_TRAINER = CustomTrainer.__name__
    BUILTIN_TRAINER = BuiltinTrainer.__name__


# Representation for the Trainer of the runtime.
@dataclass
class RuntimeTrainer:
    trainer_type: TrainerType
    framework: str
    num_nodes: int = 1  # The default value is set in the APIs.
    device: str = constants.UNKNOWN
    device_count: str = constants.UNKNOWN
    __command: tuple[str, ...] = field(init=False, repr=False)

    @property
    def command(self) -> tuple[str, ...]:
        return self.__command

    def set_command(self, command: tuple[str, ...]):
        self.__command = command


# Representation for the Training Runtime.
@dataclass
class Runtime:
    name: str
    trainer: RuntimeTrainer
    pretrained_model: Optional[str] = None


# Representation for the TrainJob steps.
@dataclass
class Step:
    name: str
    status: Optional[str]
    pod_name: str
    device: str = constants.UNKNOWN
    device_count: str = constants.UNKNOWN


# Representation for the TrainJob.
# TODO (andreyvelich): Discuss what fields users want to get.
@dataclass
class TrainJob:
    name: str
    creation_timestamp: datetime
    runtime: Runtime
    steps: list[Step]
    num_nodes: int
    status: Optional[str] = constants.UNKNOWN


# Configuration for the HuggingFace dataset initializer.
# TODO (andreyvelich): Discuss how to keep these configurations is sync with pkg.initializers.types
@dataclass
class HuggingFaceDatasetInitializer:
    storage_uri: str
    access_token: Optional[str] = None


# Configuration for the HuggingFace model initializer.
@dataclass
class HuggingFaceModelInitializer:
    storage_uri: str
    access_token: Optional[str] = None


@dataclass
class Initializer:
    """Initializer defines configurations for dataset and pre-trained model initialization

    Args:
        dataset (`Optional[HuggingFaceDatasetInitializer]`): The configuration for one of the
            supported dataset initializers.
        model (`Optional[HuggingFaceModelInitializer]`): The configuration for one of the
            supported model initializers.
    """

    dataset: Optional[HuggingFaceDatasetInitializer] = None
    model: Optional[HuggingFaceModelInitializer] = None
