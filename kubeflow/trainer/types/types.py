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
from typing import Callable, Optional, Union

from kubeflow.trainer.constants import constants


# Configuration for the Custom Trainer.
@dataclass
class CustomTrainer:
    """Custom Trainer configuration. Configure the self-contained function
        that encapsulates the entire model training process.

    Args:
        func (`Callable`): The function that encapsulates the entire model training process.
        func_args (`Optional[dict]`): The arguments to pass to the function.
        packages_to_install (`Optional[list[str]]`):
            A list of Python packages to install before running the function.
        pip_index_urls (`list[str]`): The PyPI URLs from which to install
            Python packages. The first URL will be the index-url, and remaining ones
            are extra-index-urls.
        num_nodes (`Optional[int]`): The number of nodes to use for training.
        resources_per_node (`Optional[dict]`): The computing resources to allocate per node.
        env (`Optional[dict[str, str]]`): The environment variables to set in the training nodes.
    """

    func: Callable
    func_args: Optional[dict] = None
    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    env: Optional[dict[str, str]] = None


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
    column_map: Optional[dict[str, str]] = None


@dataclass
class LoraConfig:
    """Configuration for the LoRA/QLoRA/DoRA.
    REF: https://meta-pytorch.org/torchtune/main/tutorials/memory_optimizations.html

    Args:
        apply_lora_to_mlp (`Optional[bool]`):
            Whether to apply LoRA to the MLP in each transformer layer.
        apply_lora_to_output (`Optional[bool]`):
            Whether to apply LoRA to the model's final output projection.
        lora_attn_modules (`list[str]`):
            A list of strings specifying which layers of the model to apply LoRA,
            default is ["q_proj", "v_proj", "output_proj"]:
            1. "q_proj" applies LoRA to the query projection layer.
            2. "k_proj" applies LoRA to the key projection layer.
            3. "v_proj" applies LoRA to the value projection layer.
            4. "output_proj" applies LoRA to the attention output projection layer.
        lora_rank (`Optional[int]`): The rank of the low rank decomposition.
        lora_alpha (`Optional[int]`):
            The scaling factor that adjusts the magnitude of the low-rank matrices' output.
        lora_dropout (`Optional[float]`):
            The probability of applying Dropout to the low rank updates.
        quantize_base (`Optional[bool]`): Whether to enable model quantization.
        use_dora (`Optional[bool]`): Whether to enable DoRA.
    """

    apply_lora_to_mlp: Optional[bool] = None
    apply_lora_to_output: Optional[bool] = None
    lora_attn_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "output_proj"]
    )
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    quantize_base: Optional[bool] = None
    use_dora: Optional[bool] = None


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
        peft_config (`Optional[LoraConfig]`):
            Configuration for the PEFT(Parameter-Efficient Fine-Tuning),
            including LoRA/QLoRA/DoRA, etc.
        dataset_preprocess_config (`Optional[TorchTuneInstructDataset]`):
            Configuration for the dataset preprocessing.
        resources_per_node (`Optional[Dict]`): The computing resources to allocate per node.
    """

    dtype: Optional[DataType] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    loss: Optional[Loss] = None
    num_nodes: Optional[int] = None
    peft_config: Optional[LoraConfig] = None
    dataset_preprocess_config: Optional[TorchTuneInstructDataset] = None
    resources_per_node: Optional[dict] = None


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
    status: str = constants.UNKNOWN


# Configuration for the HuggingFace dataset initializer.
# TODO (andreyvelich): Discuss how to keep these configurations is sync with pkg.initializers.types
@dataclass
class HuggingFaceDatasetInitializer:
    """Configuration for downloading datasets from HuggingFace Hub."""

    storage_uri: str
    access_token: Optional[str] = None


@dataclass
class DataCacheInitializer:
    """Configuration for distributed data caching system for training workloads.

    Args:
        storage_uri (`str`): The URI for the cached data in the format
            'cache://<SCHEMA_NAME>/<TABLE_NAME>'. This specifies the location
            where the data cache will be stored and accessed.
        metadata_loc (`str`): The metadata file path of an iceberg table.
        num_data_nodes (`int`): The number of data nodes in the distributed cache
            system. Must be greater than 1.
        head_cpu (`Optional[str]`): The CPU resources to allocate for the cache head node.
        head_mem (`Optional[str]`): The memory resources to allocate for the cache head node.
        worker_cpu (`Optional[str]`): The CPU resources to allocate for each cache worker node.
        worker_mem (`Optional[str]`): The memory resources to allocate for each cache worker node.
        iam_role (`Optional[str]`): The IAM role to use for accessing metadata_loc file.
    """

    storage_uri: str
    metadata_loc: str
    num_data_nodes: int
    head_cpu: Optional[str] = None
    head_mem: Optional[str] = None
    worker_cpu: Optional[str] = None
    worker_mem: Optional[str] = None
    iam_role: Optional[str] = None

    def __post_init__(self):
        """Validate DataCacheInitializer parameters."""
        if self.num_data_nodes <= 1:
            raise ValueError(f"num_data_nodes must be greater than 1, got {self.num_data_nodes}")

        # Validate storage_uri format
        if not self.storage_uri.startswith("cache://"):
            raise ValueError(f"storage_uri must start with 'cache://', got {self.storage_uri}")

        uri_path = self.storage_uri[len("cache://") :]
        parts = uri_path.split("/")

        if len(parts) != 2:
            raise ValueError(
                f"storage_uri must be in format "
                f"'cache://<SCHEMA_NAME>/<TABLE_NAME>', got {self.storage_uri}"
            )


# Configuration for the HuggingFace model initializer.
@dataclass
class HuggingFaceModelInitializer:
    storage_uri: str
    access_token: Optional[str] = None


@dataclass
class Initializer:
    """Initializer defines configurations for dataset and pre-trained model initialization

    Args:
        dataset (`Optional[Union[HuggingFaceDatasetInitializer, DataCacheInitializer]]`):
            The configuration for one of the supported dataset initializers.
        model (`Optional[HuggingFaceModelInitializer]`): The configuration for one of the
            supported model initializers.
    """

    dataset: Optional[Union[HuggingFaceDatasetInitializer, DataCacheInitializer]] = None
    model: Optional[HuggingFaceModelInitializer] = None
