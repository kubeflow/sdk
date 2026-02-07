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

from dataclasses import fields
import inspect
import os
import textwrap
from typing import Any, Callable, Optional, Union
from urllib.parse import urlparse

from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


def get_container_devices(
    resources: Optional[models.IoK8sApiCoreV1ResourceRequirements],
) -> Optional[tuple[str, str]]:
    """
    Get the device type and device count for the given container.
    """

    # If containers resource limits are empty, return Unknown.
    if resources is None or resources.limits is None:
        return None

    # TODO (andreyvelich): We should discuss how to get container device type.
    # Potentially, we can use the trainer.kubeflow.org/device label from the runtime or
    # node types.
    # TODO (andreyvelich): Support other resource labels (e.g. NPUs).
    if constants.GPU_LABEL in resources.limits:
        device = constants.GPU_LABEL.split("/")[1]
        device_count = resources.limits[constants.GPU_LABEL].actual_instance
    elif constants.TPU_LABEL in resources.limits:
        device = constants.TPU_LABEL.split("/")[1]
        device_count = resources.limits[constants.TPU_LABEL].actual_instance
    elif any(k.startswith(constants.GPU_MIG_PREFIX) for k in resources.limits):
        mig_keys = [k for k in resources.limits if k.startswith(constants.GPU_MIG_PREFIX)]
        if len(mig_keys) > 1:
            raise ValueError(f"Multiple MIG resource types are not supported yet: {mig_keys}")
        mig_key = mig_keys[0]
        device = mig_key.split("/")[1]
        device_count = resources.limits[mig_key].actual_instance
    elif constants.CPU_LABEL in resources.limits:
        device = constants.CPU_LABEL
        device_count = resources.limits[constants.CPU_LABEL].actual_instance
    else:
        raise Exception(f"Unknown device type in the container resources: {resources.limits}")
    if device_count is None:
        raise Exception(f"Failed to get device count for resources: {resources.limits}")

    return device, str(device_count)


def get_runtime_trainer_container(
    replicated_jobs: list[models.JobsetV1alpha2ReplicatedJob],
) -> Optional[models.IoK8sApiCoreV1Container]:
    """
    Get the runtime node container from the given replicated jobs.
    """

    for rjob in replicated_jobs:
        if not (rjob.template.spec and rjob.template.spec.template.spec):
            raise Exception(f"Invalid ReplicatedJob template: {rjob}")
        # The ancestor label defines Trainer container in the ReplicatedJobs.
        if not (
            rjob.template.metadata
            and rjob.template.metadata.labels
            and constants.TRAINJOB_ANCESTOR_LABEL in rjob.template.metadata.labels
        ):
            continue

        for container in rjob.template.spec.template.spec.containers:
            if container.name == constants.NODE:
                return container

    return None


def get_runtime_trainer(
    framework: str,
    replicated_jobs: list[models.JobsetV1alpha2ReplicatedJob],
    ml_policy: models.TrainerV1alpha1MLPolicy,
) -> types.RuntimeTrainer:
    """
    Get the RuntimeTrainer object.
    """

    trainer_container = get_runtime_trainer_container(replicated_jobs)

    if not (trainer_container and trainer_container.image):
        raise Exception(f"Runtime doesn't have trainer container {replicated_jobs}")

    trainer = types.RuntimeTrainer(
        trainer_type=(
            types.TrainerType.BUILTIN_TRAINER
            if framework == types.TORCH_TUNE
            else types.TrainerType.CUSTOM_TRAINER
        ),
        framework=framework,
        image=trainer_container.image,
    )

    # Get the container devices.
    if devices := get_container_devices(trainer_container.resources):
        trainer.device, trainer.device_count = devices

    # Torch and MPI plugins override accelerator count.
    if ml_policy.torch and ml_policy.torch.num_proc_per_node:
        num_proc = ml_policy.torch.num_proc_per_node.actual_instance
        if isinstance(num_proc, int):
            trainer.device_count = str(num_proc)
    elif ml_policy.mpi and ml_policy.mpi.num_proc_per_node:
        trainer.device_count = str(ml_policy.mpi.num_proc_per_node)

    # Multiply accelerator_count by the number of nodes.
    if trainer.device_count.isdigit() and ml_policy.num_nodes:
        trainer.device_count = str(int(trainer.device_count) * ml_policy.num_nodes)

    # Add number of training nodes.
    if ml_policy.num_nodes:
        trainer.num_nodes = ml_policy.num_nodes

    # Set the Trainer entrypoint.
    if framework == types.TORCH_TUNE:
        trainer.set_command(constants.TORCH_TUNE_COMMAND)
    elif ml_policy.torch:
        trainer.set_command(constants.TORCH_COMMAND)
    elif ml_policy.mpi:
        trainer.set_command(constants.MPI_COMMAND)
    else:
        trainer.set_command(constants.DEFAULT_COMMAND)

    return trainer


def get_trainjob_initializer_step(
    pod_name: str,
    pod_spec: models.IoK8sApiCoreV1PodSpec,
    pod_status: Optional[models.IoK8sApiCoreV1PodStatus],
) -> types.Step:
    """
    Get the TrainJob initializer step from the given Pod name, spec, and status.
    """

    container = next(
        c
        for c in pod_spec.containers
        if c.name in {constants.DATASET_INITIALIZER, constants.MODEL_INITIALIZER}
    )

    step = types.Step(
        name=container.name,
        status=pod_status.phase if pod_status else None,
        pod_name=pod_name,
    )

    if devices := get_container_devices(container.resources):
        step.device, step.device_count = devices

    return step


def get_trainjob_node_step(
    pod_name: str,
    pod_spec: models.IoK8sApiCoreV1PodSpec,
    pod_status: Optional[models.IoK8sApiCoreV1PodStatus],
    trainjob_runtime: types.Runtime,
    replicated_job_name: str,
    job_index: int,
) -> types.Step:
    """
    Get the TrainJob trainer node step from the given Pod name, spec, and status.
    """

    container = next(c for c in pod_spec.containers if c.name == constants.NODE)

    step = types.Step(
        name=f"{constants.NODE}-{job_index}",
        status=pod_status.phase if pod_status else None,
        pod_name=pod_name,
    )

    if devices := get_container_devices(container.resources):
        step.device, step.device_count = devices

    # For the MPI use-cases, the launcher container is always node-0
    # Thus, we should increase the index for other nodes.
    if (
        trainjob_runtime.trainer.command[0] == "mpirun"
        and replicated_job_name != constants.LAUNCHER
    ):
        # TODO (andreyvelich): We should also override the device_count
        # based on OMPI_MCA_orte_set_default_slots value. Right now, it is hard to do
        # since we inject this env only to the Launcher Pod.
        step.name = f"{constants.NODE}-{job_index + 1}"

    if container.env:
        for env in container.env:
            if (
                env.value
                and env.value.isdigit()
                and env.name == constants.TORCH_ENV_NUM_PROC_PER_NODE
            ):
                step.device_count = env.value

    return step


def get_resources_per_node(
    resources_per_node: dict,
) -> models.IoK8sApiCoreV1ResourceRequirements:
    """
    Get the Trainer resources for the training node from the given dict.
    """

    # Convert only standard resource keys and aliases to lowercase.
    # Extended resources (e.g., "example.com/Custom-NPU") preserve their original case.
    standard_resources = {constants.CPU_LABEL, "memory", "gpu", "storage", "ephemeral-storage"}
    resource_aliases = {"gpu": "nvidia.com/gpu", "storage": "ephemeral-storage"}

    resources = {}
    for k, v in resources_per_node.items():
        key = k.lower() if k.lower() in standard_resources or k.lower().startswith("mig-") else k
        key = resource_aliases.get(key, key)
        resources[key] = models.IoK8sApimachineryPkgApiResourceQuantity(v)

    # Optional alias for MIG: "mig-<profile>" -> "nvidia.com/mig-<profile>"
    # Example: "mig-1g.5gb" -> "nvidia.com/mig-1g.5gb"
    mig_alias_keys = [k for k in resources if k.startswith("mig-")]
    for k in mig_alias_keys:
        resources[f"{constants.GPU_MIG_PREFIX}{k[len('mig-') :]}"] = resources.pop(k)

    mig_keys = [k for k in resources if k.startswith(constants.GPU_MIG_PREFIX)]
    if len(mig_keys) > 1:
        raise ValueError(f"Multiple MIG resource types are not supported: {mig_keys}")
    if mig_keys and "nvidia.com/gpu" in resources:
        raise ValueError(
            f"GPU (nvidia.com/gpu) and MIG ({mig_keys[0]}) cannot be requested together"
        )

    return models.IoK8sApiCoreV1ResourceRequirements(
        requests=resources,
        limits=resources,
    )


def get_script_for_python_packages(
    packages_to_install: list[str],
    pip_index_urls: list[str],
) -> str:
    """
    Get init script to install Python packages from the given pip index URLs.
    """
    packages_str = " ".join(packages_to_install)

    # first url will be the index-url.
    options = [f"--index-url {pip_index_urls[0]}"]
    options.extend(f"--extra-index-url {extra_index_url}" for extra_index_url in pip_index_urls[1:])

    header_script = textwrap.dedent(
        """
        if ! [ -x "$(command -v pip)" ]; then
            python -m ensurepip || python -m ensurepip --user || apt-get install python-pip
        fi

        """
    )

    script_for_python_packages = (
        header_script
        + "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
        + "--no-warn-script-location {} --user {}".format(
            " ".join(options),
            packages_str,
        )
        + " ||\nPIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
        + "--no-warn-script-location {} {}\n".format(
            " ".join(options),
            packages_str,
        )
    )

    return script_for_python_packages


def get_command_using_train_func(
    runtime: types.Runtime,
    train_func: Callable,
    train_func_parameters: Optional[dict[str, Any]],
    pip_index_urls: list[str],
    packages_to_install: Optional[list[str]],
) -> list[str]:
    """
    Get the Trainer container command from the given training function and parameters.
    """
    # Check if the runtime has a Trainer.
    if not runtime.trainer:
        raise ValueError(f"Runtime must have a trainer: {runtime}")

    # Check if training function is callable.
    if not callable(train_func):
        raise ValueError(
            f"Training function must be callable, got function type: {type(train_func)}"
        )

    # Extract the function implementation.
    func_code = inspect.getsource(train_func)

    # Extract the file name where the function is defined.
    func_file = os.path.basename(inspect.getfile(train_func))

    # Function might be defined in some indented scope (e.g. in another function).
    # We need to dedent the function code.
    func_code = textwrap.dedent(func_code)

    # Wrap function code to execute it from the file. For example:
    # TODO (andreyvelich): Find a better way to run users' scripts.
    # def train(parameters):
    #   print('Start Training...')
    # train({'lr': 0.01})
    if train_func_parameters is None:
        func_call = f"{train_func.__name__}()"
    else:
        # Always unpack kwargs for training function calls.
        func_call = f"{train_func.__name__}(**{train_func_parameters})"

    # Combine everything into the final code string.
    func_code = f"{func_code}\n{func_call}\n"

    is_mpi = runtime.trainer.command[0] == "mpirun"
    # The default file location for OpenMPI is: /home/mpiuser/<FILE_NAME>.py
    if is_mpi:
        func_file = os.path.join(constants.DEFAULT_MPI_USER_HOME, func_file)

    # Install Python packages if that is required.
    install_packages = ""
    if packages_to_install:
        install_packages = get_script_for_python_packages(
            packages_to_install,
            pip_index_urls,
        )

    # Add function code to the Trainer command.
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_packages:
                exec_script = install_packages + exec_script
            command.append(exec_script)
        else:
            command.append(c)

    return command


def get_trainer_cr_from_custom_trainer(
    runtime: types.Runtime,
    trainer: Union[types.CustomTrainer, types.CustomTrainerContainer],
) -> models.TrainerV1alpha1Trainer:
    """
    Get the Trainer CR from the custom trainer.

    Args:
        runtime: The runtime configuration.
        trainer: The custom trainer or container configuration.
    """
    trainer_cr = models.TrainerV1alpha1Trainer()

    # Add number of nodes to the Trainer.
    if trainer.num_nodes:
        trainer_cr.num_nodes = trainer.num_nodes

    # Add resources per node to the Trainer.
    if trainer.resources_per_node:
        trainer_cr.resources_per_node = get_resources_per_node(trainer.resources_per_node)

    if isinstance(trainer, types.CustomTrainer):
        # If CustomTrainer is used, generate command from function.
        trainer_cr.command = get_command_using_train_func(
            runtime,
            trainer.func,
            trainer.func_args,
            trainer.pip_index_urls,
            trainer.packages_to_install,
        )

    # Set the TrainJob trainer image if that is set.
    if trainer.image:
        trainer_cr.image = trainer.image

    # Add environment variables to the Trainer.
    if trainer.env:
        trainer_cr.env = [
            models.IoK8sApiCoreV1EnvVar(name=key, value=value) for key, value in trainer.env.items()
        ]

    return trainer_cr


def get_trainer_cr_from_builtin_trainer(
    runtime: types.Runtime,
    trainer: types.BuiltinTrainer,
    initializer: Optional[types.Initializer] = None,
) -> models.TrainerV1alpha1Trainer:
    """
    Get the Trainer CR from the builtin trainer.
    """
    if not isinstance(trainer.config, types.TorchTuneConfig):
        raise ValueError(f"The BuiltinTrainer config is invalid: {trainer.config}")

    trainer_cr = models.TrainerV1alpha1Trainer()

    # Add number of nodes to the Trainer.
    if trainer.config.num_nodes:
        trainer_cr.num_nodes = trainer.config.num_nodes

    # Add resources per node to the Trainer.
    if trainer.config.resources_per_node:
        trainer_cr.resources_per_node = get_resources_per_node(trainer.config.resources_per_node)

    trainer_cr.command = list(runtime.trainer.command)
    # Parse args in the TorchTuneConfig to the Trainer, preparing for the mutation of
    # the torchtune config in the runtime plugin.
    # Ref:https://github.com/kubeflow/trainer/tree/master/docs/proposals/2401-llm-trainer-v2
    trainer_cr.args = get_args_using_torchtune_config(trainer.config, initializer)

    return trainer_cr


def get_args_using_torchtune_config(
    fine_tuning_config: types.TorchTuneConfig,
    initializer: Optional[types.Initializer] = None,
) -> list[str]:
    """
    Get the Trainer args from the TorchTuneConfig.
    """
    args = []

    # Override the dtype if it is provided.
    if fine_tuning_config.dtype:
        if not isinstance(fine_tuning_config.dtype, types.DataType):
            raise ValueError(f"Invalid dtype: {fine_tuning_config.dtype}.")

        args.append(f"dtype={fine_tuning_config.dtype}")

    # Override the batch size if it is provided.
    if fine_tuning_config.batch_size:
        args.append(f"batch_size={fine_tuning_config.batch_size}")

    # Override the epochs if it is provided.
    if fine_tuning_config.epochs:
        args.append(f"epochs={fine_tuning_config.epochs}")

    # Override the loss if it is provided.
    if fine_tuning_config.loss:
        args.append(f"loss={fine_tuning_config.loss}")

    # Override the data dir or data files if it is provided.
    if isinstance(initializer, types.Initializer) and isinstance(
        initializer.dataset, types.HuggingFaceDatasetInitializer
    ):
        storage_uri = (
            "hf://" + initializer.dataset.storage_uri
            if not initializer.dataset.storage_uri.startswith("hf://")
            else initializer.dataset.storage_uri
        )
        storage_uri_parsed = urlparse(storage_uri)
        parts = storage_uri_parsed.path.strip("/").split("/")
        relative_path = "/".join(parts[1:]) if len(parts) > 1 else "."

        if relative_path != "." and "." in relative_path:
            args.append(f"dataset.data_files={os.path.join(constants.DATASET_PATH, relative_path)}")
        else:
            args.append(f"dataset.data_dir={os.path.join(constants.DATASET_PATH, relative_path)}")

    if fine_tuning_config.peft_config:
        args += get_args_from_peft_config(fine_tuning_config.peft_config)

    if fine_tuning_config.dataset_preprocess_config:
        args += get_args_from_dataset_preprocess_config(
            fine_tuning_config.dataset_preprocess_config
        )

    return args


def get_args_from_peft_config(peft_config: types.LoraConfig) -> list[str]:
    """
    Get the args from the given PEFT config.
    """
    args = []

    if not isinstance(peft_config, types.LoraConfig):
        raise ValueError(f"Invalid PEFT config type: {type(peft_config)}.")

    field_map = {
        "apply_lora_to_mlp": "model.apply_lora_to_mlp",
        "apply_lora_to_output": "model.apply_lora_to_output",
        "lora_rank": "model.lora_rank",
        "lora_alpha": "model.lora_alpha",
        "lora_dropout": "model.lora_dropout",
        "quantize_base": "model.quantize_base",
        "use_dora": "model.use_dora",
    }

    # Override the PEFT fields if they are provided.
    for field, arg_name in field_map.items():
        value = getattr(peft_config, field, None)
        if value:
            args.append(f"{arg_name}={value}")

    # Override the LoRA attention modules if they are provided.
    if peft_config.lora_attn_modules:
        args.append(f"model.lora_attn_modules=[{','.join(peft_config.lora_attn_modules)}]")

    return args


def get_args_from_dataset_preprocess_config(
    dataset_preprocess_config: types.TorchTuneInstructDataset,
) -> list[str]:
    """
    Get the args from the given dataset preprocess config.
    """
    args = []

    if not isinstance(dataset_preprocess_config, types.TorchTuneInstructDataset):
        raise ValueError(
            f"Invalid dataset preprocess config type: {type(dataset_preprocess_config)}."
        )

    # Override the dataset type field in the torchtune config.
    args.append(f"dataset={constants.TORCH_TUNE_INSTRUCT_DATASET}")

    # Override the dataset source field if it is provided.
    if dataset_preprocess_config.source:
        if not isinstance(dataset_preprocess_config.source, types.DataFormat):
            raise ValueError(f"Invalid data format: {dataset_preprocess_config.source.value}.")

        args.append(f"dataset.source={dataset_preprocess_config.source.value}")

    # Override the split field if it is provided.
    if dataset_preprocess_config.split:
        args.append(f"dataset.split={dataset_preprocess_config.split}")

    # Override the train_on_input field if it is provided.
    if dataset_preprocess_config.train_on_input:
        args.append(f"dataset.train_on_input={dataset_preprocess_config.train_on_input}")

    # Override the new_system_prompt field if it is provided.
    if dataset_preprocess_config.new_system_prompt:
        args.append(f"dataset.new_system_prompt={dataset_preprocess_config.new_system_prompt}")

    # Override the column_map field if it is provided.
    if dataset_preprocess_config.column_map:
        args.append(f"dataset.column_map={dataset_preprocess_config.column_map}")

    return args


def get_optional_initializer_envs(
    initializer: types.BaseInitializer, required_fields: set
) -> list[models.IoK8sApiCoreV1EnvVar]:
    """Get the optional envs from the initializer config"""
    envs = []
    for f in fields(initializer):
        if f.name not in required_fields:
            value = getattr(initializer, f.name)
            if value is not None:
                # Convert list values (like ignore_patterns) to comma-separated strings
                if isinstance(value, list):
                    value = ",".join(str(item) for item in value)
                envs.append(models.IoK8sApiCoreV1EnvVar(name=f.name.upper(), value=value))
    return envs


def get_dataset_initializer(
    dataset: Union[
        types.HuggingFaceDatasetInitializer,
        types.S3DatasetInitializer,
        types.DataCacheInitializer,
    ],
) -> models.TrainerV1alpha1DatasetInitializer:
    """
    Get the TrainJob dataset initializer from the given config.
    """
    if isinstance(dataset, (types.HuggingFaceDatasetInitializer, types.S3DatasetInitializer)):
        return models.TrainerV1alpha1DatasetInitializer(
            storageUri=dataset.storage_uri,
            env=get_optional_initializer_envs(dataset, required_fields={"storage_uri"}),
        )

    elif isinstance(dataset, types.DataCacheInitializer):
        envs = [
            models.IoK8sApiCoreV1EnvVar(name="CLUSTER_SIZE", value=str(dataset.num_data_nodes + 1)),
            models.IoK8sApiCoreV1EnvVar(name="METADATA_LOC", value=dataset.metadata_loc),
        ]

        # Add env vars from optional fields (skip required fields)
        envs += get_optional_initializer_envs(
            dataset, {"storage_uri", "metadata_loc", "num_data_nodes"}
        )

        return models.TrainerV1alpha1DatasetInitializer(
            storageUri=dataset.storage_uri, env=envs if envs else None
        )

    raise ValueError(f"Dataset initializer type is invalid: {type(dataset)}")


def get_model_initializer(
    model: Union[types.HuggingFaceModelInitializer, types.S3ModelInitializer],
) -> models.TrainerV1alpha1ModelInitializer:
    """
    Get the TrainJob model initializer from the given config.
    """
    if isinstance(model, (types.HuggingFaceModelInitializer, types.S3ModelInitializer)):
        return models.TrainerV1alpha1ModelInitializer(
            storageUri=model.storage_uri,
            env=get_optional_initializer_envs(model, required_fields={"storage_uri"}),
        )

    raise ValueError(f"Model initializer type is invalid: {type(model)}")
