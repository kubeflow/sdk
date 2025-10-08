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

"""Kubernetes-specific training options for the Kubeflow Trainer SDK."""

from dataclasses import dataclass
from typing import Any, Optional, Union

from kubeflow.trainer.backends.base import RuntimeBackend
from kubeflow.trainer.types.types import BuiltinTrainer, CustomTrainer, CustomTrainerContainer


@dataclass
class ContainerOverride:
    """Configuration for overriding a specific container in a pod.

    Args:
        name: Name of the container to override (must exist in TrainingRuntime).
        env: Environment variables to add/merge with the container.
             Each dict should have 'name' and 'value' or 'valueFrom' keys.
        volume_mounts: Volume mounts to add/merge with the container.
                      Each dict should have 'name' and 'mountPath' keys at minimum.
    """

    name: str
    env: Optional[list[dict]] = None
    volume_mounts: Optional[list[dict]] = None

    def __post_init__(self):
        """Validate the container override configuration."""
        # Validate container name
        if not self.name or not self.name.strip():
            raise ValueError("Container name must be a non-empty string")

        if self.env is not None:
            if not isinstance(self.env, list):
                raise ValueError("env must be a list of dictionaries")
            for env_var in self.env:
                if not isinstance(env_var, dict):
                    raise ValueError("Each env entry must be a dictionary")
                if "name" not in env_var:
                    raise ValueError("Each env entry must have a 'name' key")
                if not env_var.get("name"):
                    raise ValueError("env 'name' must be a non-empty string")
                if "value" not in env_var and "valueFrom" not in env_var:
                    raise ValueError("Each env entry must have either 'value' or 'valueFrom' key")
                # Validate valueFrom structure if present
                if "valueFrom" in env_var:
                    value_from = env_var["valueFrom"]
                    if not isinstance(value_from, dict):
                        raise ValueError("env 'valueFrom' must be a dictionary")
                    # valueFrom must have one of these keys
                    valid_keys = {"configMapKeyRef", "secretKeyRef", "fieldRef", "resourceFieldRef"}
                    if not any(key in value_from for key in valid_keys):
                        raise ValueError(
                            f"env 'valueFrom' must contain one of: {', '.join(valid_keys)}"
                        )

        if self.volume_mounts is not None:
            if not isinstance(self.volume_mounts, list):
                raise ValueError("volume_mounts must be a list of dictionaries")
            for mount in self.volume_mounts:
                if not isinstance(mount, dict):
                    raise ValueError("Each volume_mounts entry must be a dictionary")
                if "name" not in mount:
                    raise ValueError("Each volume_mounts entry must have a 'name' key")
                if not mount.get("name"):
                    raise ValueError("volume_mounts 'name' must be a non-empty string")
                if "mountPath" not in mount:
                    raise ValueError("Each volume_mounts entry must have a 'mountPath' key")
                mount_path = mount.get("mountPath")
                if not mount_path or not isinstance(mount_path, str):
                    raise ValueError("volume_mounts 'mountPath' must be a non-empty string")
                if not mount_path.startswith("/"):
                    raise ValueError(
                        f"volume_mounts 'mountPath' must be an absolute path "
                        f"(start with /): {mount_path}"
                    )


@dataclass
class PodSpecOverride:
    """Configuration for overriding pod template specifications.

    Args:
        service_account_name: Service account to use for the pods.
        node_selector: Node selector to place pods on specific nodes.
        affinity: Affinity rules for pod scheduling.
        tolerations: Tolerations for pod scheduling.
        volumes: Volumes to add/merge with the pod.
        init_containers: Init containers to add/merge with the pod.
        containers: Containers to add/merge with the pod.
        scheduling_gates: Scheduling gates for the pods.
        image_pull_secrets: Image pull secrets for the pods.
    """

    service_account_name: Optional[str] = None
    node_selector: Optional[dict[str, str]] = None
    affinity: Optional[dict] = None
    tolerations: Optional[list[dict]] = None
    volumes: Optional[list[dict]] = None
    init_containers: Optional[list[ContainerOverride]] = None
    containers: Optional[list[ContainerOverride]] = None
    scheduling_gates: Optional[list[dict]] = None
    image_pull_secrets: Optional[list[dict]] = None


@dataclass
class PodTemplateOverride:
    """Configuration for overriding pod templates for specific job types.

    Args:
        target_jobs: List of job names to apply the overrides to (e.g., ["node", "launcher"]).
        metadata: Metadata overrides for the pod template (labels, annotations).
        spec: Spec overrides for the pod template.
    """

    target_jobs: list[str]
    metadata: Optional[dict] = None
    spec: Optional[PodSpecOverride] = None


@dataclass
class Labels:
    """Add labels to the TrainJob resource metadata (.metadata.labels).

    Supported backends:
        - Kubernetes

    Args:
        labels: Dictionary of label key-value pairs to add to TrainJob metadata.
    """

    labels: dict[str, str]

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: Optional[Union[CustomTrainer, BuiltinTrainer]],
        backend: RuntimeBackend,
    ) -> None:
        """Apply labels to the job specification.

        Args:
            job_spec: Job specification dictionary to modify.
            trainer: Optional trainer instance for context.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support labels.
        """
        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Labels option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        metadata = job_spec.setdefault("metadata", {})
        metadata["labels"] = self.labels


@dataclass
class Annotations:
    """Add annotations to the TrainJob resource metadata (.metadata.annotations).

    Supported backends:
        - Kubernetes

    Args:
        annotations: Dictionary of annotation key-value pairs to add to TrainJob metadata.
    """

    annotations: dict[str, str]

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: Optional[Union[CustomTrainer, BuiltinTrainer]],
        backend: RuntimeBackend,
    ) -> None:
        """Apply annotations to the job specification.

        Args:
            job_spec: Job specification dictionary to modify.
            trainer: Optional trainer instance for context.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support annotations.
        """
        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"Annotations option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        metadata = job_spec.setdefault("metadata", {})
        metadata["annotations"] = self.annotations


@dataclass
class SpecLabels:
    """Add labels to derivative JobSet and Jobs (.spec.labels).

    These labels will be merged with the TrainingRuntime values and applied to
    the JobSet and Jobs created by the TrainJob.

    Supported backends:
        - Kubernetes

    Args:
        labels: Dictionary of label key-value pairs to add to JobSet and Jobs.
    """

    labels: dict[str, str]

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: Optional[Union[CustomTrainer, BuiltinTrainer]],
        backend: RuntimeBackend,
    ) -> None:
        """Apply spec-level labels to the job specification.

        Args:
            job_spec: Job specification dictionary to modify.
            trainer: Optional trainer instance for context.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support spec labels.
        """
        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"SpecLabels option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        spec = job_spec.setdefault("spec", {})
        spec["labels"] = self.labels


@dataclass
class SpecAnnotations:
    """Add annotations to derivative JobSet and Jobs (.spec.annotations).

    These annotations will be merged with the TrainingRuntime values and applied to
    the JobSet and Jobs created by the TrainJob.

    Supported backends:
        - Kubernetes

    Args:
        annotations: Dictionary of annotation key-value pairs to add to JobSet and Jobs.
    """

    annotations: dict[str, str]

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: Optional[Union[CustomTrainer, BuiltinTrainer]],
        backend: RuntimeBackend,
    ) -> None:
        """Apply spec-level annotations to the job specification.

        Args:
            job_spec: Job specification dictionary to modify.
            trainer: Optional trainer instance for context.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support spec annotations.
        """
        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"SpecAnnotations option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        spec = job_spec.setdefault("spec", {})
        spec["annotations"] = self.annotations


class PodTemplateOverrides:
    """Add pod template overrides to the TrainJob (.spec.podTemplateOverrides).

    Supported backends:
        - Kubernetes

    Args:
        *overrides: One or more PodTemplateOverride objects.
    """

    def __init__(self, *overrides: PodTemplateOverride):
        """Initialize with variable number of PodTemplateOverride objects."""
        if not overrides:
            raise ValueError("At least one PodTemplateOverride must be provided")
        self.pod_overrides = list(overrides)

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: Optional[Union[CustomTrainer, BuiltinTrainer]],
        backend: RuntimeBackend,
    ) -> None:
        """Apply pod template overrides to the job specification.

        Args:
            job_spec: Job specification dictionary to modify.
            trainer: Optional trainer instance for context.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend does not support pod template overrides.
        """
        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"PodTemplateOverrides option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )
        spec = job_spec.setdefault("spec", {})
        pod_overrides = spec.setdefault("podTemplateOverrides", [])

        for override in self.pod_overrides:
            api_override = {"targetJobs": [{"name": job} for job in override.target_jobs]}

            if override.metadata:
                api_override["metadata"] = override.metadata

            if override.spec:
                spec_dict = {}

                if override.spec.service_account_name:
                    spec_dict["serviceAccountName"] = override.spec.service_account_name
                if override.spec.node_selector:
                    spec_dict["nodeSelector"] = override.spec.node_selector
                if override.spec.affinity:
                    spec_dict["affinity"] = override.spec.affinity
                if override.spec.tolerations:
                    spec_dict["tolerations"] = override.spec.tolerations
                if override.spec.volumes:
                    spec_dict["volumes"] = override.spec.volumes
                if override.spec.scheduling_gates:
                    spec_dict["schedulingGates"] = override.spec.scheduling_gates
                if override.spec.image_pull_secrets:
                    spec_dict["imagePullSecrets"] = override.spec.image_pull_secrets

                # Handle container overrides
                if override.spec.init_containers:
                    spec_dict["initContainers"] = []
                    for container in override.spec.init_containers:
                        container_dict = {"name": container.name}
                        if container.env:
                            container_dict["env"] = container.env
                        if container.volume_mounts:
                            container_dict["volumeMounts"] = container.volume_mounts
                        spec_dict["initContainers"].append(container_dict)

                if override.spec.containers:
                    spec_dict["containers"] = []
                    for container in override.spec.containers:
                        container_dict = {"name": container.name}
                        if container.env:
                            container_dict["env"] = container.env
                        if container.volume_mounts:
                            container_dict["volumeMounts"] = container.volume_mounts
                        spec_dict["containers"].append(container_dict)

                if spec_dict:
                    api_override["spec"] = spec_dict

            pod_overrides.append(api_override)


@dataclass
class TrainerCommand:
    """Override the trainer container command (.spec.trainer.command).

    Can only be used with CustomTrainerContainer. CustomTrainer generates its own
    command from the function, and BuiltinTrainer uses pre-configured commands.

    Supported backends:
        - Kubernetes

    Args:
        command: List of command strings to override the default trainer command.
    """

    command: list[str]

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: Optional[Union[CustomTrainer, BuiltinTrainer, CustomTrainerContainer]],
        backend: RuntimeBackend,
    ) -> None:
        """Apply trainer command override to the job specification.

        Args:
            job_spec: The job specification to modify.
            trainer: Optional trainer context for validation.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend doesn't support or trainer type conflicts.
        """
        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"TrainerCommand option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        if trainer is not None and not isinstance(trainer, CustomTrainerContainer):
            raise ValueError(
                "TrainerCommand can only be used with CustomTrainerContainer. "
                "CustomTrainer generates its own command from the function, and "
                "BuiltinTrainer uses pre-configured commands."
            )

        spec = job_spec.setdefault("spec", {})
        trainer_spec = spec.setdefault("trainer", {})
        trainer_spec["command"] = self.command


@dataclass
class TrainerArgs:
    """Override the trainer container arguments (.spec.trainer.args).

    Can only be used with CustomTrainerContainer. CustomTrainer generates its own
    arguments from the function, and BuiltinTrainer uses pre-configured arguments.

    Supported backends:
        - Kubernetes

    Args:
        args: List of argument strings to override the default trainer arguments.
    """

    args: list[str]

    def __call__(
        self,
        job_spec: dict[str, Any],
        trainer: Optional[Union[CustomTrainer, BuiltinTrainer, CustomTrainerContainer]],
        backend: RuntimeBackend,
    ) -> None:
        """Apply trainer args override to the job specification.

        Args:
            job_spec: The job specification to modify.
            trainer: Optional trainer context for validation.
            backend: Backend instance for validation.

        Raises:
            ValueError: If backend doesn't support or trainer type conflicts.
        """
        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend

        if not isinstance(backend, KubernetesBackend):
            raise ValueError(
                f"TrainerArgs option is not compatible with {type(backend).__name__}. "
                f"Supported backends: KubernetesBackend"
            )

        if trainer is not None and not isinstance(trainer, CustomTrainerContainer):
            raise ValueError(
                "TrainerArgs can only be used with CustomTrainerContainer. "
                "CustomTrainer generates its own arguments from the function, and "
                "BuiltinTrainer uses pre-configured arguments."
            )

        spec = job_spec.setdefault("spec", {})
        trainer_spec = spec.setdefault("trainer", {})
        trainer_spec["args"] = self.args
