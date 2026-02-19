# Specialized Trainer Abstractions and RuntimeConfig for the Kubeflow SDK

<!--
This proposal targets the kubeflow/sdk repository.
Directory: docs/proposals/285-specialized-trainers/README.md
-->

|                |                                                              |
| -------------- | ------------------------------------------------------------ |
| **Authors**    | @szaher                                                      |
| **Status**     | Draft                                                        |
| **Created**    | 2026-02-11                                                   |
| **Reviewers**  |                                                              |
| **Supersedes** | N/A                                                          |
| **Relevant Issues** | https://github.com/kubeflow/sdk/issues/285              |

## Table of Contents

<!-- toc -->
- [Specialized Trainer Abstractions and RuntimeConfig for the Kubeflow SDK](#specialized-trainer-abstractions-and-runtimeconfig-for-the-kubeflow-sdk)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Motivation](#motivation)
    - [User Value](#user-value)
    - [Personas](#personas)
    - [Goals](#goals)
    - [Non-Goals](#non-goals)
  - [Current State Analysis](#current-state-analysis)
    - [CustomTrainer](#customtrainer)
    - [BuiltinTrainer](#builtintrainer)
    - [TrainerClient.train()](#trainerclienttrain)
    - [Identified Limitations](#identified-limitations)
  - [Proposal](#proposal)
    - [A. BaseTrainer Abstract Interface](#a-basetrainer-abstract-interface)
    - [B. Tier 1: Framework-Specific Trainers](#b-tier-1-framework-specific-trainers)
      - [TorchTrainer](#torchtrainer)
      - [MPITrainer](#mpitrainer)
      - [JAXTrainer](#jaxtrainer)
      - [XGBoostTrainer](#xgboosttrainer)
    - [C. Tier 2: Application-Level Trainers](#c-tier-2-application-level-trainers)
    - [D. RuntimeConfig](#d-runtimeconfig)
    - [E. TrainerClient Changes](#e-trainerclient-changes)
  - [Design Details](#design-details)
    - [Runtime Auto-Discovery](#runtime-auto-discovery)
    - [Runtime Validation](#runtime-validation)
    - [Framework Argument Separation](#framework-argument-separation)
    - [Backend Integration](#backend-integration)
    - [Type Hierarchy Diagram](#type-hierarchy-diagram)
  - [User-Facing API Examples](#user-facing-api-examples)
    - [Before (Current)](#before-current)
    - [After (Proposed)](#after-proposed)
  - [Migration and Backward Compatibility](#migration-and-backward-compatibility)
  - [Test Plan](#test-plan)
    - [Unit Tests](#unit-tests)
    - [Integration Tests](#integration-tests)
    - [Backward Compatibility Tests](#backward-compatibility-tests)
  - [Implementation Plan](#implementation-plan)
  - [Alternatives Considered](#alternatives-considered)
    - [1. Extend CustomTrainer with a `framework` field instead of new classes](#1-extend-customtrainer-with-a-framework-field-instead-of-new-classes)
    - [2. Use Pydantic `BaseModel` instead of `@dataclass`](#2-use-pydantic-basemodel-instead-of-dataclass)
    - [3. Put RuntimeConfig inside BaseTrainer instead of as a separate parameter](#3-put-runtimeconfig-inside-basetrainer-instead-of-as-a-separate-parameter)
    - [4. Automatic runtime selection with scoring/ranking instead of strict single-match](#4-automatic-runtime-selection-with-scoringranking-instead-of-strict-single-match)
    - [5. Have specialized trainers inherit from CustomTrainer](#5-have-specialized-trainers-inherit-from-customtrainer)
  - [References](#references)
<!-- /toc -->

---

## Overview

This proposal introduces two backward-compatible enhancements to the Kubeflow SDK
(`kubeflow/sdk`) trainer subsystem:

1. **Specialized, framework-aware trainer abstractions** — A new `BaseTrainer` abstract
   interface and a suite of framework-specific implementations (`TorchTrainer`, `MPITrainer`,
   `JAXTrainer`, etc.) that automatically discover and validate the correct
   `ClusterTrainingRuntime` using the `trainer.kubeflow.org/framework` label. This fills the
   "missing middle" between the overly generic `CustomTrainer` and the overly narrow
   `BuiltinTrainer`.

2. **`RuntimeConfig` dataclass** — A dedicated configuration object that cleanly separates
   per-job runtime environment settings (packages, pip config, environment variables) from
   training logic and scaling parameters. This replaces the current pattern where
   `CustomTrainer` conflates runtime concerns with trainer concerns.

Both changes are purely additive. Existing code using `CustomTrainer`, `BuiltinTrainer`, and
`TrainerClient.train()` remains fully functional without modification.

---

## Motivation

### User Value

The Kubeflow Trainer v2 architecture (KEP-2170) introduced a powerful separation between
the *what* (`TrainJob`) and the *how* (`TrainingRuntime` / `ClusterTrainingRuntime`). The
SDK exposes this through `TrainerClient.train()`, which accepts a trainer and an optional
runtime reference. However, the current SDK abstractions create a usability gap:

- **`CustomTrainer`** requires the user to know the runtime name, manually look it up
  via `get_runtime()`, and pass both training arguments and runtime-environment settings
  (packages, pip URLs, env vars) into a single flat dataclass. It provides no
  framework-specific validation or argument handling.

- **`BuiltinTrainer`** is restricted to a single use case (`TorchTuneConfig`) and does
  not accept user-defined training functions.

For the majority of distributed training workloads — "run this PyTorch DDP function on
N nodes" or "run this MPI script across a cluster" — neither abstraction fits well.
Users must either use the low-level `CustomTrainer` with manual runtime wiring, or
fall back to raw YAML.

### Personas

This proposal benefits all three personas defined in KEP-2170:

| Persona | Current Pain | Proposed Improvement |
|---|---|---|
| **Data Scientist / ML Engineer** | Must understand runtime names and Kubernetes concepts to use `CustomTrainer` | Uses `TorchTrainer(func=my_fn)` — runtime is auto-discovered |
| **MLOps Engineer** | Must help data scientists find the correct runtime name for their framework | Framework validation catches mismatches at submission time |
| **Platform Admin / DevOps** | Cannot enforce that users pick the correct runtime for their framework | Trainers validate `trainer.kubeflow.org/framework` labels on runtimes |

### Goals

1. Define a `BaseTrainer` abstract interface that all trainer implementations satisfy,
   enabling the SDK and backends to handle any trainer polymorphically.
2. Implement Tier 1 framework-specific trainers (`TorchTrainer`, `MPITrainer`,
   `JAXTrainer`, `XGBoostTrainer`) that auto-discover runtimes by the
   `trainer.kubeflow.org/framework` label and validate runtime compatibility.
3. Provide a clear extension point (Tier 2) for community-contributed, application-level
   trainers (e.g., `TransformersTrainer`, `DeepSpeedTrainer`).
4. Introduce a `RuntimeConfig` dataclass to cleanly separate per-job runtime environment
   settings from training-loop and scaling configuration.
5. Maintain 100% backward compatibility with the existing `CustomTrainer`,
   `CustomTrainerContainer`, `BuiltinTrainer`, and `TrainerClient.train()` APIs.

### Non-Goals

1. **Controller/CRD changes.** This proposal is SDK-only. No changes to the Kubeflow
   Trainer controller, `TrainJob` CRD, or `ClusterTrainingRuntime` CRD are required.
2. **New runtime labels or conventions.** We rely on the existing
   `trainer.kubeflow.org/framework` label already required on all runtimes.
3. **Deprecating `CustomTrainer` or `BuiltinTrainer`.** Both remain supported.
   Specialized trainers are an additional option, not a replacement.
4. **Tier 2 trainer implementations.** This proposal defines the extension mechanism
   and interface. Concrete Tier 2 implementations (HuggingFace, DeepSpeed, Unsloth,
   Axolotl) will be proposed in follow-up KEPs.
5. **Changes to the `TrainJobTemplate` dataclass.** Template support for specialized
   trainers can be added incrementally.

---

## Current State Analysis

The following is the current SDK API surface as of `kubeflow-sdk v0.1` (source:
[`kubeflow/trainer/types/types.py`](https://github.com/kubeflow/sdk/blob/main/kubeflow/trainer/types/types.py)).

### CustomTrainer

```python
@dataclass
class CustomTrainer:
    func: Callable
    func_args: Optional[dict] = None
    image: Optional[str] = None
    packages_to_install: Optional[list[str]] = None          # Runtime concern
    pip_index_urls: list[str] = field(                       # Runtime concern
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    num_nodes: Optional[int] = None                          # Scaling concern
    resources_per_node: Optional[dict] = None                # Scaling concern
    env: Optional[dict[str, str]] = None                     # Runtime concern
```

**Issues:**

- Mixes runtime-environment settings (`packages_to_install`, `pip_index_urls`, `env`)
  with scaling/resource settings (`num_nodes`, `resources_per_node`) and training logic
  (`func`, `func_args`).
- No framework awareness. A user can pass a PyTorch training function with an MPI
  runtime and the SDK will not catch the mismatch until the controller rejects the job
  or, worse, it fails at execution time.
- `func_args` is an untyped `dict` that conflates user hyperparameters with framework
  arguments (e.g., `rdzv_endpoint`, `nnodes`) that the Trainer controller already
  injects via environment variables.

### BuiltinTrainer

```python
@dataclass
class BuiltinTrainer:
    config: TorchTuneConfig
```

- Hardcoded to `TorchTuneConfig`. Cannot be extended to other config-driven frameworks
  without modifying the class itself.

### TrainerClient.train()

```python
def train(
    self,
    runtime: Optional[Union[str, types.Runtime]] = None,
    initializer: Optional[types.Initializer] = None,
    trainer: Optional[
        Union[types.CustomTrainer, types.CustomTrainerContainer, types.BuiltinTrainer]
    ] = None,
    options: Optional[list] = None,
) -> str:
```

- The `trainer` parameter type union must be extended for each new trainer type.
- No concept of runtime auto-discovery: if `runtime` is `None`, it defaults to
  `torch-distributed` regardless of the trainer type.

### Identified Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | **Missing middle abstraction** | 90% of workloads fall between BuiltinTrainer (too specific) and CustomTrainer (too generic) |
| 2 | **Mixed concerns in CustomTrainer** | Runtime config, scaling config, and training logic are tangled in one dataclass |
| 3 | **No framework validation** | Mismatched trainer/runtime combinations fail late — at execution, not submission |
| 4 | **No framework-specific arguments** | torch-specific args (e.g., `max-restarts`, `monitor-interval`) have no typed home |
| 5 | **BuiltinTrainer is not extensible** | Adding a new config-driven framework requires changing the BuiltinTrainer class |
| 6 | **Flat `func_args` dict** | User hyperparameters mix with framework arguments the controller injects |

---

## Proposal

### A. BaseTrainer Abstract Interface

Introduce an abstract base class that defines the contract all trainers must satisfy.
This enables the SDK, backends, and `TrainerClient` to work with any trainer
polymorphically through a single, stable interface.

```python
# kubeflow/trainer/types/types.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, ClassVar

@dataclass
class BaseTrainer(ABC):
    """Abstract base class for all specialized trainer implementations.

    Class Attributes:
        supported_frameworks: Framework identifiers this trainer supports.
            Must match values of the `trainer.kubeflow.org/framework` label
            on ClusterTrainingRuntime resources.
    """

    supported_frameworks: ClassVar[list[str]]

    @abstractmethod
    def get_train_func(self) -> Optional[Callable]:
        """Return the user-provided training function, or None for
        container/config-driven trainers."""
        ...

    @abstractmethod
    def get_train_func_args(self) -> Optional[dict]:
        """Return the arguments to pass to the training function."""
        ...

    @abstractmethod
    def get_framework_args(self) -> dict:
        """Return framework-specific CLI/env arguments that do not overlap
        with arguments injected by the Kubeflow Trainer controller
        (e.g., rdzv_endpoint, nnodes are excluded)."""
        ...

    def get_num_nodes(self) -> Optional[int]:
        """Return the number of nodes for distributed training."""
        return getattr(self, "num_nodes", None)

    def get_resources_per_node(self) -> Optional[dict]:
        """Return resource requirements per node."""
        return getattr(self, "resources_per_node", None)

    def get_image(self) -> Optional[str]:
        """Return a custom container image, if any."""
        return getattr(self, "image", None)

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate that the given runtime is compatible with this trainer.

        The default implementation checks the runtime's framework label against
        `supported_frameworks`. Subclasses may add additional validation.

        Raises:
            ValueError: If the runtime's framework is not in supported_frameworks.
        """
        # Runtime.trainer.framework holds the value of
        # the trainer.kubeflow.org/framework label.
        if runtime.trainer.framework not in self.supported_frameworks:
            raise ValueError(
                f"{type(self).__name__} supports frameworks "
                f"{self.supported_frameworks}, but runtime '{runtime.name}' "
                f"has framework '{runtime.trainer.framework}'"
            )
```

**Design decisions:**

- `supported_frameworks` is a `ClassVar`, not an instance field. It is a property of
  the trainer *class*, not of individual instances.
- `validate_runtime()` has a default implementation so subclasses get validation for
  free but can extend it.
- The interface uses simple return types (`Optional[Callable]`, `dict`, `Optional[int]`)
  rather than framework-specific types, keeping the base class framework-agnostic.
- Methods use `get_*` naming to clearly indicate they are accessors, not setters.

### B. Tier 1: Framework-Specific Trainers

Each Tier 1 trainer maps 1:1 to a framework identified by the
`trainer.kubeflow.org/framework` label value.

#### TorchTrainer

```python
@dataclass
class TorchTrainer(BaseTrainer):
    """Trainer for PyTorch distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: torch`.

    Args:
        func: The training function. Each node executes this function within
            the distributed environment configured by the runtime.
        func_args: Arguments passed to the training function. Should contain
            only user hyperparameters — framework arguments like rdzv_endpoint
            and nnodes are injected by the Kubeflow Trainer controller.
        num_nodes: Number of nodes for distributed training.
        resources_per_node: Resource requirements per node (cpu, memory, gpu).
        image: Optional custom container image.
        max_restarts: Maximum number of worker group restarts before failing.
            Maps to torchrun --max-restarts.
        monitor_interval: Interval in seconds for the elastic agent to monitor
            workers. Maps to torchrun --monitor-interval.
    """

    supported_frameworks: ClassVar[list[str]] = ["torch"]

    func: Callable
    func_args: Optional[dict] = None
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    image: Optional[str] = None

    # Torch-specific arguments (non-overlapping with controller-injected args)
    max_restarts: Optional[int] = None
    monitor_interval: Optional[float] = None

    def get_train_func(self) -> Optional[Callable]:
        return self.func

    def get_train_func_args(self) -> Optional[dict]:
        return self.func_args

    def get_framework_args(self) -> dict:
        args = {}
        if self.max_restarts is not None:
            args["max-restarts"] = str(self.max_restarts)
        if self.monitor_interval is not None:
            args["monitor-interval"] = str(self.monitor_interval)
        return args
```

#### MPITrainer

```python
@dataclass
class MPITrainer(BaseTrainer):
    """Trainer for MPI-based distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: mpi`.

    Args:
        func: The training function.
        func_args: Arguments passed to the training function.
        num_nodes: Number of nodes for distributed training.
        resources_per_node: Resource requirements per node.
        image: Optional custom container image.
        num_proc_per_node: Number of MPI processes per node.
    """

    supported_frameworks: ClassVar[list[str]] = ["mpi"]

    func: Callable
    func_args: Optional[dict] = None
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    image: Optional[str] = None

    # MPI-specific arguments
    num_proc_per_node: Optional[int] = None

    def get_train_func(self) -> Optional[Callable]:
        return self.func

    def get_train_func_args(self) -> Optional[dict]:
        return self.func_args

    def get_framework_args(self) -> dict:
        args = {}
        if self.num_proc_per_node is not None:
            args["num-proc-per-node"] = str(self.num_proc_per_node)
        return args
```

#### JAXTrainer

```python
@dataclass
class JAXTrainer(BaseTrainer):
    """Trainer for JAX distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: jax`.

    Args:
        func: The training function.
        func_args: Arguments passed to the training function.
        num_nodes: Number of nodes for distributed training.
        resources_per_node: Resource requirements per node.
        image: Optional custom container image.
    """

    supported_frameworks: ClassVar[list[str]] = ["jax"]

    func: Callable
    func_args: Optional[dict] = None
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    image: Optional[str] = None

    def get_train_func(self) -> Optional[Callable]:
        return self.func

    def get_train_func_args(self) -> Optional[dict]:
        return self.func_args

    def get_framework_args(self) -> dict:
        return {}
```

#### XGBoostTrainer

```python
@dataclass
class XGBoostTrainer(BaseTrainer):
    """Trainer for XGBoost distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: xgboost`.
    """

    supported_frameworks: ClassVar[list[str]] = ["xgboost"]

    func: Callable
    func_args: Optional[dict] = None
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    image: Optional[str] = None

    def get_train_func(self) -> Optional[Callable]:
        return self.func

    def get_train_func_args(self) -> Optional[dict]:
        return self.func_args

    def get_framework_args(self) -> dict:
        return {}
```

### C. Tier 2: Application-Level Trainers

Tier 2 trainers compose Tier 1 trainers or extend `BaseTrainer` directly to provide
higher-level, application-specific APIs. This proposal defines the extension point;
concrete implementations are deferred to follow-up proposals.

```python
# Example: future HuggingFaceTrainer (NOT part of this proposal's implementation scope)

@dataclass
class TransformersTrainer(BaseTrainer):
    """Trainer for HuggingFace Transformers training.

    Wraps HuggingFace's Trainer API and maps to a PyTorch runtime.
    """

    supported_frameworks: ClassVar[list[str]] = ["torch"]

    model_name: str
    training_args: dict
    dataset: str
    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None

    def get_train_func(self) -> Optional[Callable]:
        # Returns a generated function that uses HF Trainer internally
        ...

    def get_train_func_args(self) -> Optional[dict]:
        return {"model_name": self.model_name, "dataset": self.dataset, **self.training_args}

    def get_framework_args(self) -> dict:
        return {}
```

The existing `BuiltinTrainer` with `TorchTuneConfig` is conceptually a Tier 2 trainer.
In a future proposal, it could be refactored to extend `BaseTrainer`. For now, it
remains unchanged for backward compatibility.

### D. RuntimeConfig

Extract runtime-environment settings from `CustomTrainer` into a dedicated dataclass.
This provides a clean separation of concerns and allows runtime configuration to be
reused across any trainer type.

```python
@dataclass
class PipConfig:
    """Configuration for pip package installation.

    Args:
        index_urls: PyPI index URLs. The first URL is the primary index;
            remaining URLs are extra indexes.
        quiet: Suppress pip output during installation.
        install_as_user: Use --user flag for pip install (useful when the
            container runs as non-root).
    """

    index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    quiet: bool = True
    install_as_user: bool = False


@dataclass
class RuntimeConfig:
    """Per-job runtime environment configuration.

    Separates runtime-environment concerns (what packages to install, what
    environment variables to set) from training-loop and scaling concerns.

    This is passed to `TrainerClient.train()` and applies regardless of
    the trainer type used.

    Args:
        packages: Python packages to install before running the training
            function (e.g., ["transformers>=4.40", "datasets"]).
        pip_config: Configuration for pip installation behavior.
        env: Environment variables to set in all training nodes.
    """

    packages: Optional[list[str]] = None
    pip_config: Optional[PipConfig] = None
    env: Optional[dict[str, str]] = None
```

**Design decisions:**

- Uses `@dataclass` (not Pydantic `BaseModel`) to be consistent with the rest of the
  SDK codebase.
- `PipConfig` is a separate dataclass rather than inline fields, because pip
  configuration is a distinct concern with its own options.
- `packages` replaces `packages_to_install` for brevity.
- `RuntimeConfig` is optional — when not provided, the trainer's own fields
  (`packages_to_install`, `env` on `CustomTrainer`) or the runtime defaults are used.
  This preserves backward compatibility.

### E. TrainerClient Changes

The `TrainerClient.train()` method signature is extended to accept the new types:

```python
class TrainerClient:

    def train(
        self,
        runtime: Optional[Union[str, "Runtime"]] = None,
        initializer: Optional["Initializer"] = None,
        trainer: Optional[
            Union[
                "CustomTrainer",
                "CustomTrainerContainer",
                "BuiltinTrainer",
                "BaseTrainer",        # NEW: accepts any specialized trainer
            ]
        ] = None,
        runtime_config: Optional["RuntimeConfig"] = None,  # NEW
        options: Optional[list] = None,
    ) -> str:
```

When a `BaseTrainer` subclass is passed:

1. If `runtime` is `None`, the SDK calls `list_runtimes()` and filters by the
   `trainer.kubeflow.org/framework` label matching the trainer's
   `supported_frameworks`.
2. If exactly one matching runtime is found, it is used automatically.
3. If multiple matching runtimes are found, a `ValueError` is raised listing the
   available options and instructing the user to specify one explicitly.
4. If `runtime` is provided (as a name or `Runtime` object), the trainer's
   `validate_runtime()` method is called to verify compatibility.
5. The backend receives the trainer through the existing `BaseTrainer` interface
   methods (`get_train_func()`, `get_framework_args()`, etc.).

When `runtime_config` is provided, its values take precedence over any
runtime-environment fields on `CustomTrainer` (for backward compatibility, those
fields remain on `CustomTrainer` but `RuntimeConfig` is the preferred mechanism).

---

## Design Details

### Runtime Auto-Discovery

The auto-discovery logic lives in the `TrainerClient` (not in the backend), ensuring
consistent behavior across all backends:

```python
def _resolve_runtime(
    self,
    trainer: BaseTrainer,
    runtime: Optional[Union[str, Runtime]],
) -> Runtime:
    """Resolve the runtime for a specialized trainer.

    If runtime is provided, validate it. If not, auto-discover by framework label.
    """
    if runtime is not None:
        # Explicit runtime — validate compatibility
        if isinstance(runtime, str):
            runtime = self.get_runtime(runtime)
        trainer.validate_runtime(runtime)
        return runtime

    # Auto-discover: find runtimes matching the trainer's frameworks
    all_runtimes = self.list_runtimes()
    matching = [
        r for r in all_runtimes
        if r.trainer.framework in trainer.supported_frameworks
    ]

    if len(matching) == 0:
        raise ValueError(
            f"No runtime found for frameworks {trainer.supported_frameworks}. "
            f"Available runtimes: {[r.name for r in all_runtimes]}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"Multiple runtimes found for frameworks "
            f"{trainer.supported_frameworks}: {[r.name for r in matching]}. "
            f"Please specify the runtime explicitly."
        )

    return matching[0]
```

### Runtime Validation

Validation happens at two levels:

1. **Framework label check** (in `BaseTrainer.validate_runtime()`): Ensures the
   runtime's `trainer.kubeflow.org/framework` label value is in the trainer's
   `supported_frameworks` list.

2. **Framework-specific checks** (in subclass overrides): For example, `MPITrainer`
   could verify that the runtime's MPI policy source is configured correctly.

Validation errors are raised as `ValueError` at submission time, *before* the
`TrainJob` CR is created in the cluster.

### Framework Argument Separation

The current `CustomTrainer.func_args` dict mixes user hyperparameters with framework
arguments. Specialized trainers solve this by separating them into two methods:

| Method | Contains | Example |
|---|---|---|
| `get_train_func_args()` | User hyperparameters | `{"learning_rate": 1e-4, "epochs": 10}` |
| `get_framework_args()` | Framework-specific CLI args not injected by the controller | `{"max-restarts": "3", "monitor-interval": "5"}` |

Arguments that the Kubeflow Trainer controller already injects (e.g., `rdzv_endpoint`,
`nnodes`, `nproc_per_node`, `node_rank`) are **excluded** from `get_framework_args()`.
The specialized trainer documentation explicitly lists which arguments it manages vs.
which the controller manages.

### Backend Integration

Each backend (`KubernetesBackend`, `ContainerBackend`, `LocalProcessBackend`) must be
updated to handle `BaseTrainer` instances. The integration follows this pattern:

```python
# In each backend's train() method:

def train(self, runtime, initializer, trainer, runtime_config, options):
    if isinstance(trainer, BaseTrainer):
        # Use the BaseTrainer interface
        func = trainer.get_train_func()
        func_args = trainer.get_train_func_args()
        framework_args = trainer.get_framework_args()
        num_nodes = trainer.get_num_nodes()
        resources = trainer.get_resources_per_node()
        image = trainer.get_image()
        # Build TrainJob spec using these values + framework_args
        ...
    elif isinstance(trainer, CustomTrainer):
        # Existing logic, unchanged
        ...
```

The `runtime_config` parameter is applied uniformly: packages are installed in the
init container, environment variables are set on all training pods.

### Type Hierarchy Diagram

```
                    BaseTrainer (ABC)
                    ├── get_train_func()
                    ├── get_train_func_args()
                    ├── get_framework_args()
                    ├── get_num_nodes()
                    ├── get_resources_per_node()
                    ├── get_image()
                    └── validate_runtime()
                         │
          ┌──────────────┼──────────────┬───────────────┐
          │              │              │               │
    TorchTrainer    MPITrainer    JAXTrainer    XGBoostTrainer
    (framework:     (framework:   (framework:   (framework:
     "torch")        "mpi")        "jax")        "xgboost")
          │
          │  (future Tier 2, via follow-up proposals)
          │
    ┌─────┴──────────┐
    │                │
HuggingFace     DeepSpeed
  Trainer        Trainer


    Existing (unchanged):

    CustomTrainer          BuiltinTrainer         CustomTrainerContainer
    (flat dataclass,       (TorchTuneConfig,      (image-based,
     no base class)         no base class)          no base class)


    New:

    RuntimeConfig ─── PipConfig
    (per-job env)      (pip settings)
```

---

## User-Facing API Examples

### Before (Current)

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

# User must know the runtime name
client = TrainerClient()

# Must manually look up runtime
runtime = client.get_runtime("torch-distributed")

# Runtime config mixed into trainer
job_id = client.train(
    runtime=runtime,
    trainer=CustomTrainer(
        func=train_pytorch,
        func_args={"lr": 1e-4, "epochs": 10},
        packages_to_install=["transformers", "datasets"],
        pip_index_urls=["https://pypi.org/simple"],
        env={"NCCL_DEBUG": "INFO"},
        num_nodes=4,
        resources_per_node={"gpu": 1, "cpu": 3, "memory": "16Gi"},
    ),
)
```

### After (Proposed)

```python
from kubeflow.trainer import TrainerClient, TorchTrainer, RuntimeConfig

client = TrainerClient()

# Runtime is auto-discovered from trainer.kubeflow.org/framework: torch
# Runtime environment is cleanly separated
job_id = client.train(
    trainer=TorchTrainer(
        func=train_pytorch,
        func_args={"lr": 1e-4, "epochs": 10},
        num_nodes=4,
        resources_per_node={"gpu": 1, "cpu": 3, "memory": "16Gi"},
        max_restarts=3,  # Typed, torch-specific argument
    ),
    runtime_config=RuntimeConfig(
        packages=["transformers", "datasets"],
        env={"NCCL_DEBUG": "INFO"},
    ),
)
```

**Explicit runtime selection (when multiple runtimes exist for a framework):**

```python
job_id = client.train(
    runtime="torch-elastic",  # Explicit selection
    trainer=TorchTrainer(
        func=train_pytorch,
        func_args={"lr": 1e-4},
        num_nodes=4,
        resources_per_node={"gpu": 2},
    ),
)
```

**MPI example:**

```python
from kubeflow.trainer import MPITrainer, RuntimeConfig

job_id = client.train(
    trainer=MPITrainer(
        func=train_horovod,
        num_nodes=8,
        resources_per_node={"gpu": 4, "memory": "32Gi"},
        num_proc_per_node=4,
    ),
    runtime_config=RuntimeConfig(
        packages=["horovod[pytorch]"],
    ),
)
```

---

## Migration and Backward Compatibility

| Aspect | Impact |
|---|---|
| `CustomTrainer` | **No change.** Remains fully functional. `packages_to_install`, `pip_index_urls`, and `env` fields are retained. |
| `CustomTrainerContainer` | **No change.** |
| `BuiltinTrainer` | **No change.** |
| `TrainerClient.train()` | **Additive only.** New `runtime_config` parameter is optional with default `None`. The `trainer` parameter type union is extended to include `BaseTrainer`. |
| `TrainJobTemplate` | **No change in this proposal.** Future work can extend it to support `BaseTrainer` subclasses. |
| `RuntimeConfig` vs `CustomTrainer` fields | When both `RuntimeConfig` and `CustomTrainer` fields are provided, `RuntimeConfig` takes precedence. This is documented but does not break existing code since `RuntimeConfig` defaults to `None`. |
| Python version | No new Python version requirements. Uses `dataclass`, `ABC`, `ClassVar` — all available in Python 3.9+. |
| SDK public exports | New classes are exported from `kubeflow.trainer` (`TorchTrainer`, `MPITrainer`, `JAXTrainer`, `XGBoostTrainer`, `RuntimeConfig`, `PipConfig`). No existing exports are removed or renamed. |

---

## Test Plan

### Unit Tests

1. **BaseTrainer interface compliance**: Verify that each Tier 1 trainer correctly
   implements all abstract methods.
2. **`validate_runtime()` — positive**: Each trainer validates a runtime with a
   matching framework label.
3. **`validate_runtime()` — negative**: Each trainer raises `ValueError` for a
   runtime with a non-matching framework label.
4. **`get_framework_args()`**: Verify that each trainer returns only non-overlapping
   arguments (excludes controller-injected args).
5. **`RuntimeConfig` defaults**: Verify `None` defaults and precedence over
   `CustomTrainer` fields.
6. **Runtime auto-discovery — single match**: Mock `list_runtimes()` to return one
   matching runtime; verify it is selected.
7. **Runtime auto-discovery — no match**: Mock `list_runtimes()` to return no
   matching runtimes; verify `ValueError`.
8. **Runtime auto-discovery — multiple matches**: Mock `list_runtimes()` to return
   multiple matching runtimes; verify `ValueError` with runtime names in the
   message.

### Integration Tests

1. **End-to-end with `KubernetesBackend`**: Submit a `TorchTrainer` job against a
   cluster with the `torch-distributed` runtime installed; verify the `TrainJob` CR
   is created with the correct runtime reference.
2. **End-to-end with `ContainerBackend`**: Submit a `TorchTrainer` job locally;
   verify the container is launched with the correct entrypoint and arguments.
3. **`RuntimeConfig` application**: Verify that packages from `RuntimeConfig` are
   installed in the training container and env vars are set.

### Backward Compatibility Tests

1. All existing `CustomTrainer` tests pass without modification.
2. All existing `BuiltinTrainer` tests pass without modification.
3. Existing `TrainJobTemplate` usage continues to work.

---

## Implementation Plan

This proposal can be implemented incrementally across multiple PRs:

**Phase 1: Core Interface and RuntimeConfig**
- Add `BaseTrainer` abstract class to `kubeflow/trainer/types/types.py`
- Add `RuntimeConfig` and `PipConfig` dataclasses
- Add `_resolve_runtime()` to `TrainerClient`
- Extend `TrainerClient.train()` signature
- Unit tests for the interface and RuntimeConfig

**Phase 2: TorchTrainer**
- Implement `TorchTrainer`
- Update `KubernetesBackend`, `ContainerBackend`, `LocalProcessBackend` to handle
  `BaseTrainer`
- Integration tests
- Documentation and examples

**Phase 3: MPITrainer, JAXTrainer, XGBoostTrainer**
- Implement remaining Tier 1 trainers
- Framework-specific validation and argument handling
- Tests and documentation

**Phase 4: Public API exports and documentation**
- Export new classes from `kubeflow.trainer.__init__`
- Update SDK documentation on sdk.kubeflow.org
- Add migration guide examples

---

## Alternatives Considered

### 1. Extend CustomTrainer with a `framework` field instead of new classes

Add a `framework: Optional[str]` field to `CustomTrainer` and use it for runtime
discovery and validation.

**Rejected because:**
- Does not provide a place for framework-specific typed arguments (`max_restarts`,
  `num_proc_per_node`).
- Does not enable the Tier 2 extension model.
- Violates the open-closed principle: the `CustomTrainer` class would need to grow
  with each new framework.

### 2. Use Pydantic `BaseModel` instead of `@dataclass`

Use Pydantic for automatic validation, serialization, and schema generation.

**Rejected because:**
- The existing SDK codebase uses `@dataclass` exclusively. Introducing Pydantic
  would add a dependency and create an inconsistency in the codebase.
- Pydantic validation can be replicated with `__post_init__` where needed.

### 3. Put RuntimeConfig inside BaseTrainer instead of as a separate parameter

Make `RuntimeConfig` a field on `BaseTrainer` so that each trainer carries its own
runtime config.

**Rejected because:**
- Runtime configuration (packages, env vars) is orthogonal to trainer type. The
  same `RuntimeConfig` should be usable with `CustomTrainer`,
  `CustomTrainerContainer`, or any `BaseTrainer` subclass.
- Keeping it as a separate `train()` parameter maintains clean separation of concerns.

### 4. Automatic runtime selection with scoring/ranking instead of strict single-match

When multiple runtimes match a framework, automatically pick the "best" one using a
scoring heuristic (e.g., prefer non-deprecated, prefer more specific labels).

**Rejected because:**
- Implicit selection heuristics are fragile and hard to debug. When multiple runtimes
  exist for the same framework, it is a deliberate platform configuration and the
  user should explicitly choose.
- A clear error message listing available runtimes is more useful than a possibly
  wrong automatic selection.

### 5. Have specialized trainers inherit from CustomTrainer

Make `TorchTrainer` a subclass of `CustomTrainer` instead of a new `BaseTrainer`
hierarchy.

**Rejected because:**
- `CustomTrainer` carries runtime-environment fields (`packages_to_install`,
  `pip_index_urls`, `env`) that specialized trainers should not expose (those belong
  in `RuntimeConfig`).
- Inheriting from `CustomTrainer` would force specialized trainers to carry fields
  that violate the separation of concerns this proposal aims to achieve.

---

## References

- [KEP-2170: Kubeflow Trainer V2 API](https://github.com/kubeflow/trainer/blob/master/docs/proposals/2170-kubeflow-trainer-v2/README.md)
- [Kubeflow SDK Repository](https://github.com/kubeflow/sdk)
- [Kubeflow Trainer Repository](https://github.com/kubeflow/trainer)
- [Kubeflow Community Proposal Workflow](https://github.com/kubeflow/community/blob/master/proposal-workflow.md)
- [Runtime Guide — trainer.kubeflow.org/framework label](https://www.kubeflow.org/docs/components/trainer/operator-guides/runtime/)
- [Kubeflow Trainer Getting Started](https://www.kubeflow.org/docs/components/trainer/getting-started/)
- [SDK Types Source Code](https://github.com/kubeflow/sdk/blob/main/kubeflow/trainer/types/types.py)
- [SDK TrainerClient Source Code](https://github.com/kubeflow/sdk/blob/main/kubeflow/trainer/api/trainer_client.py)
