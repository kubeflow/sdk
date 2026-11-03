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
    - [Why Specialized Trainers?](#why-specialized-trainers)
    - [Goals](#goals)
    - [Non-Goals](#non-goals)
  - [Current State Analysis](#current-state-analysis)
    - [CustomTrainer](#customtrainer)
    - [BuiltinTrainer](#builtintrainer)
    - [TrainerClient.train()](#trainerclienttrain)
    - [Identified Limitations](#identified-limitations)
  - [Proposal](#proposal)
    - [A. BaseTrainer Abstract Interface](#a-basetrainer-abstract-interface)
    - [B. FuncTrainer — Function-Driven Base](#b-functrainer--function-driven-base)
      - [TorchTrainer](#torchtrainer)
      - [DeepSpeedTrainer](#deepspeedtrainer)
      - [JAXTrainer](#jaxtrainer)
      - [XGBoostTrainer](#xgboosttrainer)
    - [C. ConfigTrainer — Config-Driven Base](#c-configtrainer--config-driven-base)
      - [BuiltinTrainer Migration Path](#builtintrainer-migration-path)
    - [D. RuntimeConfig](#d-runtimeconfig)
    - [E. TrainerClient Changes](#e-trainerclient-changes)
  - [Design Details](#design-details)
    - [Runtime Auto-Discovery](#runtime-auto-discovery)
    - [Runtime Validation](#runtime-validation)
    - [Trainer Responsibility Boundary](#trainer-responsibility-boundary)
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
  - [Graduation Criteria](#graduation-criteria)
    - [Alpha (target: current cycle)](#alpha-target-current-cycle)
    - [Beta](#beta)
    - [GA](#ga)
  - [Open Questions](#open-questions)
  - [Alternatives Considered](#alternatives-considered)
    - [1. Extend CustomTrainer with a `framework` field instead of new classes](#1-extend-customtrainer-with-a-framework-field-instead-of-new-classes)
    - [2. Use Pydantic `BaseModel` instead of `@dataclass`](#2-use-pydantic-basemodel-instead-of-dataclass)
    - [3. Put RuntimeConfig inside BaseTrainer instead of as a separate parameter](#3-put-runtimeconfig-inside-basetrainer-instead-of-as-a-separate-parameter)
    - [4. Automatic runtime selection with scoring/ranking instead of strict single-match](#4-automatic-runtime-selection-with-scoringranking-instead-of-strict-single-match)
    - [5. Flat hierarchy: all trainers inherit directly from BaseTrainer](#5-flat-hierarchy-all-trainers-inherit-directly-from-basetrainer)
    - [6. Have specialized trainers inherit from CustomTrainer](#6-have-specialized-trainers-inherit-from-customtrainer)
  - [References](#references)
<!-- /toc -->

---

## Overview

This proposal introduces two backward-compatible enhancements to the Kubeflow SDK
(`kubeflow/sdk`) trainer subsystem:

1. **Specialized, framework-aware trainer abstractions** — A three-level type hierarchy:
   `BaseTrainer` (common interface) → `FuncTrainer` (function-driven) and `ConfigTrainer`
   (config-driven), with framework-specific implementations (`TorchTrainer`,
   `DeepSpeedTrainer`, `JAXTrainer`, etc.) that automatically discover and validate the
   correct `ClusterTrainingRuntime` using the `trainer.kubeflow.org/framework` label.
   This fills the "missing middle" between the overly generic `CustomTrainer` and the
   overly narrow `BuiltinTrainer`.

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
N nodes" or "run this DeepSpeed training script across a cluster" — neither abstraction
fits well.
Users must either use the low-level `CustomTrainer` with manual runtime wiring, or
fall back to raw YAML.

### Personas

This proposal benefits all three personas defined in KEP-2170:

| Persona | Current Pain | Proposed Improvement |
|---|---|---|
| **Data Scientist / ML Engineer** | Must understand runtime names and Kubernetes concepts to use `CustomTrainer` | Uses `TorchTrainer(func=my_fn)` — runtime is auto-discovered |
| **MLOps Engineer** | Must help data scientists find the correct runtime name for their framework | Framework validation catches mismatches at submission time |
| **Platform Admin / DevOps** | Cannot enforce that users pick the correct runtime for their framework | Trainers validate `trainer.kubeflow.org/framework` labels on runtimes |

### Why Specialized Trainers?

Beyond runtime auto-discovery and framework validation, specialized trainers provide
a set of capabilities that `CustomTrainer` cannot offer:

| Capability | `CustomTrainer` | Specialized Trainer (e.g., `TorchTrainer`) |
|---|---|---|
| **Runtime selection** | User must know and pass the runtime name | Auto-discovered from `trainer.kubeflow.org/framework` label |
| **Framework validation** | None — mismatches fail at execution time | Validated at submission time, before `TrainJob` is created |
| **Typed framework arguments** | Untyped `func_args` dict mixes hyperparams with framework args (`max_restarts`, `deepspeed_config`) | Dedicated typed fields with IDE autocomplete and documentation |
| **Separation of concerns** | Runtime env (`packages_to_install`, `env`), scaling (`num_nodes`), training logic (`func`), and framework args all in one flat dataclass | Training logic in `FuncTrainer`, config in `ConfigTrainer`, runtime env in `RuntimeConfig`, scaling on `BaseTrainer` |
| **IDE/type-checker support** | `func_args: dict` — no autocomplete or type checking | Typed fields — autocomplete, mypy, and docstrings per framework |
| **Extensibility** | Adding framework support requires modifying `CustomTrainer` or creating ad-hoc wrappers | New framework = new subclass of `FuncTrainer` or `ConfigTrainer` |
| **Self-documenting API** | `CustomTrainer(func=..., func_args={"max_restarts": 3})` — unclear what's a hyperparam vs. framework arg | `TorchTrainer(func=..., max_restarts=3)` — intent is clear from the type |

**In summary:** Specialized trainers encode framework knowledge into the type system.
The trainer class itself tells you what framework it targets, what arguments it accepts,
and what runtimes it is compatible with. This shifts errors from runtime to definition
time and makes the SDK self-documenting.

### Goals

1. Define a `BaseTrainer` abstract interface that all trainer implementations satisfy,
   enabling the SDK and backends to handle any trainer polymorphically.
2. Define two intermediate abstract classes — `FuncTrainer` for function-driven
   trainers and `ConfigTrainer` for config-driven trainers — that provide shared
   fields and default implementations for their respective categories.
3. Implement framework-specific function-driven trainers (`TorchTrainer`,
   `DeepSpeedTrainer`, `JAXTrainer`, `XGBoostTrainer`) that auto-discover runtimes
   by the `trainer.kubeflow.org/framework` label and validate runtime compatibility.
4. Provide a clear extension point for community-contributed config-driven trainers
   (e.g., `TorchTuneTrainer`, `UnslothTrainer`, `VeRLTrainer`).
5. Introduce a `RuntimeConfig` dataclass to cleanly separate per-job runtime environment
   settings from training-loop and scaling configuration.
6. Maintain 100% backward compatibility with the existing `CustomTrainer`,
   `CustomTrainerContainer`, `BuiltinTrainer`, and `TrainerClient.train()` APIs.

### Non-Goals

1. **Controller/CRD changes.** This proposal is SDK-only. No changes to the Kubeflow
   Trainer controller, `TrainJob` CRD, or `ClusterTrainingRuntime` CRD are required.
2. **New runtime labels or conventions.** We rely on the existing
   `trainer.kubeflow.org/framework` label already required on all runtimes.
3. **Deprecating `CustomTrainer` or `BuiltinTrainer`.** Both remain supported.
   Specialized trainers are an additional option, not a replacement. `ConfigTrainer`
   is designed as the successor to `BuiltinTrainer` for config-driven trainers,
   but the migration is deferred to a follow-up proposal.
4. **Tier 2 trainer implementations.** This proposal defines the extension mechanism
   and interface. Concrete Tier 2 implementations (TorchTune, Transformers, Unsloth,
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
- No framework awareness. A user can pass a PyTorch training function with a
  DeepSpeed runtime and the SDK will not catch the mismatch until the controller
  rejects the job or, worse, it fails at execution time.
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

`BaseTrainer` holds common fields and methods shared by both function-driven and
config-driven trainers. The training-mode-specific concerns (`func`/`func_args` vs.
`config`) are pushed down to the two intermediate classes described in sections B and C.

```python
# kubeflow/trainer/types/types.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, ClassVar

@dataclass
class BaseTrainer(ABC):
    """Abstract base class for all specialized trainer implementations.

    Provides common fields for scaling and resource configuration, runtime
    auto-discovery via the `supported_frameworks` class variable, and
    framework validation.

    Subclasses should not inherit from this directly — use `FuncTrainer`
    for function-driven trainers or `ConfigTrainer` for config-driven trainers.

    Class Attributes:
        supported_frameworks: Framework identifiers this trainer supports.
            Must match values of the `trainer.kubeflow.org/framework` label
            on ClusterTrainingRuntime resources. Declared as a tuple (immutable)
            and ordered by preference — the first entry is the preferred framework
            for auto-discovery.

    Args:
        num_nodes: Number of nodes for distributed training.
        resources_per_node: Resource requirements per node (cpu, memory, gpu).
        image: Optional custom container image.
    """

    supported_frameworks: ClassVar[tuple[str, ...]]

    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    image: Optional[str] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            if not hasattr(cls, "supported_frameworks") or not cls.supported_frameworks:
                raise TypeError(
                    f"{cls.__name__} must define a non-empty "
                    f"'supported_frameworks' class variable"
                )

    @abstractmethod
    def get_framework_args(self) -> dict:
        """Return framework-specific CLI/env arguments that do not overlap
        with arguments injected by the Kubeflow Trainer controller
        (e.g., rdzv_endpoint, nnodes are excluded)."""
        ...

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate that the given runtime is compatible with this trainer.

        The default implementation checks the runtime's framework label against
        `supported_frameworks`. Subclasses may add additional validation.

        Raises:
            ValueError: If the runtime's framework is not in supported_frameworks.
        """
        if runtime.trainer.framework not in self.supported_frameworks:
            raise ValueError(
                f"{type(self).__name__} supports frameworks "
                f"{self.supported_frameworks}, but runtime '{runtime.name}' "
                f"has framework '{runtime.trainer.framework}'"
            )
```

**Design decisions:**

- `supported_frameworks` is a `ClassVar[tuple[str, ...]]` — immutable and class-level.
  It is a property of the trainer *class*, not of individual instances. The tuple is
  ordered by preference: the first entry is the preferred framework for auto-discovery.
  `__init_subclass__` enforces that every concrete subclass defines a non-empty
  `supported_frameworks`, catching missing declarations at class definition time
  rather than at runtime.
- Common fields (`num_nodes`, `resources_per_node`, `image`) live on `BaseTrainer` so
  every trainer inherits them without repetition.
- `get_framework_args()` is the only abstract method on `BaseTrainer`. Training-mode
  concerns (`get_train_func()`, `get_config()`) are defined on the intermediate classes.
- `validate_runtime()` has a default implementation so subclasses get validation for
  free but can extend it.
- Methods use `get_*` naming to clearly indicate they are accessors, not setters.

### B. FuncTrainer — Function-Driven Base

`FuncTrainer` is the base class for trainers where the user provides a Python training
function. It owns the `func` and `func_args` fields and implements the corresponding
accessors, so concrete framework trainers only need to add framework-specific fields.

```python
@dataclass
class FuncTrainer(BaseTrainer):
    """Base class for function-driven trainers.

    The user provides a training function that is serialized and executed
    within the distributed environment configured by the runtime.

    Args:
        func: The training function. Each node executes this function.
        func_args: Arguments passed to the training function. Should contain
            only user hyperparameters — framework arguments like rdzv_endpoint
            and nnodes are injected by the Kubeflow Trainer controller.
    """

    func: Callable
    func_args: Optional[dict] = None

    def get_train_func(self) -> Callable:
        """Return the user-provided training function."""
        return self.func

    def get_train_func_args(self) -> Optional[dict]:
        """Return the arguments to pass to the training function."""
        return self.func_args

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate framework and trainer_type compatibility.

        FuncTrainer requires runtimes with trainer_type == CUSTOM_TRAINER.
        """
        super().validate_runtime(runtime)
        if runtime.trainer.trainer_type != TrainerType.CUSTOM_TRAINER:
            raise ValueError(
                f"{type(self).__name__} requires a runtime with "
                f"trainer_type={TrainerType.CUSTOM_TRAINER.value}, but "
                f"runtime '{runtime.name}' has "
                f"trainer_type={runtime.trainer.trainer_type.value}"
            )
```

Concrete function-driven trainers extend `FuncTrainer` and only need to define
`supported_frameworks`, framework-specific fields, and `get_framework_args()`:

#### TorchTrainer

```python
@dataclass
class TorchTrainer(FuncTrainer):
    """Trainer for PyTorch distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: torch`.

    Args:
        max_restarts: Maximum number of worker group restarts before failing.
            Maps to torchrun --max-restarts.
        monitor_interval: Interval in seconds for the elastic agent to monitor
            workers. Maps to torchrun --monitor-interval.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torch",)

    # Torch-specific arguments (non-overlapping with controller-injected args)
    max_restarts: Optional[int] = None
    monitor_interval: Optional[float] = None

    def get_framework_args(self) -> dict:
        args = {}
        if self.max_restarts is not None:
            args["max-restarts"] = str(self.max_restarts)
        if self.monitor_interval is not None:
            args["monitor-interval"] = str(self.monitor_interval)
        return args
```

#### DeepSpeedTrainer

```python
@dataclass
class DeepSpeedTrainer(FuncTrainer):
    """Trainer for DeepSpeed distributed training workloads.

    DeepSpeed can be bootstrapped via either `torchrun` or `mpirun`, so this
    trainer supports both torch-based and MPI-based runtimes. The SDK
    auto-discovers a compatible runtime by matching the
    `trainer.kubeflow.org/framework` label against the supported frameworks.
    When both runtime types are available, the user must specify the runtime
    explicitly.

    Args:
        deepspeed_config: Path or dict for the DeepSpeed JSON configuration.
            When provided, the config is passed to the DeepSpeed launcher
            via the --deepspeed_config flag.
        num_proc_per_node: Number of processes per node. Maps to
            --num_gpus (DeepSpeed launcher) or --nproc_per_node (torchrun).
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("deepspeed", "torch")

    # DeepSpeed-specific arguments
    deepspeed_config: Optional[Union[str, dict]] = None
    num_proc_per_node: Optional[int] = None

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate framework compatibility and launcher support.

        In addition to the standard framework label check, DeepSpeedTrainer
        verifies that the runtime's launcher is compatible (torchrun or mpirun).
        """
        super().validate_runtime(runtime)
        # TODO: Check runtime.trainer.command for launcher compatibility
        # once the Runtime type exposes launcher metadata.

    def get_framework_args(self) -> dict:
        args = {}
        if self.deepspeed_config is not None:
            if isinstance(self.deepspeed_config, dict):
                import json
                args["deepspeed_config"] = json.dumps(self.deepspeed_config)
            else:
                args["deepspeed_config"] = self.deepspeed_config
        if self.num_proc_per_node is not None:
            args["num-proc-per-node"] = str(self.num_proc_per_node)
        return args
```

#### JAXTrainer

```python
@dataclass
class JAXTrainer(FuncTrainer):
    """Trainer for JAX distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: jax`.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("jax",)

    def get_framework_args(self) -> dict:
        return {}
```

#### XGBoostTrainer

```python
@dataclass
class XGBoostTrainer(FuncTrainer):
    """Trainer for XGBoost distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: xgboost`.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("xgboost",)

    def get_framework_args(self) -> dict:
        return {}
```

### C. ConfigTrainer — Config-Driven Base

`ConfigTrainer` is the base class for trainers that are driven by a configuration
object rather than a user-provided function. The runtime's entrypoint (e.g.,
`tune run`, `accelerate launch`) handles execution based on the config.

This replaces the current `BuiltinTrainer` pattern with an extensible, `BaseTrainer`-
compatible design that supports runtime auto-discovery and framework validation.

```python
@dataclass
class ConfigTrainer(BaseTrainer):
    """Base class for config-driven trainers.

    Config-driven trainers do not accept a user training function. Instead,
    they accept a configuration object that fully describes the training job.
    The runtime's entrypoint handles execution based on the config.

    Subclasses must implement `get_config()` to return the configuration
    as a dictionary that can be passed to the runtime entrypoint.
    """

    @abstractmethod
    def get_config(self) -> dict:
        """Return the training configuration as a dictionary.

        The returned dict is passed to the runtime entrypoint as arguments
        or mounted as a config file, depending on the backend.
        """
        ...

    def get_framework_args(self) -> dict:
        """Default implementation: delegates to get_config().

        Subclasses may override to separate framework args from config args.
        """
        return self.get_config()

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate framework and trainer_type compatibility.

        ConfigTrainer requires runtimes with trainer_type == BUILTIN_TRAINER.
        """
        super().validate_runtime(runtime)
        if runtime.trainer.trainer_type != TrainerType.BUILTIN_TRAINER:
            raise ValueError(
                f"{type(self).__name__} requires a runtime with "
                f"trainer_type={TrainerType.BUILTIN_TRAINER.value}, but "
                f"runtime '{runtime.name}' has "
                f"trainer_type={runtime.trainer.trainer_type.value}"
            )
```

Concrete config-driven trainers extend `ConfigTrainer`. These are **not part of this
proposal's implementation scope** — they will be proposed in follow-up KEPs. The
examples below illustrate the extension pattern:

```python
# Example: future TorchTuneTrainer

@dataclass
class TorchTuneTrainer(ConfigTrainer):
    """Config-driven trainer for TorchTune recipes.

    Replaces the current BuiltinTrainer(config=TorchTuneConfig(...)) pattern.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torch",)

    config: TorchTuneConfig

    def get_config(self) -> dict:
        return self.config.to_dict()
```

```python
# Example: future UnslothTrainer

@dataclass
class UnslothTrainer(ConfigTrainer):
    """Config-driven trainer for Unsloth fine-tuning."""

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torch",)

    model: str
    dataset: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    def get_config(self) -> dict:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
        }
```

```python
# Example: future VeRLTrainer

@dataclass
class VeRLTrainer(ConfigTrainer):
    """Config-driven trainer for VeRL RLHF training."""

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torch",)

    model: str
    reward_model: str
    algorithm: str = "ppo"

    def get_config(self) -> dict:
        return {
            "model": self.model,
            "reward_model": self.reward_model,
            "algorithm": self.algorithm,
        }
```

#### BuiltinTrainer Migration Path

The existing `BuiltinTrainer` with `TorchTuneConfig` is conceptually a `ConfigTrainer`.
This proposal does **not** deprecate or modify `BuiltinTrainer`, but `ConfigTrainer`
is designed to be its successor:

| Aspect | `BuiltinTrainer` (current) | `ConfigTrainer` subclass (future) |
|---|---|---|
| Runtime discovery | None (hardcoded) | Auto-discovery via `supported_frameworks` |
| Framework validation | None | `validate_runtime()` |
| RuntimeConfig support | No | Yes |
| Extension model | Modify `BuiltinTrainer` class | Create new `ConfigTrainer` subclass |
| Config type | Hardcoded `TorchTuneConfig` | Any config via `get_config()` |

A follow-up proposal will define the concrete migration from `BuiltinTrainer` to
`ConfigTrainer` subclasses for TorchTune and other config-driven frameworks. Until then,
`BuiltinTrainer` remains fully supported and unchanged.

### D. RuntimeConfig

Extract runtime-environment settings from `CustomTrainer` into a dedicated dataclass.
This provides a clean separation of concerns and allows runtime configuration to be
reused across any trainer type.

```python
@dataclass
class RuntimeConfig:
    """Per-job runtime environment configuration.

    Separates runtime-environment concerns (what packages to install, what
    environment variables to set) from training-loop and scaling concerns.

    This is passed to `TrainerClient.train()` and applies regardless of
    the trainer type used. It is a separate parameter on `train()` — not
    embedded in trainers — because runtime configuration is orthogonal to
    both trainer type and initializer. The same RuntimeConfig applies to
    all training pods, and keeping it at the `train()` call site allows
    users to set custom env for initializers in the future without changing
    the trainer classes.

    Args:
        packages_to_install: Python packages to install before running the
            training function (e.g., ["transformers>=4.40", "datasets"]).
        pip_index_urls: PyPI index URLs. The first URL is the primary index;
            remaining URLs are extra indexes.
        env: Environment variables to set in all training nodes.
    """

    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: Optional[dict[str, str]] = None
```

**Design decisions:**

- Uses `@dataclass` (not Pydantic `BaseModel`) to be consistent with the rest of the
  SDK codebase.
- Field names (`packages_to_install`, `pip_index_urls`) are consistent with the
  existing `CustomTrainer` fields and with KFP's `PipelinesClient`.
- Pip configuration is flattened into `RuntimeConfig` rather than nested in a separate
  `PipConfig` type, keeping the API surface minimal. Additional pip options (e.g.,
  `--quiet`, `--user`) can be added as fields later if needed.
- `RuntimeConfig` is a separate `train()` parameter — not embedded in trainers —
  because runtime configuration is orthogonal to trainer type. The same `RuntimeConfig`
  should be usable with `CustomTrainer`, `FuncTrainer`, or `ConfigTrainer` subclasses.
  It also allows future extension to apply env vars to initializers.
- `RuntimeConfig` is optional — when not provided, the trainer's own fields
  (`packages_to_install`, `env` on `CustomTrainer`) or the runtime defaults are used.
  This preserves backward compatibility.
- **Merge semantics:** When both `RuntimeConfig` and `CustomTrainer` fields are
  provided, `RuntimeConfig` fields override `CustomTrainer` fields **only when the
  `RuntimeConfig` field is not `None`**. For example, if
  `RuntimeConfig(env={"DEBUG": "1"})` is passed alongside
  `CustomTrainer(packages_to_install=["torch"])`, the trainer's `packages_to_install`
  is preserved because `RuntimeConfig.packages_to_install` is `None`. This is a
  field-level merge, not a wholesale replacement.

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
5. The backend dispatches based on the trainer type: `FuncTrainer` subclasses are
   handled via `get_train_func()` / `get_train_func_args()`, while `ConfigTrainer`
   subclasses are handled via `get_config()`. Both share `get_framework_args()`.

When `runtime_config` is provided, its values take precedence over any
runtime-environment fields on `CustomTrainer` (for backward compatibility, those
fields remain on `CustomTrainer` but `RuntimeConfig` is the preferred mechanism).

**`runtime=None` behavior by trainer type:**

| Trainer type | `runtime=None` behavior |
|---|---|
| `CustomTrainer` / `BuiltinTrainer` | Defaults to `constants.DEFAULT_TRAINING_RUNTIME` ("torch-distributed"), preserving existing behavior. |
| `BaseTrainer` subclasses (`TorchTrainer`, etc.) | Auto-discovery via `_resolve_runtime()` — finds runtimes matching `supported_frameworks`. |

This distinction is intentional: existing code must not change behavior, while new
trainer types benefit from auto-discovery. The `train()` docstring documents this
difference explicitly.

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

    # Auto-discover: find runtimes matching the trainer's frameworks.
    # Iterate supported_frameworks in declaration order (most preferred first)
    # to provide deterministic selection when exactly one runtime matches
    # the most-preferred framework.
    all_runtimes = self.list_runtimes()
    matching = []
    for framework in trainer.supported_frameworks:
        matching = [
            r for r in all_runtimes
            if r.trainer.framework == framework
        ]
        if matching:
            break

    if len(matching) == 0:
        raise ValueError(
            f"No runtime found for frameworks {trainer.supported_frameworks}. "
            f"Available runtimes: {[r.name for r in all_runtimes]}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"Multiple runtimes found for framework "
            f"'{matching[0].trainer.framework}': "
            f"{[r.name for r in matching]}. "
            f"Please specify the runtime explicitly."
        )

    return matching[0]
```

**Multi-runtime selection strategy:**

Auto-discovery iterates `supported_frameworks` in declaration order (most preferred
first). For each framework, it collects matching runtimes. If exactly one runtime
matches the most-preferred framework, it is selected automatically. If multiple
runtimes match the same framework, the SDK raises a `ValueError` listing the
available options. If no runtimes match the most-preferred framework, discovery
falls through to the next framework in the tuple.

For example, `DeepSpeedTrainer` declares `supported_frameworks = ("deepspeed", "torch")`.
On a cluster with only a `torch-distributed` runtime, auto-discovery falls through
`"deepspeed"` (no match) and selects the `torch-distributed` runtime. On a cluster
with both a `deepspeed-mpi` and a `torch-distributed` runtime, auto-discovery finds
`deepspeed-mpi` first (matching `"deepspeed"`) and selects it without ambiguity.

When multiple runtimes match the same framework, the user resolves the ambiguity by
passing the `runtime` parameter to `train()`:

```python
# Two torch runtimes exist: "torch-distributed" and "torch-elastic"
# Auto-discovery raises ValueError listing both options.

# User resolves by specifying explicitly:
client.train(
    runtime="torch-elastic",
    trainer=TorchTrainer(func=my_fn, num_nodes=4),
)
```

This is a deliberate design choice. The `runtime` parameter on `train()` is the
single, existing mechanism for runtime selection. Adding a `runtime_name` to
`RuntimeConfig` or to trainer classes would conflate concerns — `RuntimeConfig` is for
packages and environment, trainers are for training logic, and runtime selection belongs
to the `train()` call site. See also
[Alternative #4](#4-automatic-runtime-selection-with-scoringranking-instead-of-strict-single-match)
for why priority-based scoring was rejected.

### Runtime Validation

Validation happens at three levels:

1. **Framework label check** (in `BaseTrainer.validate_runtime()`): Ensures the
   runtime's `trainer.kubeflow.org/framework` label value is in the trainer's
   `supported_frameworks` list.

2. **Trainer type check** (in `FuncTrainer.validate_runtime()` and
   `ConfigTrainer.validate_runtime()`): Ensures the runtime's `trainer_type` matches
   the trainer category. `FuncTrainer` requires `TrainerType.CUSTOM_TRAINER`;
   `ConfigTrainer` requires `TrainerType.BUILTIN_TRAINER`. This catches mismatches
   such as passing a function-driven trainer to a config-only runtime.

3. **Framework-specific checks** (in concrete trainer overrides): For example,
   `DeepSpeedTrainer` could verify that the runtime's launcher configuration
   (torchrun vs. mpirun) is compatible with the selected runtime.

**Validation strategy — SDK vs. control plane:**

SDK validation raises `ValueError` at submission time, *before* the `TrainJob` CR is
created in the cluster. This is intentionally a hard fail, not a warning, because:

1. **Fast feedback.** A `ValueError` with a clear message is immediate. A warning
   that the user ignores leads to a `TrainJob` that fails minutes later in the
   controller or at execution time, wasting cluster resources.
2. **Deterministic checks.** The SDK validates against concrete, known properties
   (framework label, `trainer_type`) — not heuristics. These checks are
   authoritative at the SDK level.
3. **Control plane remains the final arbiter.** The controller's webhook may enforce
   additional constraints (resource quotas, policy, version compatibility) that the
   SDK does not know about. SDK validation is a *subset* of control-plane validation,
   not a replacement.
4. **Overridable.** Subclasses can override `validate_runtime()` to relax or extend
   validation for custom use cases.

In summary: the SDK fails fast on checks it *can* perform (framework, trainer_type),
and defers to the controller for checks it *cannot* perform (quotas, policies).

### Trainer Responsibility Boundary

Trainers are **data objects**, not builders. They expose structured data about the
training job; they do not construct the `TrainJob` CRD, build container entrypoints,
or interact with the Kubernetes API. The responsibility boundary is:

| Concern | Owner | Rationale |
|---|---|---|
| Training function / config | **Trainer** (`get_train_func()`, `get_config()`) | Trainer knows what to run |
| Framework-specific CLI args | **Trainer** (`get_framework_args()`) | Trainer knows its framework's options |
| Scaling & resources | **Trainer** (`num_nodes`, `resources_per_node`) | User sets these on the trainer |
| Serializing function into `command` | **Backend** (`get_command_using_train_func()`) | Backend knows the serialization format |
| Building `TrainJob` CRD / container spec | **Backend** | Backend knows the target platform (K8s, container, local) |
| Injecting distributed args (`rdzv_endpoint`, `nnodes`, etc.) | **Controller** | Controller owns the distributed topology |
| Enforcing policies, quotas, webhooks | **Controller** | Control plane is the final authority |

This separation ensures that:
- Adding a new trainer does **not** require changes to the backend — only a new
  `FuncTrainer` or `ConfigTrainer` subclass.
- Adding a new backend does **not** require changes to trainers — backends consume
  the same `BaseTrainer` interface.
- The controller continues to own distributed coordination args, avoiding conflicts
  between SDK-provided and controller-injected arguments.

### Framework Argument Separation

The current `CustomTrainer.func_args` dict mixes user hyperparameters with framework
arguments. The three-level hierarchy solves this structurally:

| Layer | Method | Contains | Maps to in `TrainJob` CRD |
|---|---|---|---|
| `FuncTrainer` | `get_train_func_args()` | User hyperparameters | Embedded in serialized `trainer.command` |
| `ConfigTrainer` | `get_config()` | Full training configuration | `trainer.args` (parsed by runtime entrypoint) |
| `BaseTrainer` | `get_framework_args()` | Framework CLI args not injected by the controller | Appended to `trainer.args` |

Arguments that the Kubeflow Trainer controller already injects (e.g., `rdzv_endpoint`,
`nnodes`, `nproc_per_node`, `node_rank`) are **excluded** from `get_framework_args()`.
The specialized trainer documentation explicitly lists which arguments it manages vs.
which the controller manages.

### Backend Integration

Each backend (`KubernetesBackend`, `ContainerBackend`, `LocalProcessBackend`) must be
updated to handle `BaseTrainer` instances. The backend reads from the trainer's
interface methods and maps them to platform-specific constructs:

```python
# In KubernetesBackend — building the TrainJob CR:

def _build_trainer_cr(self, runtime, trainer):
    trainer_cr = TrainerV1alpha1Trainer()
    trainer_cr.num_nodes = trainer.num_nodes
    trainer_cr.resources_per_node = trainer.resources_per_node
    trainer_cr.image = trainer.image

    if isinstance(trainer, FuncTrainer):
        # Serialize function into command (same as CustomTrainer today)
        trainer_cr.command = get_command_using_train_func(
            runtime,
            trainer.get_train_func(),
            trainer.get_train_func_args(),
            runtime_config.pip_index_urls if runtime_config else None,
            runtime_config.packages_to_install if runtime_config else None,
        )
        # Framework args go into trainer.args
        framework_args = trainer.get_framework_args()
        if framework_args:
            trainer_cr.args = [
                f"--{k}={v}" for k, v in framework_args.items()
            ]

    elif isinstance(trainer, ConfigTrainer):
        # Config-driven: use runtime's command, pass config as args
        trainer_cr.command = list(runtime.trainer.command)
        trainer_cr.args = [
            f"{k}={v}" for k, v in trainer.get_config().items()
        ]

    return trainer_cr
```

The `runtime_config` parameter is applied uniformly: packages are installed in the
init container, environment variables are set on all training pods.

### Type Hierarchy Diagram

```
                        BaseTrainer (ABC)
                        ├── supported_frameworks (ClassVar)
                        ├── num_nodes, resources_per_node, image
                        ├── get_framework_args()  [abstract]
                        └── validate_runtime()
                              │
              ┌───────────────┴───────────────┐
              │                               │
        FuncTrainer (ABC)              ConfigTrainer (ABC)
        ├── func: Callable             ├── get_config()  [abstract]
        ├── func_args: dict            └── get_framework_args()
        ├── get_train_func()                  │
        └── get_train_func_args()             │  (future, via follow-up proposals)
              │                               │
    ┌─────────┼─────────┬──────────┐    ┌─────┼──────────┬──────────┐
    │         │         │          │    │     │          │          │
  Torch   DeepSpeed   JAX    XGBoost  Torch  Unsloth   VeRL    Axolotl
  Trainer  Trainer  Trainer  Trainer  Tune   Trainer  Trainer  Trainer
              │                      Trainer
              │
    (supports both "deepspeed"
     and "torch" frameworks)


    Existing (unchanged):

    CustomTrainer          BuiltinTrainer         CustomTrainerContainer
    (flat dataclass,       (TorchTuneConfig,      (image-based,
     no base class)         no base class)          no base class)


    New:

    RuntimeConfig
    (per-job env: packages, pip URLs, env vars)
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
        packages_to_install=["transformers", "datasets"],
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

**DeepSpeed example:**

```python
from kubeflow.trainer import DeepSpeedTrainer, RuntimeConfig

job_id = client.train(
    trainer=DeepSpeedTrainer(
        func=train_deepspeed,
        num_nodes=8,
        resources_per_node={"gpu": 4, "memory": "32Gi"},
        num_proc_per_node=4,
        deepspeed_config={
            "train_batch_size": 32,
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        },
    ),
    runtime_config=RuntimeConfig(
        packages_to_install=["deepspeed"],
    ),
)
```

---

## Migration and Backward Compatibility

| Aspect | Impact |
|---|---|
| `CustomTrainer` | **No change.** Remains fully functional. `packages_to_install`, `pip_index_urls`, and `env` fields are retained. |
| `CustomTrainerContainer` | **No change.** |
| `BuiltinTrainer` | **No change in Alpha.** `ConfigTrainer` is designed as its successor (see [BuiltinTrainer Migration Path](#builtintrainer-migration-path)). In Beta, `BuiltinTrainer` will emit a `FutureWarning` directing users to `TorchTuneTrainer`. Formal deprecation occurs at GA. |
| `TrainerClient.train()` | **Additive only.** New `runtime_config` parameter is optional with default `None`. The `trainer` parameter type union is extended to include `BaseTrainer`. |
| `TrainJobTemplate` | **No change in this proposal.** Future work can extend it to support `BaseTrainer` subclasses. |
| `RuntimeConfig` vs `CustomTrainer` fields | When both `RuntimeConfig` and `CustomTrainer` fields are provided, `RuntimeConfig` takes precedence. This is documented but does not break existing code since `RuntimeConfig` defaults to `None`. |
| Python version | No new Python version requirements. Uses `dataclass`, `ABC`, `ClassVar` — all available in Python 3.9+. |
| SDK public exports | New classes are exported from `kubeflow.trainer` (`BaseTrainer`, `FuncTrainer`, `ConfigTrainer`, `TorchTrainer`, `DeepSpeedTrainer`, `JAXTrainer`, `XGBoostTrainer`, `RuntimeConfig`). No existing exports are removed or renamed. |

---

## Test Plan

### Unit Tests

1. **Type hierarchy compliance**: Verify that each `FuncTrainer` subclass correctly
   inherits `func`/`func_args` fields and each `ConfigTrainer` subclass implements
   `get_config()`.
2. **`validate_runtime()` — positive**: Each trainer validates a runtime with a
   matching framework label and compatible `trainer_type`.
3. **`validate_runtime()` — negative framework**: Each trainer raises `ValueError`
   for a runtime with a non-matching framework label.
4. **`validate_runtime()` — negative trainer_type**: `FuncTrainer` subclass raises
   `ValueError` for a `BUILTIN_TRAINER` runtime; `ConfigTrainer` subclass raises
   `ValueError` for a `CUSTOM_TRAINER` runtime.
5. **`get_framework_args()`**: Verify that each trainer returns only non-overlapping
   arguments (excludes controller-injected args).
6. **`RuntimeConfig` defaults**: Verify `None` defaults and precedence over
   `CustomTrainer` fields.
7. **Runtime auto-discovery — single match**: Mock `list_runtimes()` to return one
   matching runtime; verify it is selected.
8. **Runtime auto-discovery — no match**: Mock `list_runtimes()` to return no
   matching runtimes; verify `ValueError`.
9. **Runtime auto-discovery — multiple matches**: Mock `list_runtimes()` to return
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

**Phase 1: Core Type Hierarchy and RuntimeConfig**
- Add `BaseTrainer`, `FuncTrainer`, `ConfigTrainer` to `kubeflow/trainer/types/types.py`
- Add `RuntimeConfig` dataclass
- Add `_resolve_runtime()` to `TrainerClient`
- Extend `TrainerClient.train()` signature
- Unit tests for the type hierarchy and RuntimeConfig

**Phase 2: TorchTrainer**
- Implement `TorchTrainer` (extends `FuncTrainer`)
- Update `KubernetesBackend`, `ContainerBackend`, `LocalProcessBackend` to handle
  `FuncTrainer` and `ConfigTrainer` dispatch
- Integration tests
- Documentation and examples

**Phase 3: DeepSpeedTrainer, JAXTrainer, XGBoostTrainer**
- Implement remaining `FuncTrainer` subclasses
- DeepSpeedTrainer with multi-runtime support (torch and deepspeed/MPI runtimes)
- Framework-specific validation and argument handling
- Tests and documentation

**Phase 4: Public API exports and documentation**
- Export new classes from `kubeflow.trainer.__init__`
- Update SDK documentation on sdk.kubeflow.org
- Add migration guide examples

> **Note:** Phase 1 and Phase 2 should ship together in the same SDK release.
> Releasing the type hierarchy without backend support would make `TorchTrainer`
> importable but unusable, producing a confusing `ValueError` at `train()` time.

---

## Graduation Criteria

### Alpha (target: current cycle)

- `BaseTrainer`, `FuncTrainer`, `ConfigTrainer`, and `RuntimeConfig` types are
  implemented and exported.
- `TorchTrainer` is implemented with full backend support (Kubernetes, Container,
  LocalProcess).
- Runtime auto-discovery and `validate_runtime()` are functional.
- Unit tests cover the type hierarchy, validation (positive and negative), and
  auto-discovery (single-match, no-match, multi-match).
- At least one integration test exercises `TorchTrainer` end-to-end against the
  Kubernetes backend.
- All existing `CustomTrainer` and `BuiltinTrainer` tests continue to pass.
- `RuntimeConfig` merge semantics are implemented and tested.

### Beta

- `DeepSpeedTrainer`, `JAXTrainer`, and `XGBoostTrainer` are implemented.
- At least one `ConfigTrainer` subclass (`TorchTuneTrainer`) is implemented,
  proving the config-driven extension model and beginning `BuiltinTrainer` migration.
- `BuiltinTrainer` emits a `FutureWarning` directing users to `TorchTuneTrainer`.
- SDK documentation on sdk.kubeflow.org covers all new trainer types with examples.
- Migration guide is published.

### GA

- All Tier 1 trainers are stable with no breaking changes for at least one release.
- `BuiltinTrainer` is formally deprecated (removal deferred to a future major version).
- `CustomTrainer` runtime-environment fields (`packages_to_install`, `pip_index_urls`,
  `env`) are formally deprecated in favor of `RuntimeConfig`.
- Community has contributed at least one Tier 2 `ConfigTrainer` subclass.

---

## Open Questions

The following design questions should be resolved before or during implementation:

1. **DeepSpeed launcher detection.** `DeepSpeedTrainer` supports both `torchrun` and
   `mpirun` launchers. How should `validate_runtime()` detect which launcher a runtime
   uses? The `Runtime` type currently does not expose launcher metadata. Options:
   (a) inspect `runtime.trainer.command`, (b) add a launcher label to runtimes,
   (c) defer validation to the controller.

2. **`ConfigTrainer.get_config()` return type.** The current return type `dict` is
   untyped. Should `ConfigTrainer` subclasses return a typed configuration object
   instead (e.g., a Pydantic model or typed dataclass) to get the same static-analysis
   benefits that `FuncTrainer` provides via typed fields?

3. **Observability.** When runtime auto-discovery selects a runtime, should the SDK
   log the selected runtime name and framework at `INFO` level? This would aid
   debugging but adds a logging dependency.

4. **`resources_per_node` typing.** The current `Optional[dict]` is untyped. Should
   this be a structured type (e.g., `ResourceRequirements` dataclass with `cpu`,
   `memory`, `gpu` fields) to enable IDE autocomplete and validation?

5. **Backend validation safety net.** Currently, `validate_runtime()` is only called
   via `TrainerClient._resolve_runtime()`. Should backends also call
   `validate_runtime()` as a defensive check, in case trainers are passed to backends
   directly?

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
- Users may want custom env for initializers too. A separate `train()` parameter
  can be extended to apply to both trainers and initializers without changing trainer
  classes.

### 4. Automatic runtime selection with scoring/ranking instead of strict single-match

When multiple runtimes match a framework, automatically pick the "best" one using a
scoring heuristic (e.g., prefer non-deprecated, prefer more specific labels).

**Rejected because:**
- Implicit selection heuristics are fragile and hard to debug. When multiple runtimes
  exist for the same framework, it is a deliberate platform configuration and the
  user should explicitly choose.
- A clear error message listing available runtimes is more useful than a possibly
  wrong automatic selection.

### 5. Flat hierarchy: all trainers inherit directly from BaseTrainer

Have all concrete trainers (both function-driven and config-driven) inherit directly
from `BaseTrainer` without the `FuncTrainer` / `ConfigTrainer` intermediate layer.

**Rejected because:**
- Config-driven trainers would carry `get_train_func()` returning `None` — semantically
  incorrect and error-prone.
- Function-driven trainers would each redeclare `func` and `func_args` fields —
  unnecessary repetition.
- Backend dispatch would rely on runtime checks (`if trainer.get_train_func() is None`)
  instead of type checks (`isinstance(trainer, ConfigTrainer)`).
- The intermediate classes encode the fundamental difference between the two trainer
  modes at the type level, making the API self-documenting.

### 6. Have specialized trainers inherit from CustomTrainer

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
