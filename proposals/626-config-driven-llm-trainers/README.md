# KEP-626: Config-Driven LLM Trainers

## Authors

- Yassin Nouh - [@YassinNouh21](https://github.com/YassinNouh21)

Tracking issue: [kubeflow/sdk#626](https://github.com/kubeflow/sdk/issues/626). Builds on
[KEP-285: Specialized Trainer Abstractions](https://github.com/kubeflow/sdk/pull/308) by
Saad Zaher — [@szaher](https://github.com/szaher) — which remains the canonical issue for the
broader specialized-trainers effort; this KEP is limited to config-driven LLM/RLHF trainers.

<!-- toc -->
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
  - [BaseTrainer](#basetrainer)
  - [ConfigTrainer](#configtrainer)
  - [FrameworkConfig](#frameworkconfig)
  - [Dynamic Registration](#dynamic-registration)
  - [TRLConfig](#trlconfig)
  - [TorchTune Transition](#torchtune-transition)
  - [User-Facing API](#user-facing-api)
  - [Backend Changes](#backend-changes)
  - [Control Plane](#control-plane)
  - [Framework Analysis](#framework-analysis)
- [Migration and Backward Compatibility](#migration-and-backward-compatibility)
- [Implementation Phases](#implementation-phases)
- [Test Plan](#test-plan)
- [Open Questions](#open-questions)
- [Alternatives Considered](#alternatives-considered)
  - [1. One trainer class per framework](#1-one-trainer-class-per-framework)
  - [2. One config class per post-training method](#2-one-config-class-per-post-training-method)
  - [3. Entry-point discovery for out-of-tree frameworks](#3-entry-point-discovery-for-out-of-tree-frameworks)
- [Implementation History](#implementation-history)
- [References](#references)
<!-- /toc -->

## Summary

This KEP makes config-driven LLM fine-tuning frameworks pluggable in the Kubeflow SDK.
Today TorchTune is the only framework the SDK can represent, hardcoded at four points; DPO
and GRPO post-training cannot be expressed at all. The proposal introduces a small
`FrameworkConfig` base class (framework label + entrypoint command + argument rendering), an
explicit registration decorator, and `TRLConfig` as the first framework built on it —
reachable through one concrete `ConfigTrainer` while the existing `BuiltinTrainer` keeps its
exact public signature.

It builds on [Saad's KEP-285 proposal](https://github.com/kubeflow/sdk/pull/308) and keeps
its `BaseTrainer` foundation. Per the 2026-07-15 community call, the scope is deliberately
narrowed to config-driven trainers only; function-driven trainers (`TorchTrainer`,
`DeepSpeedTrainer`, ...) and `RuntimeConfig` follow in later phases.

## Motivation

TorchTune is hardcoded in the SDK at four points:

| # | Coupling | Location |
|---|---|---|
| 1 | `BuiltinTrainer.config` is annotated with the concrete `TorchTuneConfig` type | `types.py:233-242` |
| 2 | The framework identifier is derived by reflecting on that annotation. The code's own comment reads "Change it to list: BUILTIN_CONFIGS, once we support more Builtin Trainer configs." | `types.py:245-246` |
| 3 | `trainer_type` and the container entrypoint are selected by string-comparing the runtime's framework label against that constant | `utils.py:124-129`, `:151-158` |
| 4 | Config-to-argument translation is guarded by an `isinstance` check against `TorchTuneConfig` | `utils.py:461-462` |

Coupling #3 is load-bearing: a runtime labelled `trainer.kubeflow.org/framework: trl`
resolves to `CUSTOM_TRAINER` today, so no config-driven framework other than TorchTune can
run, however the types are arranged.

Two external facts sharpen the urgency. TorchTune's upstream development stopped on 15 July
2025 ([meta-pytorch/torchtune#2883](https://github.com/meta-pytorch/torchtune/issues/2883)),
and the Trainer project plans to remove its TorchTune runtimes. Meanwhile GRPO-style
post-training via TRL is already in flight in the Trainer
([kubeflow/trainer#3508](https://github.com/kubeflow/trainer/issues/3508),
[kubeflow/trainer#3718](https://github.com/kubeflow/trainer/pull/3718)) with no SDK surface
to drive it.

### Goals

1. Introduce `ConfigTrainer` and the `FrameworkConfig` contract so the framework is carried
   by a config object, not hardcoded in the SDK.
2. Support Hugging Face TRL as the first framework: SFT, DPO, and GRPO.
3. Provide dynamic registration so in-tree and out-of-tree frameworks plug in without
   touching backend code.
4. Maintain 100% backward compatibility: `BuiltinTrainer(config=TorchTuneConfig(...))` keeps
   its exact behavior; nothing is deprecated.
5. Record a framework-landscape analysis to ground which frameworks the pattern covers and
   what each needs from the control plane.

### Non-Goals

1. Function-driven specialized trainers (`TorchTrainer`, `DeepSpeedTrainer`, `JAXTrainer`,
   `XGBoostTrainer`) and `RuntimeConfig` — deferred to a later phase; Saad's proposal
   remains the reference design for them.
2. Control-plane extension points for arbitrary frameworks — deferred; see
   [Open Questions](#open-questions).
3. Automatic runtime provisioning (SDK creating `TrainingRuntime` objects).
4. New discovery mechanisms: `trainer.kubeflow.org/framework` remains the sole discovery
   key.

## Proposal

### BaseTrainer

The minimal abstract root from Saad's proposal, kept for continuity. Later phases add the
function-driven branch under it; this KEP does not specify that branch.

```python
# kubeflow/trainer/types/types.py

@dataclass(kw_only=True)
class BaseTrainer(ABC):
    """Abstract base for specialized trainers.

    Class Attributes:
        supported_frameworks: Framework labels this trainer supports. Must match
            values of the `trainer.kubeflow.org/framework` runtime label.
    """

    supported_frameworks: ClassVar[tuple[str, ...]]

    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    image: Optional[str] = None

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Raise ValueError if the runtime's framework label is unsupported."""
        if runtime.trainer.framework not in self.supported_frameworks:
            raise ValueError(
                f"{type(self).__name__} supports frameworks "
                f"{self.supported_frameworks}, but runtime '{runtime.name}' "
                f"has framework '{runtime.trainer.framework}'"
            )
```

The hierarchy is `@dataclass(kw_only=True)`: `BaseTrainer`'s fields have defaults, so
without `kw_only` a subclass could not declare a required field. Existing types
(`CustomTrainer`, `BuiltinTrainer`) keep their bare declarations; their construction
signatures are public API.

### ConfigTrainer

One concrete class for all config-driven jobs — jobs where the runtime image owns the
training loop (`tune run`, `trl`) and the user supplies only parameters. (`FrameworkConfig`
is specified in the next section.)

```python
@dataclass(kw_only=True)
class ConfigTrainer(BaseTrainer):
    """Trainer for config-driven frameworks: carries a FrameworkConfig that
    fully describes the training job; the runtime's entrypoint executes it."""

    config: FrameworkConfig

    @property
    def supported_frameworks(self) -> tuple[str, ...]:
        """The framework is carried by the config, not fixed per class."""
        return (self.config.framework,)

    def validate_runtime(self, runtime: "Runtime") -> None:
        super().validate_runtime(runtime)
        if runtime.trainer.trainer_type != TrainerType.BUILTIN_TRAINER:
            raise ValueError(
                f"{type(self).__name__} requires a runtime with "
                f"trainer_type={TrainerType.BUILTIN_TRAINER.value}"
            )
```

`ConfigTrainer` is **not** subclassed per framework: a `ConfigTrainer` holding a `TRLConfig`
and one holding a `TorchTuneConfig` differ in data, not in what the trainer does — it hands
the config's `command` and `to_args()` to the backend. The framework axis therefore lives in
the config ([Alternative 1](#1-one-trainer-class-per-framework) is the rejected pole).

`BuiltinTrainer` stays exactly where it is today, existing and unchanged: same construction
signature, its `config` field widened from the concrete `TorchTuneConfig` to the
`FrameworkConfig` base. Both entry points accept any registered framework:

```python
BuiltinTrainer(config=TorchTuneConfig(...))   # today, still valid
ConfigTrainer(config=TRLConfig(...))          # new code, same machinery
```

### FrameworkConfig

A config answers the three questions the SDK must ask to run any config-driven job — which
runtime label matches it, which entrypoint runs it, and how it renders itself as that
entrypoint's arguments:

```python
# kubeflow/trainer/types/types.py

@dataclass(kw_only=True)
class FrameworkConfig(abc.ABC):
    """Base class for the configuration of a config-driven framework."""

    framework: ClassVar[str] = ""            # claims the runtime label
    command: ClassVar[tuple[str, ...]] = ()  # container entrypoint

    @abstractmethod
    def to_args(self) -> list[str]:
        """Render this config as arguments for `command`."""
        ...
```

`to_args()` is abstract because rendering is the one behavior that genuinely differs per
framework — TorchTune emits nested `model.lora_rank=8` overrides, TRL emits
`--flag value` pairs. Each framework's rendering lives in its own class and is added without
touching any other; a shared renderer with a style switch would just relocate the
`isinstance` ladder this KEP deletes.

The name follows the SDK's existing vocabulary: `backends/` means *execution* backend
(`KubernetesBackend`, `ContainerBackend`), while this object is the `config=` argument.
KEP-2839 called the same interface `LLMBackend`; this proposal supersedes that SDK portion
and renames it.

### Dynamic Registration

The framework label resolves through a registry rather than a constant. It serves exactly
one lookup — the one `get_runtime_trainer()` performs — and is the extension path for
out-of-tree frameworks.

```python
# kubeflow/trainer/types/registry.py

_FRAMEWORK_CONFIGS: dict[str, type["FrameworkConfig"]] = {}


def register_framework(cls: type["FrameworkConfig"]) -> type["FrameworkConfig"]:
    """Claim the framework label declared in `cls.framework`."""
    if not (cls.framework and cls.command):
        raise ValueError(f"{cls.__name__} must declare a framework and a command")
    _FRAMEWORK_CONFIGS[cls.framework] = cls
    return cls


def get_framework(framework: str) -> Optional[type["FrameworkConfig"]]:
    return _FRAMEWORK_CONFIGS.get(framework)
```

Registration is an explicit decorator, keyed off the `ClassVar` so there is no string to
drift, matching how the ecosystem already registers plugins: the Trainer control plane's
`pkg/runtime/framework/plugins/registry.go`, KEP-2839's `@register_backend` sketch, and the
[sdk#310](https://github.com/kubeflow/sdk/pull/310) PoC. There is no entry-point discovery:
an out-of-tree config must be imported before it can be constructed, and importing it
registers it, so the decorator is sufficient. Registration is import-time and
last-writer-wins; in-tree configs are exported from `kubeflow/trainer/__init__.py` so they
are always registered.

An out-of-tree framework is one decorated `FrameworkConfig` subclass in any pip package —
no SDK change, no backend change.

### TRLConfig

The first framework. The runtime entrypoint is the TRL CLI, which covers SFT, DPO, and GRPO
as subcommands; the post-training *method* is a field, not a class, because methods differ
only in which fields apply under the same command
([Alternative 2](#2-one-config-class-per-post-training-method)).

```python
# kubeflow/trainer/types/trl.py
# Imports: `from kubeflow.trainer.types.registry import register_framework`,
# `from kubeflow.trainer.types.types import FrameworkConfig`

class TRLMethod(Enum):
    """Post-training method; the value is the TRL CLI subcommand."""

    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


@register_framework
@dataclass
class TRLConfig(FrameworkConfig):
    """Configuration for Hugging Face TRL post-training.

    Raises:
        ValueError: If a field is set that does not apply to the selected method.
    """

    framework: ClassVar[str] = "trl"
    command: ClassVar[tuple[str, ...]] = ("trl",)

    method: TRLMethod
    model_name_or_path: str
    dataset_name: str

    learning_rate: Optional[float] = None
    per_device_train_batch_size: Optional[int] = None
    num_train_epochs: Optional[int] = None
    bf16: Optional[bool] = None
    use_peft: Optional[bool] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_target_modules: Optional[list[str]] = None

    beta: Optional[float] = None              # DPO
    max_prompt_length: Optional[int] = None   # DPO
    reward_funcs: Optional[list[str]] = None  # GRPO: reward function identifiers,
    num_generations: Optional[int] = None     # GRPO   passed through to the CLI

    extra_args: Optional[list[str]] = None    # passthrough for CLI flags not modeled
                                              # above; escape hatch against TRL drift

    _METHOD_SCOPED_FIELDS: ClassVar[dict[str, frozenset[str]]] = {
        "beta": frozenset({TRLMethod.DPO.value}),
        "max_prompt_length": frozenset({TRLMethod.DPO.value}),
        "reward_funcs": frozenset({TRLMethod.GRPO.value}),
        "num_generations": frozenset({TRLMethod.GRPO.value}),
    }

    def __post_init__(self) -> None:
        for name, methods in self._METHOD_SCOPED_FIELDS.items():
            if getattr(self, name) is not None and self.method.value not in methods:
                raise ValueError(
                    f"'{name}' applies to {sorted(methods)}, not method={self.method.value}"
                )
        if self.method is TRLMethod.GRPO and not self.reward_funcs:
            raise ValueError("method=grpo requires at least one entry in 'reward_funcs'")

    def to_args(self) -> list[str]:
        """Render as `[<subcommand>, --flag, value, ...]` for the TRL CLI."""
        args: list[str] = [self.method.value]
        for f in fields(self):
            if f.name in ("method", "extra_args"):
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if value is True:
                args.append(f"--{f.name}")           # store_true flag
            elif isinstance(value, list):
                args.append(f"--{f.name}")
                args.extend(str(item) for item in value)
            else:
                args += [f"--{f.name}", str(value)]
        if self.extra_args:
            args.extend(self.extra_args)
        return args
```

The typed fields cover the common surface; `extra_args` is the deliberate escape hatch so
the dataclass does not have to chase every flag of every TRL release — argument semantics
follow the TRL version in the runtime image.

`LoraConfig` is not reused for TRL: it is TorchTune-shaped (`apply_lora_to_output`,
`quantize_base` have no TRL analogue), and TRL's PEFT surface is `--use_peft` / `--lora_r` /
`--lora_alpha` / `--lora_target_modules`. Unsloth is not a config either — it has no CLI or
training loop of its own; an Unsloth-accelerated image is still labelled `trl` and selected
with `train(runtime=...)`.

### TorchTune Transition

`TorchTuneConfig` keeps every field and its construction signature, and gains the
`@register_framework` decorator (claiming the `torchtune` label) plus the three
`FrameworkConfig` members (`framework = "torchtune"`, `command = ("tune", "run")`, and a
`to_args()` that delegates to the existing emitters, which move verbatim from the backend to
`kubeflow/trainer/types/torchtune.py`). Existing jobs produce byte-identical `TrainJob`
arguments.

TorchTune is, however, no longer the flagship: upstream development halted
([#2883](https://github.com/meta-pytorch/torchtune/issues/2883)) and the Trainer project
plans to remove its TorchTune runtimes. `TorchTuneConfig` remains supported for as long as
those runtimes exist; its eventual removal follows the Trainer's deprecation schedule and is
out of scope here.

### User-Facing API

```python
from kubeflow.trainer import ConfigTrainer, TrainerClient, TRLConfig, TRLMethod

TrainerClient().train(
    trainer=ConfigTrainer(
        config=TRLConfig(
            method=TRLMethod.DPO,
            model_name_or_path="Qwen/Qwen2.5-0.5B",
            dataset_name="trl-lib/ultrafeedback_binarized",
            beta=0.1,
        ),
        num_nodes=2,
        resources_per_node={"gpu": 1},
    ),
)
```

When `runtime` is omitted, the SDK lists runtimes and matches the
`trainer.kubeflow.org/framework` label against `config.framework`: exactly one match is
used; zero or multiple matches raise `ValueError` naming the candidates and asking for an
explicit `runtime`. With a runtime given, `validate_runtime()` checks compatibility. This
auto-selection applies only to `ConfigTrainer`; existing trainer types keep today's default
runtime behavior.

### Backend Changes

Two hardcoded branches become lookups; adding a further framework then requires no backend
change at all:

```python
# get_runtime_trainer() — was: framework == types.TORCH_TUNE   (utils.py:124-129)
trainer_type = (
    TrainerType.BUILTIN_TRAINER if get_framework(framework) else TrainerType.CUSTOM_TRAINER
)

# _build_trainer_cr() — was: isinstance + TORCH_TUNE_COMMAND   (utils.py:151-158, 461-462)
trainer_cr.command = list(trainer.config.command)
trainer_cr.args = trainer.config.to_args()
```

Deleted: the reflection-derived `types.TORCH_TUNE` constant, `constants.TORCH_TUNE_COMMAND`,
and the `isinstance(trainer.config, TorchTuneConfig)` guard.

**Classification is capability, not exclusion.** `BUILTIN_TRAINER` marks a runtime as
config-driven *capable*; it must not forbid `CustomTrainer`. Today the backend hard-rejects
`CustomTrainer` against non-CUSTOM runtimes (`backend.py:795`); that check relaxes so
`CustomTrainer` may target any runtime. This matters concretely: the in-flight GRPO runtime
([#3718](https://github.com/kubeflow/trainer/pull/3718)) is driven through a custom
entrypoint script — the `CustomTrainer` pattern — and must keep working unchanged after
`TRLConfig` registers and reclassifies `trl`-labelled runtimes. For out-of-tree frameworks,
classification depends on which configs the process has imported; with the relaxation this
is benign — an unimported framework's runtime lists as `CUSTOM_TRAINER` and remains fully
usable. One behavior stays in the
backend: TorchTune's `dataset.data_files=` / `dataset.data_dir=` override is computed from
the Hugging Face dataset initializer (`utils.py:512-527`) — staging knowledge the backend
owns — and is appended to whatever `to_args()` renders. `TRLConfig` needs nothing from the
initializer: `model_name_or_path` and `dataset_name` are passed through for the `trl` CLI to
resolve.

### Control Plane

This proposal requires no `TrainJob` or `ClusterTrainingRuntime` CRD change and no new
controller plugin. A runtime is an image plus a `command` the SDK appends arguments to:

```yaml
kind: ClusterTrainingRuntime
metadata:
  labels:
    trainer.kubeflow.org/framework: trl    # the SDK's discovery key
spec:
  mlPolicy: { torch: {} }                  # generic distribution plugin injects
  # containers: [{image: ghcr.io/kubeflow/trainer/trl-trainer,   # the torchrun env
  #               command: [trl]}]
```

Per framework, the cost is three artifacts in `kubeflow/trainer`, mirroring what TorchTune
already has: a `cmd/trainers/trl/Dockerfile`, a runtime manifest under
`manifests/base/runtimes/trl/`, and an extension of the existing torch plugin. The torch
plugin dispatches on the trainer command (`torch.go:81` for validation, `torch.go:175` for
command mutation — both match `constants.TorchTuneEntrypoint` and call into `torchtune.go`);
TRL follows the same pattern with a `trl.go` and a `TRLEntrypoint` constant — extending the
plugin, not creating a new one.

**Command ownership** generalizes today's TorchTune flow, in precedence order: the runtime
manifest carries a default `command`; the SDK overrides it with `config.command` +
`to_args()` (exactly as it sets `TORCH_TUNE_COMMAND` today, `utils.py:151-158`); and the
plugin may mutate the final command for distributed wiring, as `torchtune.go` injects the
rendezvous endpoint. Whether TRL needs equivalent mutation for multi-node — or the injected
torchrun env alone suffices for its accelerate launcher — is settled by the Phase-1 PoC;
`trl.go` is the extension point either way.

All three artifacts are Trainer-repository work, coordinated with the in-flight GRPO effort
([#3508](https://github.com/kubeflow/trainer/issues/3508),
[#3718](https://github.com/kubeflow/trainer/pull/3718)), for which this KEP provides the SDK
surface. Note that #3718 currently drives TRL's `GRPOTrainer` through a custom entrypoint
script rather than the `trl` CLI; converging on one runtime shape is part of that
consolidation.

### Framework Analysis

Requested in the 2026-07-15 community call: which frameworks the pattern covers, and what
each needs from the control plane.

| Framework | Entrypoint | Distributed launch | Control-plane needs | Fits `FrameworkConfig`? |
|---|---|---|---|---|
| TRL | `trl sft/dpo/grpo --flags` | accelerate/torchrun (reads injected env) | torch `mlPolicy`; optional `trl.go` validation | Yes — flag CLI, direct fit |
| TorchTune | `tune run <recipe> k=v` | torchrun wrapper | torch `mlPolicy`; existing `torchtune.go` (recipe/config mutation) | Yes — in tree today; upstream halted |
| LlamaFactory | `llamafactory-cli train` — flags **or** yaml+`k=v` overrides (HfArgumentParser/OmegaConf) | torchrun (`FORCE_TORCHRUN`) | torch `mlPolicy`; nothing framework-specific | Yes — reference out-of-tree plugin |
| ms-swift | `swift sft --flags` | torchrun | torch `mlPolicy` | Yes |
| litgpt | `litgpt finetune --flags` | Fabric/torchrun | torch `mlPolicy` | Yes |
| Axolotl | `axolotl train <config.yaml>` | accelerate | torch `mlPolicy`; config **file** must be staged (ConfigMap/volume) | With staging support — file-first CLI |
| Unsloth | no independent CLI | n/a (runs inside TRL) | none of its own | No — acceleration layer, modeled as a `trl`-labelled runtime image |
| TorchForge | early-stage | torchrun | needs investigation | To be analyzed |

Takeaways: the majority of the landscape is flag-based CLIs that fit the config model
directly; file-first CLIs fit once config staging exists (the same class of staging the
backend already does for TorchTune datasets); the one tool that does not fit is not a
framework. Frameworks whose needs exceed "command + args" would require control-plane
extension points — see Open Questions.

## Migration and Backward Compatibility

| Surface | Change |
|---|---|
| `BuiltinTrainer` | **No change to its construction signature.** `config` widens from `TorchTuneConfig` to `FrameworkConfig`; existing calls produce byte-identical `TrainJob` arguments. Nothing is deprecated. |
| `TorchTuneConfig` | Fields and signature unchanged; gains the `FrameworkConfig` ClassVars and `to_args()`. |
| `CustomTrainer` / `CustomTrainerContainer` | Untouched (out of scope for this phase). |
| `TrainerClient.train()` | The `trainer` parameter union extends to accept `ConfigTrainer`. |
| `trl`-labelled runtimes | Reclassified from `CUSTOM_TRAINER` to `BUILTIN_TRAINER` in `list_runtimes()` once `TRLConfig` registers. `CustomTrainer` remains valid against them — the backend's trainer-type check relaxes (see [Backend Changes](#backend-changes)) — so existing #3718-style flows are unaffected. |
| Public exports | Added: `BaseTrainer`, `ConfigTrainer`, `FrameworkConfig`, `TRLConfig`, `TRLMethod`, `register_framework`. None removed or renamed. |
| Python version | `@dataclass(kw_only=True)` needs Python 3.10 — already the SDK's floor (`requires-python = ">=3.10"`). |

## Implementation Phases

- **Phase 1 (this KEP, alpha):** `BaseTrainer` (minimal), `ConfigTrainer`,
  `FrameworkConfig`, registry, `TRLConfig`, backend lookup changes, unit tests, and one E2E
  against a TRL runtime. Trainer-side artifacts (image, manifest, `trl.go`) land in
  kubeflow/trainer in coordination with #3508/#3718.
- **Phase 2:** function-driven trainers (`TorchTrainer`, `DeepSpeedTrainer`, `JAXTrainer`,
  `XGBoostTrainer`) and `RuntimeConfig`, per Saad's proposal; control-plane extension-point
  design informed by the framework analysis.
- **Phase 3:** further frameworks in or out of tree (LlamaFactory as the reference
  out-of-tree plugin; Axolotl once config staging exists).

## Test Plan

- **Unit tests** (`kubeflow/trainer/**/*_test.py`, no network): `FrameworkConfig` contract
  (subclass must declare `framework`/`command`, `register_framework` rejects otherwise);
  `TRLConfig` rendering per method, method-scoped field validation, `extra_args`
  passthrough; registry lookup and last-writer-wins shadowing; `ConfigTrainer`
  `validate_runtime` positive/negative; `get_runtime_trainer()` classification with and
  without a registered framework.
- **Backward-compatibility tests**: `BuiltinTrainer(config=TorchTuneConfig(...))` produces
  byte-identical `TrainJob` command/args before and after the change; `CustomTrainer`
  against a `trl`-labelled runtime keeps working after `TRLConfig` registers.
- **E2E** (Phase 1): one TRL SFT job submitted through `ConfigTrainer` against a TRL
  runtime on a kind cluster, asserting the rendered command/args and job completion.

## Open Questions

1. **Does dynamic registration extend to the control plane?** The registry here is
   SDK-side; the cluster must already hold a labelled runtime and the torch plugin
   dispatches on hardcoded per-framework entrypoint constants. Whether the control plane
   should gain generic extension points — platform engineers registering framework plugins
   the way data scientists register configs — is deferred to Phase 2 and grounded in the
   framework analysis above.
2. **TorchTune removal timeline.** `TorchTuneConfig`'s lifetime follows the Trainer's
   removal of TorchTune runtimes; the SDK should not outlive the runtimes it targets.

## Alternatives Considered

### 1. One trainer class per framework

`TorchTuneTrainer(ConfigTrainer)`, `TRLTrainer(ConfigTrainer)`, ... — the shape of the
original draft. Rejected: the trainer does nothing different per framework (it hands
`command` and `to_args()` to the backend), so the subclasses differ only in constants and
dict-copying boilerplate; each framework becomes a frozen public export; and `BuiltinTrainer`
is forced onto a deprecation path toward a `TorchTuneTrainer` successor for no behavior
change. What it gets right — typed per-framework fields — the accepted design keeps on the
config.

### 2. One config class per post-training method

`SFTConfig`, `DPOConfig`, `GRPOConfig`, mirroring TRL's Python class names. Rejected:
methods differ only in which fields apply under the same command — data, not behavior — and
methods are the volatile axis (TRL moved PPO to `trl.experimental` in a minor release). An
enum member absorbs that churn; an exported class name cannot. Once a second framework
supports DPO, `DPOConfig` needs a framework discriminator anyway.

### 3. Entry-point discovery for out-of-tree frameworks

`importlib.metadata` entry points would discover configs in installed-but-unimported
packages. Rejected for now: a config must be imported to be constructed, and importing
registers it, so discovery only fixes a cosmetic listing case — at the cost of import-time
side effects and non-deterministic registration order. It is purely additive later: an
entry-point loader can feed the same registry dict without breaking any caller.

## Implementation History

- 2026-02-11: [KEP-285 Specialized Trainer Abstractions](https://github.com/kubeflow/sdk/pull/308) proposed by @szaher
- 2026-07-15: Scope narrowed to config-driven LLM/RLHF trainers at the SDK community call;
  this proposal drafted

## References

- [Saad's KEP-285: Specialized Trainer Abstractions](https://github.com/kubeflow/sdk/pull/308) — the foundation this KEP builds on ([tracking issue #285](https://github.com/kubeflow/sdk/issues/285))
- [KEP-2839: Kubeflow Dynamic LLM Trainer Framework](https://github.com/kubeflow/trainer/issues/2839) — prior art superseded here ([proposal PR #3263](https://github.com/kubeflow/trainer/pull/3263), [sdk#310 registry PoC](https://github.com/kubeflow/sdk/pull/310))
- [GRPO support in Kubeflow Trainer](https://github.com/kubeflow/trainer/issues/3508) and [runtime PR #3718](https://github.com/kubeflow/trainer/pull/3718)
- [TRL CLI documentation](https://huggingface.co/docs/trl/en/clis)
- [TorchTune future / development halt](https://github.com/meta-pytorch/torchtune/issues/2883)
- [Runtime Guide — trainer.kubeflow.org/framework label](https://www.kubeflow.org/docs/components/trainer/operator-guides/runtime/)
