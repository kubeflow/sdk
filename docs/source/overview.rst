Overview
========

The Kubeflow SDK is a unified Python SDK that streamlines the user experience for AI practitioners
to interact with various Kubeflow projects.

Architecture
------------

The SDK provides a consistent interface across multiple Kubeflow components:

- **Trainer**: For training and fine-tuning AI models
- **Katib** (Coming Soon): For hyperparameter optimization
- **Pipelines** (Coming Soon): For building and running ML workflows
- **Model Registry** (Coming Soon): For managing model artifacts

Backend System
--------------

The SDK implements a pluggable backend architecture:

**Kubernetes Backend**
    Production backend for running workloads on Kubernetes clusters.
    Uses Custom Resource Definitions (CRDs) for declarative job management.

**Local Process Backend**
    Development backend for rapid prototyping without Kubernetes.
    Runs training jobs as local processes with virtual environments.

Key Features
------------

- **Unified API**: Consistent interface across Kubeflow projects
- **Local Development**: First-class support for local execution
- **Type Safety**: Comprehensive type hints with Pydantic validation
- **Flexible Configuration**: Support for custom trainers and builtin trainers
- **Resource Management**: Configure CPU, GPU, and memory per node
- **Distributed Training**: Multi-node training support

Components
----------

TrainerClient
~~~~~~~~~~~~~

The main client interface for interacting with Kubeflow Trainer:

- List and inspect available runtimes
- Submit training jobs
- Monitor job status and retrieve logs
- Manage job lifecycle

Trainers
~~~~~~~~

**CustomTrainer**
    For user-defined training functions that encapsulate the entire training process.

**BuiltinTrainer**
    For pre-configured trainers (like TorchTune for LLM fine-tuning) that include
    built-in post-training logic.

Runtimes
~~~~~~~~

Runtimes define the execution environment:

- ``torch-distributed``: PyTorch distributed training
- ``torchtune``: LLM fine-tuning with TorchTune
- Custom runtimes via ClusterTrainingRuntime CRDs
