# Kubeflow SDK

[![PyPI version](https://img.shields.io/pypi/v/kubeflow?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kubeflow/)
[![PyPI Downloads](https://static.pepy.tech/badge/kubeflow)](https://pepy.tech/projects/kubeflow)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)
[![Coverage Status](https://coveralls.io/repos/github/kubeflow/sdk/badge.svg?branch=main)](https://coveralls.io/github/kubeflow/sdk?branch=main)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/sdk)

Latest News 🔥

- [2026/01] We have our official documentation here at [Kubeflow SDK Documentation](https://sdk.kubeflow.org/en/latest/)
- [2025/11] Please fill [this survey](https://docs.google.com/forms/d/e/1FAIpQLSet_IAFQzMMDWolzFt5LI9lhzqOOStjIGHxgYqKBnVcRtDfrw/viewform)
  to shape the future of Kubeflow SDK.
- [2025/11] The Kubeflow SDK v0.2 is officially released. Check out
  [the announcement blog post](https://blog.kubeflow.org/sdk/intro/).

## Overview

The Kubeflow SDK is a set of unified Pythonic APIs that let you run any AI workload at any scale –
without the need to learn Kubernetes. It provides simple and consistent APIs across the Kubeflow
ecosystem, enabling users to focus on building AI applications rather than managing complex
infrastructure.

### Kubeflow SDK Benefits

- **Unified Experience**: Single SDK to interact with multiple Kubeflow projects through consistent Python APIs
- **Simplified AI Workloads**: Abstract away Kubernetes complexity and work effortlessly across all
  Kubeflow projects using familiar Python APIs
- **Built for Scale**: Seamlessly scale any AI workload — from local laptop to large-scale production
  cluster with thousands of GPUs using the same APIs.
- **Rapid Iteration**: Reduced friction between development and production environments
- **Local Development**: First-class support for local development without a Kubernetes cluster
  requiring only `pip` installation

<div style="text-align: center;">
  <img
    src="https://raw.githubusercontent.com/kubeflow/sdk/main/docs/images/kubeflow-sdk.drawio.svg"
    title="Kubeflow SDK Diagram"
    alt="Kubeflow SDK Diagram"
  />
</div>

## Kubeflow SDK Introduction

The following KubeCon + CloudNativeCon 2025 talk provides an overview of Kubeflow SDK:

[![Kubeflow SDK](https://img.youtube.com/vi/e_rFwH7CZ_E/0.jpg)](https://youtu.be/e_rFwH7CZ_E)

Additionally, check out these demos to deep dive into Kubeflow SDK capabilities:

- [LLM Fine-Tuning with BuiltinTrainer](https://youtu.be/pRWTLiW3dKQ)
- [Local & Remote Backends in Kubeflow SDK](https://youtu.be/hhaebdkJLLQ)

## Get Started

### Install Kubeflow SDK

```bash
pip install -U kubeflow
```

### Run your first PyTorch distributed job

```python
from kubeflow.trainer import TrainerClient, CustomTrainer, TrainJobTemplate

def get_torch_dist(learning_rate: str, num_epochs: str):
    import os
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    print("PyTorch Distributed Environment")
    print(f"WORLD_SIZE: {dist.get_world_size()}")
    print(f"RANK: {dist.get_rank()}")
    print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")

    lr = float(learning_rate)
    epochs = int(num_epochs)
    loss = 1.0 - (lr * 2) - (epochs * 0.01)

    if dist.get_rank() == 0:
        print(f"loss={loss}")

# Create the TrainJob template
template = TrainJobTemplate(
    runtime="torch-distributed",
    trainer=CustomTrainer(
        func=get_torch_dist,
        func_args={"learning_rate": "0.01", "num_epochs": "5"},
        num_nodes=3,
        resources_per_node={"cpu": 2},
    ),
)

# Create the TrainJob
job_id = TrainerClient().train(**template)

# Wait for TrainJob to complete
TrainerClient().wait_for_job_status(job_id)

# Print TrainJob logs
print("\n".join(TrainerClient().get_job_logs(name=job_id)))
```

### Optimize hyperparameters for your training

```python
from kubeflow.optimizer import OptimizerClient, Search, TrialConfig

# Create OptimizationJob with the same template
optimization_id = OptimizerClient().optimize(
    trial_template=template,
    trial_config=TrialConfig(num_trials=10, parallel_trials=2),
    search_space={
        "learning_rate": Search.loguniform(0.001, 0.1),
        "num_epochs": Search.choice([5, 10, 15]),
    },
)

print(f"OptimizationJob created: {optimization_id}")
```

### Manage models with Model Registry

**Install Model Registry support:**
```bash
pip install 'kubeflow[hub]'
```

To install the Model Registry server, see the [installation guide](https://www.kubeflow.org/docs/components/model-registry/installation/#standalone-installation).

```python
from kubeflow.hub import ModelRegistryClient

client = ModelRegistryClient("https://model-registry.kubeflow.svc.cluster.local", author="Your Name")

# Register a model
model = client.register_model(
    name="my-model",
    uri="s3://bucket/path/to/model",
    version="v1.0.0",
    model_format_name="pytorch",
    model_format_version="2.0",
    version_description="My trained model"
)

# Get a registered model
model = client.get_model("my-model")

# List all models
for model in client.list_models():
    print(f"Model: {model.name}")

# List model versions
for version in client.list_model_versions("my-model"):
    print(f"Version: {version.name}")
```

## Local Development

Kubeflow Trainer client supports local development without needing a Kubernetes cluster.

### Available Backends

- **KubernetesBackend** (default) - Production training on Kubernetes
- **ContainerBackend** - Local development with Docker/Podman isolation
- **LocalProcessBackend** - Quick prototyping with Python subprocesses

**Quick Start:**
Install container support: `pip install kubeflow[docker]` or `pip install kubeflow[podman]`

```python
from kubeflow.trainer import TrainerClient, ContainerBackendConfig, CustomTrainer

# Switch to local container execution
client = TrainerClient(backend_config=ContainerBackendConfig())

# Your training runs locally in isolated containers
job_id = client.train(trainer=CustomTrainer(func=train_fn))
```

## Supported Kubeflow Projects

| Project                     | Status           | Version Support | Description                                                           |
| --------------------------- | ---------------- | --------------- | --------------------------------------------------------------------- |
| **Kubeflow Trainer**        | ✅ **Available** | v2.0.0+         | Train and fine-tune AI models with various frameworks                 |
| **Kubeflow Katib**          | ✅ **Available** | v0.19.0+        | Hyperparameter optimization                                           |
| **Kubeflow Model Registry** | ✅ **Available** | v0.3.0+         | Manage model artifacts, versions and ML artifacts metadata            |
| **Kubeflow Pipelines**      | 🚧 Planned       | TBD             | Build, run, and track AI workflows                                    |
| **Kubeflow Spark Operator** | 🚧 Planned       | TBD             | Manage Spark applications for data processing and feature engineering |
| **Feast**                   | 🚧 Planned       | TBD             | Feature store for machine learning                                    |

## Community

### Getting Involved

- **Slack**: Join our [#kubeflow-ml-experience](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels) Slack channel
- **Meetings**: Attend the [Kubeflow SDK and ML Experience](https://bit.ly/kf-ml-experience) bi-weekly meetings
- **GitHub**: Discussions, issues and contributions at [kubeflow/sdk](https://github.com/kubeflow/sdk)

### Contributing

Kubeflow SDK is a community project and is still under active development. We welcome contributions! Please see our
[CONTRIBUTING Guide](https://github.com/kubeflow/sdk/blob/main/CONTRIBUTING.md) for details.

## Documentation

- **[Documentation](https://sdk.kubeflow.org/en/latest/)**: Kubeflow SDK Official Documentation
- **[Blog Post Announcement](https://blog.kubeflow.org/sdk/intro/)**: Introducing the Kubeflow SDK:
  A Pythonic API to Run AI Workloads at Scale
- **[Design Document](https://docs.google.com/document/d/1rX7ELAHRb_lvh0Y7BK1HBYAbA0zi9enB0F_358ZC58w/edit)**: Kubeflow SDK design proposal
- **[Component Guides](https://www.kubeflow.org/docs/components/)**: Individual component documentation
- **[DeepWiki](https://deepwiki.com/kubeflow/sdk)**: AI-powered repository documentation

## ✨ Contributors

We couldn't have done it without these incredible people:

<a href="https://github.com/kubeflow/sdk/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kubeflow/sdk" />
</a>
