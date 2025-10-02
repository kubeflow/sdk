# Kubeflow SDK

[![PyPI version](https://img.shields.io/pypi/v/kubeflow?color=%2334D058&label=pypi%20package)](https://pypi.org/project/kubeflow/)
[![PyPI Downloads](https://static.pepy.tech/badge/kubeflow)](https://pepy.tech/projects/kubeflow)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels)
[![Coverage Status](https://coveralls.io/repos/github/kubeflow/sdk/badge.svg?branch=main)](https://coveralls.io/github/kubeflow/sdk?branch=main)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kubeflow/sdk)

## Overview

Kubeflow SDK is a unified Python SDK that streamlines the user experience for AI Practitioners to interact with various
Kubeflow projects. It provides simple, consistent APIs across the Kubeflow ecosystem, enabling users to focus on building
ML applications rather than managing complex infrastrutcure.

### Kubeflow SDK Benefits

- **Unified Experience**: Single SDK to interact with multiple Kubeflow projects through consistent Python APIs
- **Simplified AI Workflows**: Abstract away Kubernetes complexity, allowing AI practitioners to work in familiar Python environments
- **Seamless Integration**: Designed to work together with all Kubeflow projects for end-to-end ML pipelines
- **Local Development**: First-class support for local development requiring only `pip` installation

<div style="text-align: center;">
  <img
    src="https://raw.githubusercontent.com/kubeflow/sdk/main/docs/images/persona_diagram.svg"
    width="600"
    title="Kubeflow SDK Personas"
    alt="Kubeflow SDK Personas"
  />
</div>

## Get Started

### Install Kubeflow SDK

```bash
pip install -U kubeflow
```

### Run your first PyTorch distributed job

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

def get_torch_dist():
    import os
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    print("PyTorch Distributed Environment")
    print(f"WORLD_SIZE: {dist.get_world_size()}")
    print(f"RANK: {dist.get_rank()}")
    print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")

# Create the TrainJob
job_id = TrainerClient().train(
    runtime=TrainerClient().get_runtime("torch-distributed"),
    trainer=CustomTrainer(
        func=get_torch_dist,
        num_nodes=3,
        resources_per_node={
            "cpu": 2,
        },
    ),
)

# Wait for TrainJob to complete
TrainerClient().wait_for_job_status(job_id)

# Print TrainJob logs
print("\n".join(TrainerClient().get_job_logs(name=job_id)))
```

## Supported Kubeflow Projects

| Project                     | Status           | Version Support | Description                                                          |
| --------------------------- | ---------------- | --------------- | -------------------------------------------------------------------- |
| **Kubeflow Trainer**        | ✅ **Available** | v2.0.0+         | Train and fine-tune AI models with various frameworks                |
| **Kubeflow Katib**          | 🚧 Planned       | TBD             | Hyperparameter optimization                                          |
| **Kubeflow Pipelines**      | 🚧 Planned       | TBD             | Build, run, and track AI workflows                                   |
| **Kubeflow Model Registry** | 🚧 Planned       | TBD             | Manage model artifacts, versions and ML artifacts metadata           |
| **Kubeflow Spark Operator** | 🚧 Planned       | TBD             | Manage Spark applications for data processing and feature engineering |

## Community

### Getting Involved

- **Slack**: Join our [#kubeflow-ml-experience](https://www.kubeflow.org/docs/about/community/#kubeflow-slack-channels) Slack channel
- **Meetings**: Attend the [Kubeflow SDK and ML Experience](https://bit.ly/kf-ml-experience) bi-weekly meetings
- **GitHub**: Discussions, issues and contributions at [kubeflow/sdk](https://github.com/kubeflow/sdk)

### Contributing

Kubeflow SDK is a community project and is still under active development. We welcome contributions! Please see our
[CONTRIBUTING Guide](https://github.com/kubeflow/sdk/blob/main/CONTRIBUTING.md) for details.

## Documentation

<!-- TODO(kramaranya): add kubeflow sdk docs -->

- **[Design Document](https://docs.google.com/document/d/1rX7ELAHRb_lvh0Y7BK1HBYAbA0zi9enB0F_358ZC58w/edit)**: Kubeflow SDK design proposal
- **[Component Guides](https://www.kubeflow.org/docs/components/)**: Individual component documentation
- **[DeepWiki](https://deepwiki.com/kubeflow/sdk)**: AI-powered repository documentation

## ✨ Contributors

We couldn't have done it without these incredible people:

<a href="https://github.com/kubeflow/sdk/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kubeflow/sdk" />
</a>
