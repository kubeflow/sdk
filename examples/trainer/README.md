# Kubeflow Trainer Examples

This directory contains examples demonstrating how to use the Kubeflow TrainerClient.

## Examples

### pytorch_distributed_simple.py
Demonstrates running a PyTorch distributed training job locally using `LocalProcessBackend` — no Kubernetes cluster required.
```bash
python examples/trainer/pytorch_distributed_simple.py
```

## Requirements
```bash
pip install kubeflow
```

## Local Development Backends

- **LocalProcessBackend** — Quick prototyping with Python subprocesses, no cluster needed
- **ContainerBackend** — Local development with Docker/Podman isolation
- **KubernetesBackend** — Production training on Kubernetes
