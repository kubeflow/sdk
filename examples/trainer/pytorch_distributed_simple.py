#!/usr/bin/env python3
"""
TrainerClient Examples - PyTorch Distributed Training

This example demonstrates how to use the Kubeflow TrainerClient
to run distributed PyTorch training jobs locally without a Kubernetes cluster.

Usage:
    # Run directly:
    python examples/trainer/pytorch_distributed_simple.py

    # Or in IPython:
    %run examples/trainer/pytorch_distributed_simple.py
"""

from kubeflow.trainer import TrainerClient, CustomTrainer, TrainJobTemplate
from kubeflow.trainer.backends.localprocess.backend import LocalProcessBackendConfig


def train_fn(learning_rate: str, num_epochs: str):
    """Simple training function to demonstrate distributed training."""
    import os
    lr = float(learning_rate)
    epochs = int(num_epochs)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Starting training on rank {rank}/{world_size}")
    for epoch in range(epochs):
        loss = 1.0 - (lr * 2) - (epoch * 0.01)
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f}")

    print(f"Training complete on rank {rank}")


def example_local_training():
    """
    Run a simple PyTorch distributed training job locally.
    No Kubernetes cluster required.
    """
    print("=" * 60)
    print("LOCAL TRAINING WITH LocalProcessBackend")
    print("=" * 60)

    client = TrainerClient(
        backend_config=LocalProcessBackendConfig()
    )

    template = TrainJobTemplate(
        runtime="torch-distributed",
        trainer=CustomTrainer(
            func=train_fn,
            func_args={"learning_rate": "0.01", "num_epochs": "3"},
            num_nodes=2,
            resources_per_node={"cpu": 1},
        ),
    )

    print("\nStarting TrainJob...")
    job_id = client.train(**template)
    print(f"TrainJob created: {job_id}")

    client.wait_for_job_status(job_id)
    print("\nTrainJob complete! Fetching logs...\n")

    for log in client.get_job_logs(name=job_id):
        print(log)

    print("\nExample complete.")


def main():
    print("=" * 60)
    print("KUBEFLOW TRAINERCLIENT - LOCAL TRAINING EXAMPLES")
    print("=" * 60)

    try:
        example_local_training()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have:")
        print("  1. kubeflow SDK installed: pip install kubeflow")
        print("  2. Python 3.10+")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
