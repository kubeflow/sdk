Getting Started
===============

This guide will help you get started with the Kubeflow SDK.

Your First Training Job
-----------------------

Here's a simple example using PyTorch distributed training:

.. code-block:: python

    from kubeflow.trainer import TrainerClient, CustomTrainer

    def train():
        import torch.distributed as dist

        dist.init_process_group(backend="gloo")
        print(f"Training on rank {dist.get_rank()}")

    # Create and run the training job
    client = TrainerClient()
    job_id = client.train(
        runtime=client.get_runtime("torch-distributed"),
        trainer=CustomTrainer(
            func=train,
            num_nodes=2,
            resources_per_node={"cpu": 2},
        ),
    )

    # Wait for completion
    client.wait_for_job_status(job_id)

    # View logs
    print("\n".join(client.get_job_logs(job_id)))

Next Steps
----------

- :doc:`api-reference/index` - Complete API reference
- User guides and examples coming soon!
