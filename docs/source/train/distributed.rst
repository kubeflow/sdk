Distributed Training
====================

Scale your training across multiple GPUs and nodes.

Overview
--------

Distributed training lets you:

- **Train faster** by parallelizing across multiple GPUs
- **Train larger models** that don't fit on a single GPU
- **Use more data** by distributing batches across workers

Kubeflow SDK handles the infrastructure - you focus on your model.

Multi-GPU Training
------------------

Request multiple GPUs on a single node:

.. code-block:: python

   from kubeflow.trainer import TrainerClient, CustomTrainer

   def train():
       import torch

       # PyTorch handles multi-GPU automatically with the right runtime
       model = torch.nn.Linear(10, 1)
       if torch.cuda.device_count() > 1:
           model = torch.nn.DataParallel(model)

       # ... training loop ...

   client = TrainerClient()
   client.train(
       trainer=CustomTrainer(
           func=train,
           resources_per_node={"gpu": 4}
       ),
   )

Multi-Node Training
-------------------

Distribute training across multiple machines:

.. code-block:: python

   client.train(
       trainer=CustomTrainer(
           func=train,
           num_nodes=4,  # 4 nodes
           resources_per_node={"gpu": 2},  # 2 GPUs per node = 8 total
       ),
   )

Using PyTorch Distributed
-------------------------

For efficient multi-node training, use PyTorch's DistributedDataParallel:

.. code-block:: python

   def train():
       import os
       import torch
       import torch.distributed as dist
       from torch.nn.parallel import DistributedDataParallel as DDP

       # Kubeflow sets these environment variables automatically
       dist.init_process_group(backend="nccl")
       local_rank = int(os.environ.get("LOCAL_RANK", 0))

       # Create model on correct GPU
       torch.cuda.set_device(local_rank)
       model = torch.nn.Linear(10, 1).cuda(local_rank)
       model = DDP(model, device_ids=[local_rank])

       # ... training loop ...

       dist.destroy_process_group()

   client.train(
       runtime="torch-distributed",
       trainer=CustomTrainer(func=train, num_nodes=2),
   )

Choosing the Right Strategy
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Scenario
     - Approach
     - Configuration
   * - Model fits on 1 GPU, want faster training
     - Multi-GPU DataParallel
     - ``resources_per_node={"gpu": 4}``
   * - Model fits on 1 GPU, huge dataset
     - Multi-node DDP
     - ``num_nodes=4``
   * - Model doesn't fit on 1 GPU
     - Model parallelism (advanced)
     - Custom implementation

Best Practices
--------------

**Scale batch size with workers:**

.. code-block:: python

   def train():
       world_size = int(os.environ.get("WORLD_SIZE", 1))
       batch_size = 32 * world_size  # Scale with number of workers

**Synchronize only when needed:**

Gradient synchronization happens automatically with DDP, but avoid unnecessary
communication in your training loop.

**Use efficient data loading:**

.. code-block:: python

   from torch.utils.data.distributed import DistributedSampler

   sampler = DistributedSampler(dataset)
   loader = DataLoader(dataset, sampler=sampler)
