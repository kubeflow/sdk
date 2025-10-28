LLM Fine-Tuning with TorchTune
===============================

This example demonstrates fine-tuning Large Language Models (LLMs) using the
Kubeflow SDK with the BuiltinTrainer and TorchTune runtime.

Overview
--------

This example covers:

- Fine-tuning LLMs with LoRA (Low-Rank Adaptation)
- Using HuggingFace models and datasets
- Distributed fine-tuning across multiple GPUs
- Model checkpointing and export
- Memory-efficient training with quantization

Fine-Tuning Llama 2
-------------------

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

    from kubeflow.trainer import (
        TrainerClient,
        BuiltinTrainer,
        TorchTuneConfig,
        LoraConfig,
        DataType,
        Initializer,
        HuggingFaceDatasetInitializer,
        HuggingFaceModelInitializer,
    )

    # Initialize dataset and model
    initializer = Initializer(
        dataset=HuggingFaceDatasetInitializer(
            storage_uri="yahma/alpaca-cleaned",
        ),
        model=HuggingFaceModelInitializer(
            storage_uri="meta-llama/Llama-2-7b-hf",
            access_token="your-huggingface-token-here",
        ),
    )

    # Configure LoRA fine-tuning
    trainer = BuiltinTrainer(
        config=TorchTuneConfig(
            dtype=DataType.BF16,
            batch_size=2,
            epochs=3,
            num_nodes=2,
            resources_per_node={
                "cpu": 8,
                "memory": "32Gi",
                "gpu": 2,
            },
            peft_config=LoraConfig(
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.1,
            ),
        ),
    )

    # Submit fine-tuning job
    client = TrainerClient()

    job_id = client.train(
        runtime=client.get_runtime("torchtune"),
        initializer=initializer,
        trainer=trainer,
    )

    print(f"Fine-tuning job: {job_id}")

    # Monitor progress
    client.wait_for_job_status(job_id)
    print("\\n".join(client.get_job_logs(job_id)))

Full Configuration Example
---------------------------

Complete fine-tuning with all configuration options:

.. code-block:: python

    from kubeflow.trainer import (
        TrainerClient,
        BuiltinTrainer,
        TorchTuneConfig,
        LoraConfig,
        DataType,
        Loss,
        Initializer,
        HuggingFaceDatasetInitializer,
        HuggingFaceModelInitializer,
        TorchTuneInstructDataset,
        DataFormat,
    )

    # Dataset configuration
    dataset_initializer = HuggingFaceDatasetInitializer(
        storage_uri="yahma/alpaca-cleaned",
        # Optional: specify a split
        split="train",
    )

    # Model configuration
    model_initializer = HuggingFaceModelInitializer(
        storage_uri="meta-llama/Llama-2-7b-hf",
        access_token="hf_your_token_here",
    )

    initializer = Initializer(
        dataset=dataset_initializer,
        model=model_initializer,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        lora_rank=8,           # Rank of LoRA matrices
        lora_alpha=16,         # Scaling factor
        lora_dropout=0.05,     # Dropout for LoRA layers
        # target_modules can be specified if needed
    )

    # TorchTune configuration
    config = TorchTuneConfig(
        # Training hyperparameters
        dtype=DataType.BF16,      # Use bfloat16 for memory efficiency
        batch_size=4,             # Batch size per GPU
        epochs=5,                 # Number of training epochs
        max_steps_per_epoch=500,  # Limit steps per epoch
        gradient_accumulation_steps=4,  # Accumulate gradients

        # Optimizer settings
        optimizer="AdamW",
        optimizer_kwargs={
            "lr": 2e-5,
            "weight_decay": 0.01,
        },

        # Learning rate schedule
        lr_scheduler="cosine",
        warmup_steps=100,

        # Loss function
        loss=Loss.CROSS_ENTROPY,

        # Dataset configuration
        dataset_config=TorchTuneInstructDataset(
            format=DataFormat.JSON,
            prompt_template="alpaca",  # Use Alpaca prompt format
            train_on_input=False,      # Only train on outputs
        ),

        # Distributed training
        num_nodes=4,
        resources_per_node={
            "cpu": 16,
            "memory": "64Gi",
            "gpu": 4,
        },

        # LoRA/PEFT configuration
        peft_config=lora_config,

        # Checkpointing
        save_steps=100,           # Save checkpoint every 100 steps
        checkpoint_dir="/mnt/checkpoints",

        # Logging
        log_steps=10,             # Log metrics every 10 steps
    )

    trainer = BuiltinTrainer(config=config)

    # Submit job
    client = TrainerClient()
    job_id = client.train(
        runtime=client.get_runtime("torchtune"),
        initializer=initializer,
        trainer=trainer,
    )

    print(f"Fine-tuning job submitted: {job_id}")

Fine-Tuning with Custom Dataset
--------------------------------

Using your own dataset from HuggingFace:

.. code-block:: python

    from kubeflow.trainer import (
        TrainerClient,
        BuiltinTrainer,
        TorchTuneConfig,
        LoraConfig,
        DataType,
        Initializer,
        HuggingFaceDatasetInitializer,
        HuggingFaceModelInitializer,
        TorchTuneInstructDataset,
        DataFormat,
    )

    # Your custom dataset
    dataset_initializer = HuggingFaceDatasetInitializer(
        storage_uri="your-username/your-dataset",
        split="train",
        # If private dataset:
        # access_token="your-hf-token"
    )

    model_initializer = HuggingFaceModelInitializer(
        storage_uri="meta-llama/Llama-2-13b-hf",
        access_token="your-hf-token",
    )

    # Custom prompt template
    dataset_config = TorchTuneInstructDataset(
        format=DataFormat.JSON,
        # Custom template with placeholders
        prompt_template=(
            "### Instruction:\\n{instruction}\\n\\n"
            "### Input:\\n{input}\\n\\n"
            "### Response:\\n{output}"
        ),
        train_on_input=False,
    )

    config = TorchTuneConfig(
        dtype=DataType.BF16,
        batch_size=2,
        epochs=3,
        num_nodes=2,
        resources_per_node={
            "cpu": 8,
            "memory": "32Gi",
            "gpu": 2,
        },
        dataset_config=dataset_config,
        peft_config=LoraConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
        ),
    )

    initializer = Initializer(
        dataset=dataset_initializer,
        model=model_initializer,
    )

    trainer = BuiltinTrainer(config=config)

    client = TrainerClient()
    job_id = client.train(
        runtime=client.get_runtime("torchtune"),
        initializer=initializer,
        trainer=trainer,
    )

Memory-Efficient Fine-Tuning
-----------------------------

For limited GPU memory:

.. code-block:: python

    from kubeflow.trainer import (
        TrainerClient,
        BuiltinTrainer,
        TorchTuneConfig,
        LoraConfig,
        DataType,
        Initializer,
        HuggingFaceDatasetInitializer,
        HuggingFaceModelInitializer,
    )

    config = TorchTuneConfig(
        # Use bfloat16 or float16 for memory efficiency
        dtype=DataType.BF16,

        # Smaller batch size
        batch_size=1,

        # Gradient accumulation for effective larger batch
        gradient_accumulation_steps=8,  # Effective batch = 1 * 8 = 8

        # Smaller LoRA rank
        peft_config=LoraConfig(
            lora_rank=4,      # Lower rank = less memory
            lora_alpha=8,
            lora_dropout=0.1,
        ),

        # Enable gradient checkpointing (if supported)
        optimizer_kwargs={
            "lr": 1e-4,
        },

        epochs=3,
        num_nodes=1,
        resources_per_node={
            "cpu": 8,
            "memory": "16Gi",
            "gpu": 1,  # Single GPU
        },
    )

    initializer = Initializer(
        dataset=HuggingFaceDatasetInitializer(
            storage_uri="yahma/alpaca-cleaned",
        ),
        model=HuggingFaceModelInitializer(
            storage_uri="meta-llama/Llama-2-7b-hf",
            access_token="your-token",
        ),
    )

    trainer = BuiltinTrainer(config=config)

    client = TrainerClient()
    job_id = client.train(
        runtime=client.get_runtime("torchtune"),
        initializer=initializer,
        trainer=trainer,
    )

Different Model Sizes
---------------------

Llama 2 7B
~~~~~~~~~~

.. code-block:: python

    model_initializer = HuggingFaceModelInitializer(
        storage_uri="meta-llama/Llama-2-7b-hf",
        access_token="your-token",
    )

    config = TorchTuneConfig(
        dtype=DataType.BF16,
        batch_size=4,
        num_nodes=1,
        resources_per_node={
            "cpu": 8,
            "memory": "32Gi",
            "gpu": 1,
        },
        peft_config=LoraConfig(lora_rank=8),
    )

Llama 2 13B
~~~~~~~~~~~

.. code-block:: python

    model_initializer = HuggingFaceModelInitializer(
        storage_uri="meta-llama/Llama-2-13b-hf",
        access_token="your-token",
    )

    config = TorchTuneConfig(
        dtype=DataType.BF16,
        batch_size=2,
        num_nodes=2,
        resources_per_node={
            "cpu": 16,
            "memory": "64Gi",
            "gpu": 2,
        },
        peft_config=LoraConfig(lora_rank=8),
    )

Llama 2 70B
~~~~~~~~~~~

.. code-block:: python

    model_initializer = HuggingFaceModelInitializer(
        storage_uri="meta-llama/Llama-2-70b-hf",
        access_token="your-token",
    )

    config = TorchTuneConfig(
        dtype=DataType.BF16,
        batch_size=1,
        gradient_accumulation_steps=8,
        num_nodes=8,
        resources_per_node={
            "cpu": 32,
            "memory": "256Gi",
            "gpu": 8,
        },
        peft_config=LoraConfig(lora_rank=16),
    )

Popular Datasets
----------------

Alpaca Dataset
~~~~~~~~~~~~~~

.. code-block:: python

    dataset = HuggingFaceDatasetInitializer(
        storage_uri="yahma/alpaca-cleaned",
    )

Dolly Dataset
~~~~~~~~~~~~~

.. code-block:: python

    dataset = HuggingFaceDatasetInitializer(
        storage_uri="databricks/databricks-dolly-15k",
    )

OpenAssistant Dataset
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset = HuggingFaceDatasetInitializer(
        storage_uri="OpenAssistant/oasst1",
    )

Configuration Tips
------------------

LoRA Rank Selection
~~~~~~~~~~~~~~~~~~~

- **Rank 4-8**: Good for simple tasks, less memory
- **Rank 16-32**: Better for complex tasks, more parameters
- **Rank 64+**: Approaching full fine-tuning quality

Learning Rate Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

- **7B models**: 1e-4 to 5e-5
- **13B models**: 5e-5 to 2e-5
- **70B models**: 2e-5 to 1e-5

Batch Size and Gradient Accumulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Effective batch size = batch_size × gradient_accumulation_steps × num_gpus × num_nodes

Aim for effective batch size of 32-128 for most tasks.

Monitoring Training
-------------------

Stream logs in real-time:

.. code-block:: python

    client = TrainerClient()

    job_id = client.train(
        runtime=client.get_runtime("torchtune"),
        initializer=initializer,
        trainer=trainer,
    )

    # Stream logs as they are produced
    for log_line in client.get_job_logs(job_id, follow=True):
        print(log_line)

Check training status:

.. code-block:: python

    # Get job details
    job = client.get_job(job_id)
    print(f"Status: {job.status}")
    print(f"Started: {job.start_time}")

Saving and Exporting Models
----------------------------

Models are automatically saved to the checkpoint directory specified in
``TorchTuneConfig.checkpoint_dir``. The final model is saved at the end
of training.

To export to HuggingFace Hub after training, you can use a post-training script:

.. code-block:: python

    # After training completes
    def export_model():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load the fine-tuned model
        model = AutoModelForCausalLM.from_pretrained("/mnt/checkpoints/final")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        # Push to HuggingFace Hub
        model.push_to_hub("your-username/llama2-7b-finetuned")
        tokenizer.push_to_hub("your-username/llama2-7b-finetuned")

Troubleshooting
---------------

Out of Memory
~~~~~~~~~~~~~

- Reduce ``batch_size``
- Increase ``gradient_accumulation_steps``
- Use lower ``lora_rank``
- Use ``DataType.FP16`` or ``DataType.BF16``
- Reduce model size (e.g., 7B instead of 13B)

Slow Training
~~~~~~~~~~~~~

- Check GPU utilization
- Increase ``batch_size`` if memory allows
- Verify multi-GPU communication
- Check dataloader ``num_workers``

Poor Results
~~~~~~~~~~~~

- Increase ``lora_rank``
- Train for more ``epochs``
- Adjust learning rate
- Try different ``prompt_template``
- Verify dataset quality

HuggingFace Token Issues
~~~~~~~~~~~~~~~~~~~~~~~~

- Ensure token has read access
- For private models, verify access permissions
- Set token in both model and dataset initializers if needed

Best Practices
--------------

1. **Start Small**: Begin with 7B model and small dataset
2. **Monitor Metrics**: Watch loss curves for convergence
3. **Save Checkpoints**: Set ``save_steps`` appropriately
4. **Use Version Control**: Track configurations
5. **Validate Results**: Test fine-tuned model on held-out data
6. **Resource Planning**: Estimate GPU hours needed

Additional Resources
--------------------

- :doc:`../guides/custom-trainers` - Custom trainer guide
- :doc:`../api-reference/types` - TorchTuneConfig API reference
- `TorchTune Documentation <https://pytorch.org/torchtune/>`_
- `LoRA Paper <https://arxiv.org/abs/2106.09685>`_
- `HuggingFace Model Hub <https://huggingface.co/models>`_
