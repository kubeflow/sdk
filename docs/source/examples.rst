Examples
========

The following examples demonstrate how to use the Kubeflow SDK for
distributed AI training and LLM fine-tuning.

Trainer Examples
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Example
     - Description
     - Notebook
   * - Local PyTorch Training (MNIST)
     - Train a CNN on MNIST locally using ``LocalProcessBackend`` — no Kubernetes required
     - `Open Notebook <https://github.com/kubeflow/trainer/blob/master/examples/local/local-training-mnist.ipynb>`_
   * - LLaMA 3.2 Fine-Tuning (TorchTune)
     - Fine-tune Llama-3.2-1B-Instruct on the Alpaca dataset using TorchTune ``BuiltinTrainer``
     - `Open Notebook <https://github.com/kubeflow/trainer/blob/master/examples/torchtune/llama3_2/alpaca-trainjob-yaml.ipynb>`_
   * - T5 Text Summarization (DeepSpeed)
     - Fine-tune T5 on CNN/DailyMail for text summarization using DeepSpeed on multi-GPU nodes
     - `Open Notebook <https://github.com/kubeflow/trainer/blob/master/examples/deepspeed/text-summarization/T5-Fine-Tuning.ipynb>`_

For the full list of examples, see the `Kubeflow Trainer examples directory
<https://github.com/kubeflow/trainer/tree/master/examples>`_.
