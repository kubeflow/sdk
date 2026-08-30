from __future__ import annotations

import os
import shutil

from kubeflow.trainer import TrainerClient
from kubeflow.trainer.backends.localprocess.backend import (  # for local dev
    LocalProcessBackendConfig,
)
from kubeflow.trainer.constants import constants as trainer_constants
from kubeflow.trainer.types.types import CustomTrainer, TrainJobTemplate

from kubeflow.optimizer import Objective, OptimizerClient, Search, TrialConfig, report_metrics


def train_fn(learning_rate: float, num_epochs: int) -> None:
    """Simple toy training loop.
    In a real project, replace this with your actual training code
    (PyTorch, TensorFlow, scikit-learn, etc.).
    """
    import random

    loss = 1.0
    for epoch in range(num_epochs):
        loss *= 0.9 + 0.05 * random.random()
        print(f"[epoch {epoch}] lr={learning_rate:.4f} loss={loss:.4f}")
        report_metrics({"loss": loss}, epoch=epoch)


def main() -> None:
    if os.name == "nt" and shutil.which("bash") is None:
        print(
            "Detected Windows without `bash` on PATH. The LocalProcessBackend currently requires "
            "`bash` to create a venv and run the training function.\n"
            "Falling back to running the training function in-process."
        )
        train_fn(learning_rate=0.01, num_epochs=5)
        return

    # Choose a backend:
    # - For Kubernetes (default): TrainerClient()
    # - For local development: LocalProcessBackendConfig()
    trainer_client = TrainerClient(backend_config=LocalProcessBackendConfig())
    # Define a reusable TrainJob template
    template = TrainJobTemplate(
        # LocalProcessBackend requires runtime to be set explicitly.
        runtime=trainer_constants.DEFAULT_TRAINING_RUNTIME,
        trainer=CustomTrainer(
            func=train_fn,
            func_args={
                "learning_rate": 0.01,
                "num_epochs": 5,
            },
            # You can also set num_nodes, resources_per_node, etc.
        ),
        # runtime can be left as default, or set explicitly:
        # runtime=trainer_client.get_runtime(name="torch-distributed"),
    )
    # 1. Run a single training job
    job_id = trainer_client.train(**template)
    print(f"Created TrainJob: {job_id}")
    trainjob = trainer_client.wait_for_job_status(
        job_id, status={trainer_constants.TRAINJOB_COMPLETE, trainer_constants.TRAINJOB_FAILED}
    )

    if trainjob.status == trainer_constants.TRAINJOB_FAILED:
        logs = "\n".join(trainer_client.get_job_logs(job_id))
        raise RuntimeError(f"Training job {job_id} failed.\n\nLogs:\n{logs}")

    print(f"Training job {job_id} completed successfully.")

    # 2. (Optional) Run hyperparameter optimization using Katib on Kubernetes.
    #
    # This requires:
    # - a Kubernetes cluster
    # - Katib installed and reachable from your kubeconfig context
    #
    # Run with: RUN_OPTIMIZER=1 python examples/end_to_end_workflow.py
    if os.getenv("RUN_OPTIMIZER", "").strip() in {"1", "true", "True"}:
        optimizer_client = OptimizerClient()
        optimization_id = optimizer_client.optimize(
            trial_template=template,
            trial_config=TrialConfig(
                num_trials=5,
                parallel_trials=2,
            ),
            search_space={
                "learning_rate": Search.loguniform(0.001, 0.1),
                "num_epochs": Search.choice([3, 5, 7]),
            },
            objectives=[Objective(metric="loss", direction="minimize")],
        )
        print(f"Created OptimizationJob: {optimization_id}")
        optimizer_client.wait_for_job_status(optimization_id)
        best = optimizer_client.get_best_results(optimization_id)
        print(f"Best hyperparameters and metrics: {best!r}")
    else:
        print("Skipping optimizer step. Set RUN_OPTIMIZER=1 to run hyperparameter tuning on Kubernetes.")
if __name__ == "__main__":
    main()