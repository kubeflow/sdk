# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ContainerBackend for local hyperparameter optimization.

This backend mimics KubernetesBackend behavior but runs locally using Docker/Podman
containers instead of Kubernetes resources. It stores job state in JSON files
(mimicking Katib Experiment and Trial CRs) and delegates trial execution to
TrainerClient's ContainerBackend.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import logging
import os
from pathlib import Path
import random
import re
import string
import threading
import time
from typing import Any, Optional
import uuid

from kubeflow.optimizer.backends.base import RuntimeBackend
from kubeflow.optimizer.backends.container import storage
from kubeflow.optimizer.backends.container.types import ContainerBackendConfig
from kubeflow.optimizer.constants import constants
from kubeflow.optimizer.types.algorithm_types import BaseAlgorithm, RandomSearch
from kubeflow.optimizer.types.optimization_types import (
    Metric,
    Objective,
    OptimizationJob,
    Result,
    Trial,
    TrialConfig,
)
from kubeflow.trainer import constants as trainer_constants
from kubeflow.trainer.backends.container.backend import (
    ContainerBackend as TrainerContainerBackend,
)
from kubeflow.trainer.backends.container.types import (
    ContainerBackendConfig as TrainerContainerBackendConfig,
)
from kubeflow.trainer.types.types import TrainJobTemplate

logger = logging.getLogger(__name__)


def _sample_hyperparameters_with_optuna(
    search_space: dict[str, Any],
    objective_metric: str,
    objective_direction: str,
    study_storage_path: str,
    job_name: str,
) -> dict[str, Any]:
    """Sample hyperparameters using Optuna optimization.
    
    Args:
        search_space: Dictionary mapping parameter names to Search specifications.
        objective_metric: Name of the metric to optimize.
        objective_direction: Direction of optimization ("maximize" or "minimize").
        study_storage_path: Path to store Optuna study database.
        job_name: Name of the optimization job (used as study name).
        
    Returns:
        Dictionary of sampled hyperparameter values.
    """
    try:
        import optuna
        from optuna.storages import RDBStorage
    except ImportError as err:
        # Re-raise with clearer message while preserving the original exception
        raise ImportError(
            "Optuna is required for hyperparameter optimization. "
            "Install with: pip install optuna"
        ) from err
    
    import os
    import time
    
    # Create study directory if it doesn't exist
    study_dir = os.path.join(study_storage_path, job_name)
    os.makedirs(study_dir, exist_ok=True)
    
    # Create or load Optuna study with retry logic for race conditions
    study_db = f"sqlite:///{study_dir}/optuna.db"
    direction = "maximize" if objective_direction == "maximize" else "minimize"
    
    # Use RDBStorage with enable_shared_cache=False to avoid SQLite lock issues
    storage = RDBStorage(
        url=study_db,
        engine_kwargs={"connect_args": {"timeout": 30}},  # Increase timeout
    )
    
    # Retry logic for study creation in parallel scenarios
    max_retries = 5
    for attempt in range(max_retries):
        try:
            study = optuna.create_study(
                study_name=job_name,
                storage=storage,
                direction=direction,
                load_if_exists=True,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait with exponential backoff
                time.sleep(0.1 * (2 ** attempt))
            else:
                # Preserve the original exception as the cause so stack traces show it.
                raise RuntimeError(
                    f"Failed to create/load Optuna study after {max_retries} attempts"
                ) from e
    
    # Create trial and sample hyperparameters
    trial = study.ask()
    
    samples = {}
    for param_name, param_spec in search_space.items():
        param_type = param_spec.parameter_type
        feasible_space = param_spec.feasible_space
        
        if param_type == "categorical":
            # Categorical parameter
            choices = feasible_space.list
            # Convert string values back to appropriate types
            typed_choices = []
            for choice in choices:
                try:
                    # Try to convert to int first
                    typed_choices.append(int(choice))
                except ValueError:
                    try:
                        # Try to convert to float
                        typed_choices.append(float(choice))
                    except ValueError:
                        # Keep as string
                        typed_choices.append(choice)
            
            samples[param_name] = trial.suggest_categorical(param_name, typed_choices)
            
        elif param_type == "double":
            # Continuous parameter
            min_val = float(feasible_space.min)
            max_val = float(feasible_space.max)
            
            if feasible_space.distribution == "logUniform":
                samples[param_name] = trial.suggest_float(
                    param_name, min_val, max_val, log=True
                )
            else:
                samples[param_name] = trial.suggest_float(
                    param_name, min_val, max_val, log=False
                )
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    
    # Store trial object for later reporting results
    samples["_optuna_trial"] = trial
    
    return samples


def _report_trial_result_to_optuna(
    study_storage_path: str,
    job_name: str,
    trial_number: int,
    metric_value: Optional[float],
    state: str = "COMPLETE",
):
    """Report trial results back to Optuna study.
    
    Args:
        study_storage_path: Path to Optuna study database.
        job_name: Name of the optimization job.
        trial_number: Trial number in Optuna study.
        metric_value: Objective metric value (None if trial failed).
        state: Trial state ("COMPLETE" or "FAIL").
    """
    try:
        import optuna
    except ImportError:
        return
    
    study_db = f"sqlite:///{study_storage_path}/{job_name}/optuna.db"
    study = optuna.load_study(study_name=job_name, storage=study_db)
    
    if state == "COMPLETE" and metric_value is not None:
        study.tell(trial_number, metric_value)
    else:
        # Mark trial as failed
        study.tell(trial_number, state=optuna.trial.TrialState.FAIL)


class ContainerBackend(RuntimeBackend):
    """Container backend for local hyperparameter optimization.

    Mimics KubernetesBackend but uses local JSON storage and Docker/Podman containers.
    """

    def __init__(self, cfg: ContainerBackendConfig):
        """Initialize the container backend.

        Args:
            cfg: Configuration for the container backend.

        Raises:
            RuntimeError: If neither Docker nor Podman are available.
        """
        self.cfg = cfg
        logger.debug(f"Initializing ContainerBackend with config: {cfg}")

        # Expand storage path (handle ~ for home directory).
        self.storage_path = os.path.expanduser(cfg.storage_path)
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

        # Initialize trainer's container backend for executing trials.
        trainer_config = TrainerContainerBackendConfig(
            pull_policy=cfg.pull_policy,
            auto_remove=cfg.auto_remove,
            container_host=cfg.container_host,
            container_runtime=cfg.container_runtime,
        )
        self.trainer_backend = TrainerContainerBackend(trainer_config)
        self.logger = logger

    def optimize(
        self,
        trial_template: TrainJobTemplate,
        *,
        search_space: dict[str, Any],
        trial_config: Optional[TrialConfig] = None,
        objectives: Optional[list[Objective]] = None,
        algorithm: Optional[BaseAlgorithm] = None,
    ) -> str:
        """Create an OptimizationJob for hyperparameter tuning.

        Args:
            trial_template: The TrainJob template defining the training script.
            search_space: Dictionary mapping parameter names to Search specifications.
            trial_config: Optional configuration for running trials.
            objectives: List of objectives to optimize. Defaults to single objective.
            algorithm: The optimization algorithm to use. Defaults to RandomSearch.

        Returns:
            The unique name of the optimization job.

        Raises:
            ValueError: If search_space is empty or invalid.
            RuntimeError: If trial execution fails critically.
        """
        # Generate unique name for the OptimizationJob (matches K8s backend).
        optimization_job_name = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11]
        
        logger.info(f"Starting optimization job: {optimization_job_name}")
        
        # Validate search_space.
        if not search_space:
            raise ValueError("Search space must be set.")
        
        # Set defaults.
        objectives = objectives or [Objective()]
        algorithm = algorithm or RandomSearch()
        trial_config = trial_config or TrialConfig()
        
        # Build parameters_spec (matches K8s backend logic).
        parameters_spec = []
        if trial_template.trainer.func_args is None:
            trial_template.trainer.func_args = {}
        
        for param_name, param_spec in search_space.items():
            param_spec.name = param_name
            parameters_spec.append(param_spec)
        
        # Precompute algorithm/runtime/trainer/initializer specs to avoid long
        # inline conditionals inside the experiment data dict and to keep
        # line lengths under ruff's E501 limit.
        if hasattr(algorithm, "_to_katib_spec"):
            _algorithm_spec = algorithm._to_katib_spec()
        else:
            _algorithm_spec = {"algorithmName": algorithm.__class__.__name__}

        _runtime_spec = (
            trial_template.runtime.to_dict()
            if hasattr(trial_template.runtime, "to_dict")
            else trial_template.runtime
        )

        _trainer_spec = (
            trial_template.trainer.to_dict()
            if hasattr(trial_template.trainer, "to_dict")
            else trial_template.trainer
        )

        _initializer_spec = (
            trial_template.initializer.to_dict()
            if trial_template.initializer and hasattr(trial_template.initializer, "to_dict")
            else trial_template.initializer
        )

        # Create experiment storage (mimics Katib Experiment CR).
        experiment_data = {
            "metadata": {
                "name": optimization_job_name,
                "creationTimestamp": storage._get_current_timestamp(),
            },
            "spec": {
                "parameters": parameters_spec,
                "objective": {
                    "objectiveMetricName": objectives[0].metric,
                    "type": objectives[0].direction.value,
                    "additionalMetricNames": [obj.metric for obj in objectives[1:]]
                    if len(objectives) > 1
                    else None,
                },
                "algorithm": _algorithm_spec,
                "trialTemplate": {
                    "trialSpec": {
                        # Build runtime/trainer/initializer specs separately to keep
                        # lines under 100 characters.
                        "runtime": _runtime_spec,
                        "trainer": _trainer_spec,
                        "initializer": _initializer_spec,
                    }
                },
                "maxTrialCount": trial_config.num_trials,
                "parallelTrialCount": trial_config.parallel_trials,
                "maxFailedTrialCount": trial_config.max_failed_trials,
            },
            "status": {
                "conditions": [],
                "currentOptimalTrial": None,
                "trials": 0,
                "trialsSucceeded": 0,
                "trialsFailed": 0,
            },
        }
        
        try:
            storage.create_experiment_storage(
                self.storage_path,
                optimization_job_name,
                experiment_data,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {constants.OPTIMIZATION_JOB_KIND}: {optimization_job_name}"
            ) from e
        
        # Run optimization trials (supports both sequential and parallel execution).
        try:
            self._run_optimization_loop(
                job_name=optimization_job_name,
                trial_template=trial_template,
                search_space=search_space,
                algorithm=algorithm,
                objectives=objectives,
                trial_config=trial_config,
            )
        except Exception as e:
            logger.error(f"Optimization loop failed for {optimization_job_name}: {e}")
            try:
                storage.update_experiment_status(
                    self.storage_path,
                    optimization_job_name,
                    {
                        "conditions": [{
                            "type": constants.OPTIMIZATION_JOB_FAILED,
                            "status": "True",
                            "reason": "OptimizationFailed",
                            "message": str(e),
                        }]
                    }
                )
            except Exception as status_error:
                logger.warning(f"Failed to update status: {status_error}")
            raise RuntimeError(
                f"Failed to run optimization for {constants.OPTIMIZATION_JOB_KIND}: {optimization_job_name}"
            ) from e
        
        logger.debug(
            f"{constants.OPTIMIZATION_JOB_KIND} {optimization_job_name} has been created"
        )
        
        return optimization_job_name

    def _run_optimization_loop(
        self,
        job_name: str,
        trial_template: TrainJobTemplate,
        search_space: dict[str, Any],
        algorithm: BaseAlgorithm,
        objectives: list[Objective],
        trial_config: TrialConfig,
    ):
        """Run the optimization loop with parallel trial execution support."""
        max_parallel = trial_config.parallel_trials or 1
        logger.info(
            f"Starting optimization loop for {job_name}: "
            f"{trial_config.num_trials} trials, "
            f"parallel_trials: {max_parallel}, "
            f"objective: {objectives[0].metric} ({objectives[0].direction.value})"
        )
        
        # Thread-safe counters.
        succeeded_trials = 0
        failed_trials = 0
        counters_lock = threading.Lock()
        should_stop = threading.Event()
        
        def _execute_single_trial(trial_num: int) -> tuple[int, bool]:
            """Execute a single trial. Returns (trial_num, success)."""
            nonlocal succeeded_trials, failed_trials
            
            if should_stop.is_set():
                return (trial_num, False)
            
            trial_name = f"{job_name}-trial-{trial_num:03d}"
            
            logger.info(f"Starting trial {trial_num + 1}/{trial_config.num_trials}: {trial_name}")
            
            # Initialize trial number (in case of early failure)
            optuna_trial_number = trial_num
            
            try:
                # Sample hyperparameters using Optuna.
                suggested_params = _sample_hyperparameters_with_optuna(
                    search_space=search_space,
                    objective_metric=objectives[0].metric,
                    objective_direction=objectives[0].direction.value,
                    study_storage_path=self.storage_path,
                    job_name=job_name,
                )
                
                # Extract Optuna trial object and remove from params
                optuna_trial = suggested_params.pop("_optuna_trial", None)
                optuna_trial_number = optuna_trial.number if optuna_trial else trial_num
                
                logger.debug(f"Suggested parameters for {trial_name}: {suggested_params}")
                
                # Create trial storage (mimics Katib Trial CR).
                trial_data = {
                    "metadata": {
                        "name": trial_name,
                        "creationTimestamp": storage._get_current_timestamp(),
                    },
                    "spec": {
                        "parameterAssignments": [
                            {"name": name, "value": str(value)}
                            for name, value in suggested_params.items()
                        ]
                    },
                    "status": {
                        "conditions": [],
                        "observation": {"metrics": []},
                    },
                }
                storage.create_trial(self.storage_path, job_name, trial_name, trial_data)
                
                # Run trial.
                metrics = self._run_trial(
                    trial_name=trial_name,
                    trial_template=trial_template,
                    parameters=suggested_params,
                )
                
                # Update trial with metrics.
                if metrics:
                    storage.update_trial_status(
                        self.storage_path,
                        job_name,
                        trial_name,
                        {
                            "observation": {
                                "metrics": [
                                    {"name": name, "value": value, "latest": value, "max": value, "min": value}
                                    for name, value in metrics.items()
                                ]
                            },
                            "conditions": [{
                                "type": "Succeeded",
                                "status": "True",
                                "reason": "TrialSucceeded",
                            }]
                        }
                    )
                    with counters_lock:
                        succeeded_trials += 1
                    logger.info(f"Trial {trial_name} succeeded with metrics: {metrics}")
                else:
                    logger.warning(f"Trial {trial_name} completed but no metrics found")
                    with counters_lock:
                        failed_trials += 1
                
                # Update best trial if this is better.
                self._update_best_trial_if_needed(
                    job_name=job_name,
                    trial_name=trial_name,
                    parameters=suggested_params,
                    metrics=metrics,
                    objective=objectives[0],
                )
                
                # Report results back to Optuna for adaptive sampling.
                if metrics and objectives[0].metric in metrics:
                    _report_trial_result_to_optuna(
                        study_storage_path=self.storage_path,
                        job_name=job_name,
                        trial_number=optuna_trial_number,
                        metric_value=metrics[objectives[0].metric],
                        state="COMPLETE",
                    )
                else:
                    # No metric found, mark as failed
                    _report_trial_result_to_optuna(
                        study_storage_path=self.storage_path,
                        job_name=job_name,
                        trial_number=optuna_trial_number,
                        metric_value=None,
                        state="FAIL",
                    )
                
                return (trial_num, True)
                
            except Exception as e:
                logger.error(f"Trial {trial_name} failed: {e}")
                with counters_lock:
                    failed_trials += 1
                    
                    # Check if we should stop due to max failed trials.
                    if (
                        trial_config.max_failed_trials
                        and failed_trials >= trial_config.max_failed_trials
                    ):
                        logger.warning(
                            f"Stopping optimization: reached max failed trials "
                            f"({failed_trials}/{trial_config.max_failed_trials})"
                        )
                        should_stop.set()
                
                # Report failed trial to Optuna
                _report_trial_result_to_optuna(
                    study_storage_path=self.storage_path,
                    job_name=job_name,
                    trial_number=optuna_trial_number,
                    metric_value=None,
                    state="FAIL",
                )
                
                # Update trial status to failed.
                try:
                    storage.update_trial_status(
                        self.storage_path,
                        job_name,
                        trial_name,
                        {
                            "conditions": [{
                                "type": "Failed",
                                "status": "True",
                                "reason": "TrialFailed",
                                "message": str(e),
                            }]
                        }
                    )
                except Exception as status_error:
                    logger.warning(f"Failed to update trial status: {status_error}")
                
                return (trial_num, False)
        
        # Execute trials with ThreadPoolExecutor.
        if max_parallel > 1:
            logger.info(f"Running trials in parallel with max_workers={max_parallel}")
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {
                    executor.submit(_execute_single_trial, trial_num): trial_num
                    for trial_num in range(trial_config.num_trials)
                }
                
                for future in as_completed(futures):
                    trial_num, success = future.result()
                    if should_stop.is_set():
                        for f in futures:
                            f.cancel()
                        break
        else:
            # Sequential execution.
            logger.info("Running trials sequentially")
            for trial_num in range(trial_config.num_trials):
                if should_stop.is_set():
                    break
                _execute_single_trial(trial_num)
        
        # Update final experiment status.
        final_status = {
            "trials": trial_config.num_trials,
            "trialsSucceeded": succeeded_trials,
            "trialsFailed": failed_trials,
            "conditions": []
        }
        
        if succeeded_trials > 0:
            final_status["conditions"].append({
                "type": constants.EXPERIMENT_SUCCEEDED,
                "status": "True",
                "reason": "ExperimentSucceeded",
                "message": f"Completed {succeeded_trials}/{trial_config.num_trials} trials successfully",
            })
        else:
            final_status["conditions"].append({
                "type": constants.OPTIMIZATION_JOB_FAILED,
                "status": "True",
                "reason": "AllTrialsFailed",
                "message": f"All {trial_config.num_trials} trials failed",
            })
        
        storage.update_experiment_status(self.storage_path, job_name, final_status)
        logger.info(
            f"Optimization loop completed for {job_name}: "
            f"{succeeded_trials} succeeded, {failed_trials} failed"
        )

    def _run_trial(
        self,
        trial_name: str,
        trial_template: TrainJobTemplate,
        parameters: dict[str, Any],
    ) -> Optional[dict[str, float]]:
        """Execute a single trial with the given hyperparameters."""
        logger.debug(f"Running trial {trial_name} with parameters: {parameters}")
        
        # Deep copy template to avoid mutation.
        trial_job_template = copy.deepcopy(trial_template)
        
        # Substitute hyperparameters into func_args.
        if trial_job_template.trainer.func_args is None:
            trial_job_template.trainer.func_args = {}
        
        for param_name, param_value in parameters.items():
            trial_job_template.trainer.func_args[param_name] = param_value
        
        try:
            # Create and run training job using trainer backend.
            # The train() method returns the job name
            def _set_trial_name(job_spec, trainer, backend):
                job_spec.setdefault("metadata", {})["name"] = trial_name

            actual_job_name = self.trainer_backend.train(
                trainer=trial_job_template.trainer,
                runtime=trial_job_template.runtime if hasattr(trial_job_template, 'runtime') else None,
                initializer=trial_job_template.initializer if hasattr(trial_job_template, 'initializer') else None,
                options=[_set_trial_name],
            )
            
            # Wait for training job to complete.
            self.trainer_backend.wait_for_job_status(
                name=actual_job_name,
                status={trainer_constants.TRAINJOB_COMPLETE},
                timeout=3600,
            )
            
            # Extract metrics from training logs.
            metrics = self._extract_metrics_from_trial(actual_job_name)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to run trial {trial_name}: {e}")
            raise

    def _extract_metrics_from_trial(self, trial_name: str) -> dict[str, float]:
        """Extract metrics from a completed trial."""
        metrics = {}
        
        try:
            log_lines = list(self.trainer_backend.get_job_logs(name=trial_name))
            
            # Parse metrics from logs using common patterns.
            metric_patterns = [
                r"(\w+):\s*([+-]?\d+(?:\.\d+)?)",  # "accuracy: 0.95"
                r"(\w+)\s*=\s*([+-]?\d+(?:\.\d+)?)",  # "loss = 0.123"
            ]
            
            for line in log_lines:
                for pattern in metric_patterns:
                    matches = re.findall(pattern, line)
                    for metric_name, metric_value in matches:
                            try:
                                metrics[metric_name] = float(metric_value)
                                logger.debug(
                                    "Extracted metric %s=%s from %s",
                                    metric_name,
                                    metric_value,
                                    trial_name,
                                )
                            except ValueError:
                                continue
            
            if not metrics:
                logger.warning(f"No metrics found in logs for trial {trial_name}")
            
        except Exception as e:
            logger.error(f"Failed to extract metrics from trial {trial_name}: {e}")
        
        return metrics

    def _update_best_trial_if_needed(
        self,
        job_name: str,
        trial_name: str,
        parameters: dict[str, Any],
        metrics: Optional[dict[str, float]],
        objective: Objective,
    ):
        """Update the current optimal trial if this trial is better."""
        if not metrics or objective.metric not in metrics:
            logger.debug(f"No objective metric '{objective.metric}' found for {trial_name}")
            return
        
        current_value = metrics[objective.metric]
        
        try:
            experiment = storage.load_experiment(self.storage_path, job_name)
            current_best = experiment.get("status", {}).get("currentOptimalTrial")
            
            should_update = False
            
            if current_best is None:
                should_update = True
                logger.info(f"Setting {trial_name} as initial best trial")
            else:
                best_observation = current_best.get("observation", {})
                best_metrics = best_observation.get("metrics", [])
                
                best_value = None
                for m in best_metrics:
                    if m.get("name") == objective.metric:
                        best_value = m.get("value") or m.get("latest")
                        break
                
                if best_value is None:
                    should_update = True
                    logger.info(f"Current best has no valid metric, updating to {trial_name}")
                else:
                    from kubeflow.optimizer.types.optimization_types import Direction
                    
                    if objective.direction == Direction.MAXIMIZE:
                        should_update = current_value > best_value
                    else:
                        should_update = current_value < best_value
                    
                    if should_update:
                        logger.info(
                            f"New best trial: {trial_name} "
                            f"({objective.metric}={current_value} vs {best_value})"
                        )

                def _metric_entry(name, value):
                    return {
                        "name": name,
                        "value": value,
                        "latest": value,
                        "max": value,
                        "min": value,
                    }
            
            if should_update:
                optimal_trial_data = {
                    "bestTrialName": trial_name,
                    "parameterAssignments": [
                        {"name": name, "value": str(value)}
                        for name, value in parameters.items()
                    ],
                    "observation": {
                        "metrics": [
                            _metric_entry(name, value)
                            for name, value in metrics.items()
                        ]
                    },
                }
                
                storage.update_experiment_status(
                    self.storage_path,
                    job_name,
                    {"currentOptimalTrial": optimal_trial_data}
                )
        
        except Exception as e:
            logger.error(f"Failed to update best trial: {e}")

    def list_jobs(self) -> list[OptimizationJob]:
        """List of the created OptimizationJobs"""
        result = []
        
        try:
            experiment_names = storage.list_experiments(self.storage_path)
            
            if not experiment_names:
                return result
            
            for job_name in experiment_names:
                try:
                    experiment_data = self.__get_experiment_from_storage(job_name)
                    optimization_job = self.__get_optimization_job_from_storage(experiment_data)
                    result.append(optimization_job)
                except Exception as e:
                    logger.warning(f"Failed to load job {job_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise RuntimeError("Failed to list optimization jobs") from e
        
        return result

    def get_job(self, name: str) -> OptimizationJob:
        """Get the OptimizationJob object"""
        experiment_data = self.__get_experiment_from_storage(name)
        return self.__get_optimization_job_from_storage(experiment_data)

    def get_job_logs(
        self,
        name: str,
        trial_name: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get the OptimizationJob logs from a Trial"""
        # Determine what trial to get logs from.
        if trial_name is None:
            best_trial = self._get_best_trial(name)
            if best_trial is None:
                optimization_job = self.get_job(name)
                if not optimization_job.trials:
                    return
                trial_name = optimization_job.trials[0].name
            else:
                trial_name = best_trial.name
            logger.debug(f"Getting logs from trial: {trial_name}")
        
        # Get TrainJob and delegate to trainer backend.
        try:
            trainjob = self.trainer_backend.get_job(trial_name)
            
            step = trainer_constants.NODE + "-0"
            for c in trainjob.steps:
                if c.name == step:
                    yield from self.trainer_backend.get_job_logs(
                        name=trial_name,
                        step=step,
                        follow=follow
                    )
                    break
        except Exception as e:
            logger.error(f"Failed to get logs for trial {trial_name}: {e}")
            raise RuntimeError(f"Failed to get logs for trial {trial_name}") from e

    def get_best_results(self, name: str) -> Optional[Result]:
        """Get the best hyperparameters and metrics from an OptimizationJob"""
        best_trial = self._get_best_trial(name)
        
        if best_trial is None:
            return None
        
        return Result(
            parameters=best_trial.parameters,
            metrics=best_trial.metrics,
        )

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.OPTIMIZATION_JOB_COMPLETE},
        timeout: int = 3600,
        polling_interval: int = 2,
    ) -> OptimizationJob:
        """Wait for an OptimizationJob to reach a desired status"""
        job_statuses = {
            constants.OPTIMIZATION_JOB_CREATED,
            constants.OPTIMIZATION_JOB_RUNNING,
            constants.OPTIMIZATION_JOB_COMPLETE,
            constants.OPTIMIZATION_JOB_FAILED,
        }
        if not status.issubset(job_statuses):
            raise ValueError(
                f"Expected status {status} must be a subset of {job_statuses}"
            )
        
        if polling_interval > timeout:
            raise ValueError(
                f"Polling interval {polling_interval} must be less than timeout: {timeout}"
            )
        
        for _ in range(round(timeout / polling_interval)):
            optimization_job = self.get_job(name)
            logger.debug(
                f"{constants.OPTIMIZATION_JOB_KIND} {name}, status {optimization_job.status}"
            )
            
            if (
                constants.OPTIMIZATION_JOB_FAILED not in status
                and optimization_job.status == constants.OPTIMIZATION_JOB_FAILED
            ):
                raise RuntimeError(f"{constants.OPTIMIZATION_JOB_KIND} {name} is Failed")
            
            if optimization_job.status in status:
                return optimization_job
            
            time.sleep(polling_interval)
        
        raise TimeoutError(
            f"Timeout waiting for {constants.OPTIMIZATION_JOB_KIND} {name} to reach status: "
            f"{status}"
        )

    def delete_job(self, name: str):
        """Delete the OptimizationJob"""
        try:
            trials = self.__get_trials_from_storage(name)
        except Exception as e:
            logger.error(f"Failed to get trials for job {name}: {e}")
            raise ValueError(f"Job '{name}' doesn't exist or cannot be accessed") from e
        
        for trial in trials:
            try:
                self.trainer_backend.delete_job(trial.name)
                logger.debug(f"Deleted trial {trial.name}")
            except Exception as e:
                logger.warning(f"Failed to delete trial {trial.name}: {e}")
        
        try:
            storage.delete_job_storage(self.storage_path, name)
        except Exception as e:
            logger.error(f"Failed to delete storage for job {name}: {e}")
            raise RuntimeError(f"Failed to delete job storage for {name}") from e
        
        logger.debug(f"{constants.OPTIMIZATION_JOB_KIND} {name} has been deleted")

    # ============================================================================
    # Helper Methods (Matching K8s Backend Structure)
    # ============================================================================

    def __get_experiment_from_storage(self, name: str) -> dict:
        """Load experiment data from storage (replaces K8s __get_experiment_cr)"""
        try:
            return storage.load_experiment(self.storage_path, name)
        except ValueError as e:
            raise ValueError(
                f"Failed to get {constants.OPTIMIZATION_JOB_KIND}: {name}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {constants.OPTIMIZATION_JOB_KIND}: {name}"
            ) from e

    def __get_optimization_job_from_storage(
        self, experiment_data: dict
    ) -> OptimizationJob:
        """Build OptimizationJob from storage data (replaces K8s __get_optimization_job_from_cr)"""
        if not (
            experiment_data.get("metadata")
            and experiment_data["metadata"].get("name")
            and experiment_data.get("spec")
            and experiment_data["spec"].get("parameters")
            and experiment_data["spec"].get("objective")
            and experiment_data["spec"].get("algorithm")
            and experiment_data["spec"].get("maxTrialCount")
            and experiment_data["spec"].get("parallelTrialCount")
            and experiment_data["metadata"].get("creationTimestamp")
        ):
            raise ValueError(
                f"{constants.OPTIMIZATION_JOB_KIND} data is invalid: {experiment_data}"
            )

        name = experiment_data["metadata"]["name"]
        spec = experiment_data["spec"]
        
        optimization_job = OptimizationJob(
            name=name,
            search_space=spec.get("parameters", {}),
            objectives=self._parse_objectives(spec["objective"]),
            algorithm=self._parse_algorithm(spec["algorithm"]),
            trial_config=TrialConfig(
                num_trials=spec["maxTrialCount"],
                parallel_trials=spec["parallelTrialCount"],
                max_failed_trials=spec.get("maxFailedTrialCount"),
            ),
            trials=self.__get_trials_from_storage(name),
            creation_timestamp=experiment_data["metadata"]["creationTimestamp"],
            status=constants.OPTIMIZATION_JOB_CREATED,
        )

        if experiment_data.get("status") and experiment_data["status"].get("conditions"):
            for condition in experiment_data["status"]["conditions"]:
                cond_type = condition.get("type")
                cond_status = condition.get("status")
                if cond_type == constants.EXPERIMENT_SUCCEEDED and cond_status == "True":
                    optimization_job.status = constants.OPTIMIZATION_JOB_COMPLETE
                elif cond_type == constants.OPTIMIZATION_JOB_FAILED and cond_status == "True":
                    optimization_job.status = constants.OPTIMIZATION_JOB_FAILED
                else:
                    for trial in optimization_job.trials:
                        if trial.trainjob.status == trainer_constants.TRAINJOB_RUNNING:
                            optimization_job.status = constants.OPTIMIZATION_JOB_RUNNING
                            break

        return optimization_job

    def __get_trials_from_storage(self, job_name: str) -> list[Trial]:
        """Load trials from storage (replaces K8s __get_trials_from_job)"""
        result = []
        
        try:
            trial_names = storage.list_trials(self.storage_path, job_name)
            
            if not trial_names:
                return result
            
            for trial_name in trial_names:
                try:
                    trial_data = storage.load_trial(self.storage_path, job_name, trial_name)
                    
                    if not (
                        trial_data.get("metadata")
                        and trial_data["metadata"].get("name")
                        and trial_data.get("spec")
                        and trial_data["spec"].get("parameterAssignments")
                    ):
                        logger.warning(
                            f"{constants.TRIAL_KIND} data is invalid for {trial_name}, skipping"
                        )
                        continue
                    
                    parameters = {
                        pa["name"]: pa["value"]
                        for pa in trial_data["spec"]["parameterAssignments"]
                        if pa.get("name") and pa.get("value") is not None
                    }
                    
                    try:
                        trainjob = self.trainer_backend.get_job(name=trial_name)
                    except Exception as e:
                        logger.warning(f"Failed to get TrainJob for trial {trial_name}: {e}")
                        trainjob = None
                    
                    trial = Trial(
                        name=trial_name,
                        parameters=parameters,
                        trainjob=trainjob,
                    )
                    
                    if (
                        trial_data.get("status")
                        and trial_data["status"].get("observation")
                        and trial_data["status"]["observation"].get("metrics")
                    ):
                        trial.metrics = [
                            Metric(
                                name=m["name"],
                                latest=float(m.get("latest", m.get("value", 0))),
                                max=float(m.get("max", m.get("value", 0))),
                                min=float(m.get("min", m.get("value", 0))),
                            )
                            for m in trial_data["status"]["observation"]["metrics"]
                            if m.get("name")
                        ]
                    
                    result.append(trial)
                    
                except Exception as e:
                    logger.warning(f"Failed to load trial {trial_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to list trials for job {job_name}: {e}")
            raise RuntimeError(
                f"Failed to list {constants.TRIAL_KIND}s for job: {job_name}"
            ) from e
        
        return result

    def _get_best_trial(self, name: str) -> Optional[Trial]:
        """Get best trial from storage (matches K8s _get_best_trial)"""
        experiment_data = self.__get_experiment_from_storage(name)
        
        if (
            experiment_data.get("status")
            and experiment_data["status"].get("currentOptimalTrial")
            and experiment_data["status"]["currentOptimalTrial"].get("bestTrialName")
        ):
            best_trial_name = experiment_data["status"]["currentOptimalTrial"]["bestTrialName"]
            optimal_trial = experiment_data["status"]["currentOptimalTrial"]
            
            parameters = {}
            if optimal_trial.get("parameterAssignments"):
                parameters = {
                    pa["name"]: pa["value"]
                    for pa in optimal_trial["parameterAssignments"]
                    if pa.get("name") and pa.get("value") is not None
                }
            
            metrics = []
            if optimal_trial.get("observation") and optimal_trial["observation"].get("metrics"):
                metrics = [
                    Metric(
                        name=m["name"],
                        latest=float(m.get("latest", m.get("value", 0))),
                        max=float(m.get("max", m.get("value", 0))),
                        min=float(m.get("min", m.get("value", 0))),
                    )
                    for m in optimal_trial["observation"]["metrics"]
                    if m.get("name")
                ]
            
            try:
                trainjob = self.trainer_backend.get_job(name=best_trial_name)
            except Exception as e:
                logger.warning(f"Failed to get TrainJob for best trial {best_trial_name}: {e}")
                trainjob = None
            
            return Trial(
                name=best_trial_name,
                parameters=parameters,
                metrics=metrics,
                trainjob=trainjob,
            )
        
        return None

    def _parse_objectives(self, objective_data: dict) -> list[Objective]:
        """Parse objectives from experiment spec"""
        direction = objective_data.get("type", "maximize")
        metric_name = objective_data.get("objectiveMetricName", "objective")
        
        from kubeflow.optimizer.types.optimization_types import Direction
        
        return [Objective(
            metric=metric_name,
            direction=Direction.MAXIMIZE if direction == "maximize" else Direction.MINIMIZE
        )]

    def _parse_algorithm(self, algorithm_data: dict) -> BaseAlgorithm:
        """Parse algorithm from experiment spec"""
        return RandomSearch()
