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
Storage layer for container backend - mimics Katib Experiment/Trial CRs using JSON.

This module provides functions to persist optimization job state locally using
JSON files that mirror the structure of Katib's Kubernetes Custom Resources
(Experiments and Trials). This enables the container backend to provide the
same API and behavior as the Kubernetes backend.

Storage Structure:
    {storage_path}/{job_name}/
    ├── experiment.json          # Katib Experiment CR equivalent
    │   ├── metadata: {name, creationTimestamp}
    │   ├── spec: {parameters, objective, algorithm, trialTemplate, maxTrialCount}
    │   └── status: {conditions, currentOptimalTrial, trials, trialsSucceeded}
    └── trials/                  # Katib Trial CRs equivalent
        ├── <trial-001>.json
        │   ├── metadata: {name, creationTimestamp}
        │   ├── spec: {parameterAssignments}
        │   └── status: {observation: {metrics}}
        └── <trial-002>.json

Thread Safety:
    All write operations use file locking (fcntl on Unix, msvcrt on Windows)
    to ensure safe concurrent access during parallel trial execution.
"""

from datetime import datetime, timezone
import fcntl
import json
import logging
from pathlib import Path
import platform
import shutil

logger = logging.getLogger(__name__)

# Constants matching Katib structure
EXPERIMENT_FILE = "experiment.json"
TRIALS_DIR = "trials"


def _get_job_path(storage_path: str, job_name: str) -> Path:
    """Get the path to a job's storage directory.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.

    Returns:
        Path object for the job directory.
    """
    return Path(storage_path) / job_name


def _get_experiment_path(storage_path: str, job_name: str) -> Path:
    """Get the path to the experiment.json file.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.

    Returns:
        Path object for the experiment.json file.
    """
    return _get_job_path(storage_path, job_name) / EXPERIMENT_FILE


def _get_trials_dir_path(storage_path: str, job_name: str) -> Path:
    """Get the path to the trials directory.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.

    Returns:
        Path object for the trials directory.
    """
    return _get_job_path(storage_path, job_name) / TRIALS_DIR


def _get_trial_path(storage_path: str, job_name: str, trial_name: str) -> Path:
    """Get the path to a trial's JSON file.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.
        trial_name: Name of the trial.

    Returns:
        Path object for the trial JSON file.
    """
    return _get_trials_dir_path(storage_path, job_name) / f"{trial_name}.json"


def _lock_file(file_handle):
    """Lock a file for exclusive write access (platform-independent).

    Args:
        file_handle: Open file handle to lock.

    Note:
        Uses fcntl on Unix/Linux/macOS, msvcrt on Windows.
    """
    system = platform.system()
    if system in ["Linux", "Darwin"]:  # Unix-like systems
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
    elif system == "Windows":
        import msvcrt

        msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
    # If unknown platform, proceed without locking (best effort)


def _unlock_file(file_handle):
    """Unlock a file after write operation (platform-independent).

    Args:
        file_handle: Open file handle to unlock.
    """
    system = platform.system()
    if system in ["Linux", "Darwin"]:
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
    elif system == "Windows":
        import msvcrt

        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)


def _write_json_with_lock(path: Path, data: dict):
    """Write JSON data to file with locking for thread safety.

    Args:
        path: Path to the JSON file.
        data: Dictionary to serialize as JSON.

    Raises:
        IOError: If file write fails.
        json.JSONDecodeError: If data is not JSON-serializable.
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write with exclusive lock
    with open(path, "w") as f:
        _lock_file(f)
        try:
            json.dump(data, f, indent=2, default=str)
        finally:
            _unlock_file(f)

    logger.debug(f"Wrote JSON to {path}")


def _read_json(path: Path) -> dict:
    """Read and parse JSON from file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data as dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    with open(path) as f:
        data = json.load(f)
    logger.debug(f"Read JSON from {path}")
    return data


def _get_current_timestamp() -> str:
    """Get current timestamp in ISO 8601 format (matches Kubernetes).

    Returns:
        Current timestamp as ISO 8601 string with UTC timezone.
        Example: "2025-11-08T10:30:45.123456Z"
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ============================================================================
# Experiment Storage Functions (Katib Experiment CR equivalent)
# ============================================================================


def create_experiment_storage(
    storage_path: str,
    job_name: str,
    experiment_data: dict,
) -> None:
    """Create storage for a new optimization job with experiment metadata.

    This initializes the directory structure and creates the experiment.json
    file that mirrors a Katib Experiment CR. The experiment data should include
    metadata, spec, and status sections matching Katib's structure.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.
        experiment_data: Dictionary containing experiment metadata, spec, and status.
            Should match Katib Experiment CR structure:
            {
                "metadata": {"name": str, "creationTimestamp": str},
                "spec": {
                    "parameters": list,
                    "objective": dict,
                    "algorithm": dict,
                    "trialTemplate": dict,
                    "maxTrialCount": int,
                    "parallelTrialCount": int
                },
                "status": {
                    "conditions": list,
                    "currentOptimalTrial": dict or None,
                    "trials": int,
                    "trialsSucceeded": int,
                    "trialsFailed": int,
                    "trialsRunning": int
                }
            }

    Raises:
        ValueError: If job already exists.
        IOError: If directory creation or file write fails.

    Example:
        >>> experiment_data = {
        ...     "metadata": {"name": "job-123", "creationTimestamp": "2025-11-08T10:00:00Z"},
        ...     "spec": {"parameters": [...], "maxTrialCount": 10},
        ...     "status": {"conditions": [], "trials": 0},
        ... }
        >>> create_experiment_storage("/path/to/storage", "job-123", experiment_data)
    """
    job_path = _get_job_path(storage_path, job_name)

    # Check if job already exists
    if job_path.exists():
        raise ValueError(f"Job '{job_name}' already exists at {job_path}")

    # Create directory structure
    job_path.mkdir(parents=True, exist_ok=True)
    trials_dir = _get_trials_dir_path(storage_path, job_name)
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Add creation timestamp if not present
    if "metadata" not in experiment_data:
        experiment_data["metadata"] = {}
    if "creationTimestamp" not in experiment_data["metadata"]:
        experiment_data["metadata"]["creationTimestamp"] = _get_current_timestamp()

    # Write experiment.json
    experiment_path = _get_experiment_path(storage_path, job_name)
    _write_json_with_lock(experiment_path, experiment_data)

    logger.info(f"Created experiment storage for job '{job_name}' at {job_path}")


def load_experiment(storage_path: str, job_name: str) -> dict:
    """Load experiment metadata from storage.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.

    Returns:
        Dictionary containing experiment metadata, spec, and status
        (Katib Experiment CR structure).

    Raises:
        ValueError: If job doesn't exist.
        json.JSONDecodeError: If experiment.json is corrupted.

    Example:
        >>> experiment = load_experiment("/path/to/storage", "job-123")
        >>> print(experiment["status"]["trials"])
        5
    """
    experiment_path = _get_experiment_path(storage_path, job_name)

    if not experiment_path.exists():
        raise ValueError(
            f"Job '{job_name}' not found in storage. Expected experiment file at: {experiment_path}"
        )

    try:
        return _read_json(experiment_path)
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted experiment file for job '{job_name}': {e}")
        raise ValueError(f"Corrupted experiment data for job '{job_name}'") from e


def update_experiment_status(
    storage_path: str,
    job_name: str,
    status_data: dict,
) -> None:
    """Update the status section of an experiment.

    This merges the provided status_data with the existing experiment's status
    section, similar to how Kubernetes updates CR status.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.
        status_data: Dictionary with status fields to update. Can include:
            - conditions: list of condition objects
            - currentOptimalTrial: dict with best trial info
            - trials: total trial count
            - trialsSucceeded: succeeded count
            - trialsFailed: failed count
            - trialsRunning: running count

    Raises:
        ValueError: If job doesn't exist.
        IOError: If file write fails.

    Example:
        >>> update_experiment_status(
        ...     "/path/to/storage",
        ...     "job-123",
        ...     {
        ...         "trials": 5,
        ...         "trialsSucceeded": 3,
        ...         "trialsRunning": 2,
        ...     },
        ... )
    """
    # Load current experiment
    experiment = load_experiment(storage_path, job_name)

    # Update status section
    if "status" not in experiment:
        experiment["status"] = {}

    experiment["status"].update(status_data)

    # Save back
    experiment_path = _get_experiment_path(storage_path, job_name)
    _write_json_with_lock(experiment_path, experiment)

    logger.debug(f"Updated experiment status for job '{job_name}'")


def list_experiments(storage_path: str) -> list[str]:
    """List all optimization job names in storage.

    Args:
        storage_path: Base storage directory path.

    Returns:
        List of job names (directory names that contain experiment.json).
        Returns empty list if storage directory doesn't exist or is empty.

    Example:
        >>> jobs = list_experiments("/path/to/storage")
        >>> print(jobs)
        ['job-123', 'job-456', 'job-789']
    """
    storage = Path(storage_path)

    if not storage.exists():
        logger.debug(f"Storage path {storage_path} doesn't exist, returning empty list")
        return []

    job_names = []
    for item in storage.iterdir():
        if item.is_dir():
            # Check if it has experiment.json (valid job)
            experiment_file = item / EXPERIMENT_FILE
            if experiment_file.exists():
                job_names.append(item.name)

    logger.debug(f"Found {len(job_names)} experiments in storage")
    return sorted(job_names)


# ============================================================================
# Trial Storage Functions (Katib Trial CR equivalent)
# ============================================================================


def create_trial(
    storage_path: str,
    job_name: str,
    trial_name: str,
    trial_data: dict,
) -> None:
    """Create a new trial record in storage.

    This creates a JSON file for a trial that mirrors a Katib Trial CR.
    The trial data should include metadata, spec (parameter assignments),
    and status (observation with metrics).

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.
        trial_name: Name of the trial.
        trial_data: Dictionary containing trial metadata, spec, and status.
            Should match Katib Trial CR structure:
            {
                "metadata": {"name": str, "creationTimestamp": str},
                "spec": {
                    "parameterAssignments": [
                        {"name": str, "value": str},
                        ...
                    ]
                },
                "status": {
                    "conditions": list,
                    "observation": {
                        "metrics": [
                            {"name": str, "value": float},
                            ...
                        ]
                    }
                }
            }

    Raises:
        ValueError: If trial already exists or job doesn't exist.
        IOError: If file write fails.

    Example:
        >>> trial_data = {
        ...     "metadata": {"name": "trial-001"},
        ...     "spec": {"parameterAssignments": [{"name": "lr", "value": "0.01"}]},
        ...     "status": {"observation": {"metrics": []}},
        ... }
        >>> create_trial("/path/to/storage", "job-123", "trial-001", trial_data)
    """
    trial_path = _get_trial_path(storage_path, job_name, trial_name)

    # Check if trial already exists
    if trial_path.exists():
        raise ValueError(f"Trial '{trial_name}' already exists for job '{job_name}'")

    # Verify job exists
    if not _get_experiment_path(storage_path, job_name).exists():
        raise ValueError(f"Cannot create trial: job '{job_name}' doesn't exist")

    # Add creation timestamp if not present
    if "metadata" not in trial_data:
        trial_data["metadata"] = {}
    if "creationTimestamp" not in trial_data["metadata"]:
        trial_data["metadata"]["creationTimestamp"] = _get_current_timestamp()

    # Write trial JSON
    _write_json_with_lock(trial_path, trial_data)

    logger.debug(f"Created trial '{trial_name}' for job '{job_name}'")


def load_trial(storage_path: str, job_name: str, trial_name: str) -> dict:
    """Load trial data from storage.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.
        trial_name: Name of the trial.

    Returns:
        Dictionary containing trial metadata, spec, and status
        (Katib Trial CR structure).

    Raises:
        ValueError: If trial doesn't exist.
        json.JSONDecodeError: If trial JSON is corrupted.

    Example:
        >>> trial = load_trial("/path/to/storage", "job-123", "trial-001")
        >>> metrics = trial["status"]["observation"]["metrics"]
        >>> print(metrics)
    """
    trial_path = _get_trial_path(storage_path, job_name, trial_name)

    if not trial_path.exists():
        raise ValueError(
            f"Trial '{trial_name}' not found for job '{job_name}'. Expected file at: {trial_path}"
        )

    try:
        return _read_json(trial_path)
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted trial file for '{trial_name}' in job '{job_name}': {e}")
        raise ValueError(f"Corrupted trial data for '{trial_name}' in job '{job_name}'") from e


def update_trial_status(
    storage_path: str,
    job_name: str,
    trial_name: str,
    status_data: dict,
) -> None:
    """Update the status section of a trial.

    This merges the provided status_data with the existing trial's status
    section, similar to how Kubernetes updates CR status.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.
        trial_name: Name of the trial.
        status_data: Dictionary with status fields to update. Can include:
            - conditions: list of condition objects
            - observation: dict with metrics array
            - completionTime: timestamp string

    Raises:
        ValueError: If trial doesn't exist.
        IOError: If file write fails.

    Example:
        >>> update_trial_status(
        ...     "/path/to/storage",
        ...     "job-123",
        ...     "trial-001",
        ...     {"observation": {"metrics": [{"name": "accuracy", "value": 0.95}]}},
        ... )
    """
    # Load current trial
    trial = load_trial(storage_path, job_name, trial_name)

    # Update status section
    if "status" not in trial:
        trial["status"] = {}

    trial["status"].update(status_data)

    # Save back
    trial_path = _get_trial_path(storage_path, job_name, trial_name)
    _write_json_with_lock(trial_path, trial)

    logger.debug(f"Updated trial status for '{trial_name}' in job '{job_name}'")


def list_trials(storage_path: str, job_name: str) -> list[str]:
    """List all trial names for a job.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.

    Returns:
        List of trial names (without .json extension).
        Returns empty list if no trials exist.

    Raises:
        ValueError: If job doesn't exist.

    Example:
        >>> trials = list_trials("/path/to/storage", "job-123")
        >>> print(trials)
        ['trial-001', 'trial-002', 'trial-003']
    """
    trials_dir = _get_trials_dir_path(storage_path, job_name)

    # Verify job exists
    if not _get_experiment_path(storage_path, job_name).exists():
        raise ValueError(f"Job '{job_name}' doesn't exist")

    if not trials_dir.exists():
        logger.debug(f"Trials directory doesn't exist for job '{job_name}'")
        return []

    trial_names = []
    for item in trials_dir.iterdir():
        if item.is_file() and item.suffix == ".json":
            trial_names.append(item.stem)  # Remove .json extension

    logger.debug(f"Found {len(trial_names)} trials for job '{job_name}'")
    return sorted(trial_names)


# ============================================================================
# Cleanup Functions
# ============================================================================


def delete_job_storage(storage_path: str, job_name: str) -> None:
    """Delete all storage for an optimization job.

    This removes the entire job directory including experiment.json and all
    trial files. This operation is irreversible.

    Args:
        storage_path: Base storage directory path.
        job_name: Name of the optimization job.

    Raises:
        ValueError: If job doesn't exist.
        IOError: If deletion fails (logs warning but doesn't raise).

    Example:
        >>> delete_job_storage("/path/to/storage", "job-123")
    """
    job_path = _get_job_path(storage_path, job_name)

    if not job_path.exists():
        raise ValueError(f"Job '{job_name}' doesn't exist, cannot delete")

    try:
        shutil.rmtree(job_path)
        logger.info(f"Deleted storage for job '{job_name}'")
    except Exception as e:
        logger.warning(f"Failed to delete storage for job '{job_name}': {e}")
        raise OSError(f"Failed to delete job storage: {e}") from e
