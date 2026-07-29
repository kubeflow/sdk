# Copyright The Kubeflow Authors.
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

"""trigger_pipeline component for triggering child pipeline runs."""

import typing

from kfp import dsl

__all__ = ["trigger_pipeline"]


@dsl.component(
    packages_to_install=["kfp"],
    base_image="python:3.11-slim",
)
def trigger_pipeline(
    pipeline_name: str,
    parameters: typing.Dict[str, typing.Any] = None,  # noqa: UP006
    wait_for_completion: bool = False,
    poke_interval: int = 30,
    experiment_name: str = None,
    run_name: str = None,
) -> str:
    """Creates an independent child pipeline run from within a parent pipeline.

    Analogous to Airflow's ``TriggerDagRunOperator``.  The child run gets its
    own run ID, experiment placement, UI entry, and lifecycle — it is *not* a
    nested DAG execution inside the same run.

    When ``wait_for_completion`` is ``False`` (the default), the parent task
    succeeds as soon as the child run is submitted (fire-and-forget).  When
    ``True``, the parent task polls until the child reaches a terminal state.

    .. warning::
        This component makes outbound HTTP calls to the KFP API server at
        runtime.  The container must have network access to the API server
        and the ``kfp`` package installed (handled automatically via
        ``packages_to_install``).

    Args:
        pipeline_name: The display name of a registered pipeline to trigger.
        parameters: Runtime parameters to pass to the child pipeline.
            Values may include outputs from upstream parent tasks.
        wait_for_completion: If ``True``, poll until the child run reaches a
            terminal state before succeeding.  If ``False`` (default), succeed
            immediately after the run is created.
        poke_interval: Seconds between status checks when
            ``wait_for_completion`` is ``True``.
        experiment_name: Experiment for the child run.  Uses the server
            default experiment if omitted.
        run_name: Display name for the child run.  Auto-generated if omitted.

    Returns:
        The run ID of the triggered child pipeline, which downstream tasks
        can use to reference the child run.

    Raises:
        ValueError: If the pipeline or experiment is not found.
        TimeoutError: If ``wait_for_completion`` is ``True`` and the child
            run does not reach a terminal state within the default timeout
            (7 days).
    """
    import logging
    import time

    from kfp import client

    log = logging.getLogger(__name__)

    kfp_client = client.Client()

    # Resolve pipeline name to ID
    pipeline_id = kfp_client.get_pipeline_id(pipeline_name)
    if pipeline_id is None:
        raise ValueError(
            f"Pipeline {pipeline_name!r} not found. "
            "Use list_pipelines() to see available pipelines."
        )

    # Resolve experiment name to ID (if provided)
    experiment_id: str | None = None
    if experiment_name:
        try:
            experiment = kfp_client.get_experiment(experiment_name=experiment_name)
            experiment_id = experiment.experiment_id
        except ValueError as e:
            raise ValueError(
                f"Experiment {experiment_name!r} not found. Use create_experiment() to create one."
            ) from e

    # Generate a default run name if none provided
    if not run_name:
        timestamp = time.strftime("%Y-%m-%d %H-%M-%S")
        run_name = f"trigger-{pipeline_name} {timestamp}"

    # Submit the child run
    log.info(
        "Triggering pipeline %r (id=%s) as run %r",
        pipeline_name,
        pipeline_id,
        run_name,
    )
    run_response = kfp_client.run_pipeline(
        experiment_id=experiment_id or "",
        job_name=run_name,
        pipeline_id=pipeline_id,
        params=parameters or {},
    )
    child_run_id = run_response.run_id
    log.info("Child run submitted: %s", child_run_id)

    # Optionally wait for completion
    if wait_for_completion:
        log.info(
            "Waiting for child run %s to complete (poll every %ds)...",
            child_run_id,
            poke_interval,
        )
        # Default timeout: 7 days (matching KFP backend max)
        final_state = kfp_client.wait_for_run_completion(
            run_id=child_run_id,
            timeout=604800,
            sleep_duration=poke_interval,
        )
        log.info("Child run %s reached terminal state: %s", child_run_id, final_state.state)

    return child_run_id
