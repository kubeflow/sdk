# KEP-868: Interactive TrainJob Sessions

## Authors

- N. Hariharan - [@nothariharan](https://github.com/nothariharan)

Ref: https://github.com/kubeflow/sdk/issues/868

## Summary

Add a `connect()` API to the Kubeflow Trainer SDK. `TrainerClient.connect()` creates a
TrainJob that keeps running and returns a `TrainJobSession` that runs commands in the
trainer Pod of that job.

```python
from kubeflow.trainer import CustomTrainerContainer, TrainerClient

client = TrainerClient()
runtime = client.get_runtime("mpi-interactive")

with client.connect(
    runtime=runtime,
    trainer=CustomTrainerContainer(image=runtime.trainer.image, num_nodes=4),
) as session:
    result = session.execute(["mpirun", "hostname"])
    print(result.stdout, result.stderr, result.returncode)
```

The API follows the same shape as the `SparkClient.connect()` API that was added for
Kubeflow Spark, so the two SDKs feel the same for users.

## Motivation

Today the only way to run a command in a training job is to do it by hand:

1. Create a TrainJob whose trainer command keeps the Pod alive, for example
   `sleep infinity`.
2. Wait for the job to be running.
3. Find the launcher Pod with `kubectl get pods` and the JobSet labels.
4. `kubectl exec` into that Pod.
5. Delete the TrainJob afterwards.

This is documented in the Kubeflow Trainer integration guide in the Kubeflow website
repository (https://github.com/kubeflow/website/pull/4492). It works, but it needs
`kubectl`, it needs knowledge of the JobSet labels, and it cannot be done from a
notebook. Users who want to check a distributed setup, copy data around, or drive a
long running job have to leave Python.

The Spark SDK already solved the same problem with `SparkClient.connect()`, so the
Trainer SDK should offer the same shape.

## Goals

1. Create an interactive TrainJob and get a session back from one call.
2. Run commands in the trainer Pod without leaving Python.
3. Return the output, the error output and the exit code of every command.
4. Clean up the TrainJob and its Pods when the session is closed.
5. Keep the Pod alive by default, with an option to override the command.
6. Support the same runtime, initializer, trainer and option arguments as `train()`.

## Non-Goals

1. Attaching to a session that was created by another process. The session always
   creates a new TrainJob.
2. Reconnecting to a session after the process that created it exited.
3. Supporting the LocalProcess and Container backends. They raise a
   `NotImplementedError`.
4. Interactive terminal sessions with a TTY. Commands are executed without a TTY and
   the output is captured.
5. Streaming the output of a long running command while it runs. The output is
   returned when the command finishes.

## User Personas

**Data scientist.** Runs an interactive session in a notebook to check the data and the
environment before starting a real training job.

**ML engineer.** Verifies that a distributed setup works by running `mpirun` in the
launcher Pod of a multi node job.

**Platform engineer.** Runs a diagnostic command in the training environment of the
cluster, for example to print the versions of the installed frameworks.

## User Stories

1. As a data scientist, I want to open a session in the training environment and run
   commands, so that I can explore the environment from my notebook.
2. As an ML engineer, I want to run `mpirun` in a multi node job, so that I can check
   that the nodes can reach each other.
3. As a platform engineer, I want the job and its Pods to be deleted when I am done,
   so that the cluster is left clean.

## Proposal

### TrainerClient.connect()

```python
def connect(
    self,
    runtime: str | types.Runtime | None = None,
    initializer: types.Initializer | None = None,
    trainer: types.CustomTrainerContainer | None = None,
    options: list | None = None,
    timeout: int = 600,
    polling_interval: int = 2,
) -> types.TrainJobSession:
```

| Argument | Description |
|---|---|
| `runtime` | The runtime to use, either a name or a `Runtime` object. Defaults to the torch-distributed runtime. |
| `initializer` | The dataset and model initializers, same as in `train()`. |
| `trainer` | A `CustomTrainerContainer` that provides the image and the number of nodes. |
| `options` | The same options as in `train()`, for example `Name` and `TrainerCommand`. |
| `timeout` | The maximum time to wait for the trainer Pod to be running, in seconds. Defaults to 600. |
| `polling_interval` | The interval to poll the Pods of the TrainJob, in seconds. Defaults to 2. |

`connect()` raises a `ValueError` when the trainer is not a `CustomTrainerContainer`,
because `CustomTrainer` and `BuiltinTrainer` generate their own command and their
training process exits when the training finishes. It raises a `TimeoutError` when the
trainer Pod is not running within the timeout, and a `RuntimeError` when the Pod
finished before the session was created.

### TrainJobSession

```python
class TrainJobSession:
    @property
    def name(self) -> str: ...

    def execute(
        self,
        command: str | list[str],
        *,
        container: str | None = None,
    ) -> types.ExecResult: ...

    def close(self, timeout: int = 180) -> None: ...

    def __enter__(self) -> "TrainJobSession": ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
```

`execute()` runs the command in the trainer Pod. A string command runs with `sh -c`,
so the shell features are available. A list of arguments runs directly, the same way
`kubectl exec` runs a command. The `container` argument selects the container, it
defaults to the trainer container of the Pod.

`close()` deletes the TrainJob and waits until its Pods are deleted. Waiting for the
Pods matters, because the SSH keys of the MPI runtimes are mounted from a secret that
is named after the job. If the old Pods are still there, a new job with the same name
cannot start. `close()` is idempotent and it is called when the session is used as a
context manager.

### ExecResult

```python
@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int | None = None
```

`returncode` is `None` when the Kubernetes API server did not report an exit code.

## Design Details

### Creating the interactive TrainJob

`connect()` calls `train()` with the same arguments as the caller passed, plus a
`TrainerCommand` option with the `sleep infinity` command when the caller did not
provide one. The constant lives in `kubeflow.trainer.constants.constants` as
`INTERACTIVE_COMMAND`. When the caller passes their own `TrainerCommand`, that command
is used and nothing is injected.

### Selecting the trainer Pod

The Pod that runs the trainer command is the Pod with the
`trainer.kubeflow.org/trainjob-ancestor-step: trainer` label. For the MPI runtimes it
is the launcher Pod, for the other runtimes it is the node Pod. Every runtime that
ships with Kubeflow Trainer sets that label.

The label matters for the MPI runtimes. Their node Pods only run the SSH server, and
the launcher Pod is created only after the node Pods are ready. Without the label, a
session could be created on a node Pod before the launcher Pod exists, and the
commands would run on the wrong Pod.

When no Pod has the label, the launcher Pod is used as a fallback, for runtimes that
do not set the label.

### Waiting for the session

`connect()` polls the Pods of the TrainJob until the trainer Pod is in the `Running`
phase. A Pod in the `Succeeded` or `Failed` phase raises a `RuntimeError`, because the
job cannot run any command in that state.

### Running the commands

The commands run through the Kubernetes `pods/exec` API. The client uses
`kubernetes.stream.stream` with `_preload_content=False`, which returns the
WebSocket client that streams the output of the command. The client reads the
standard output and the standard error channels until the connection is closed and
then reads the exit code from the error channel.

Executing commands in a Pod requires the `pods/exec` permission in the namespace of
the TrainJob.

### Closing the session

`close()` deletes the TrainJob and polls the Pods until they are gone. When the Pods
are still there after the timeout, the client logs a warning and returns, it does not
raise. The session is marked as closed either way, so `execute()` raises a
`RuntimeError` after `close()`.

### Backends

`connect()` is part of the `RuntimeBackend` interface. The Kubernetes backend
implements it. The LocalProcess and Container backends raise a `NotImplementedError`,
the same way they do for the other Kubernetes only APIs.

## Alternatives

**Do nothing and keep the kubectl flow.** The flow works, but it needs `kubectl` and
it cannot be used from a notebook. It is also the reason the website guide has to
explain the JobSet labels.

**Add an `exec` method to `TrainerClient` that takes a Pod name.** This would leave
the user with the job creation, the Pod lookup and the cleanup. The session API keeps
all of that inside the SDK.

**Use a TTY for the commands.** A TTY would allow interactive shells, but the output
is then mixed with the terminal control sequences. Capturing the output in three
separate fields is easier to use from a notebook.

## Implementation History

The implementation is in the following pull request:
https://github.com/kubeflow/sdk/pull/886

## Test Plan

### Unit tests

The unit tests are in `kubeflow/trainer/backends/kubernetes/backend_test.py`,
`kubeflow/trainer/types/types_test.py` and `kubeflow/trainer/api/trainer_client_test.py`.
They cover:

1. `connect()` creates the TrainJob with the `sleep infinity` command.
2. `connect()` does not override a `TrainerCommand` that the caller passed.
3. `connect()` rejects a trainer that is not a `CustomTrainerContainer`.
4. `connect()` waits for the trainer Pod and does not use the node Pods of the MPI
   runtimes.
5. `connect()` times out when the Pod does not start, and fails when the Pod
   finished.
6. `connect()` falls back to the launcher Pod when the ancestor label is not set.
7. `execute()` runs the command in the trainer Pod with the `pods/exec` API, with
   the right container, and with the command as a list or as a string.
8. `execute()` returns the output, the error output and the exit code, and it
   reports an empty command, a missing Pod, a Pod that is not running and a failure
   of the exec call.
9. `close()` deletes the TrainJob, waits for the Pods and is idempotent.
10. The context manager closes the session.
11. The client passes all the arguments to the backend, and the LocalProcess and
    Container backends raise a `NotImplementedError`.

### End to end test

The end to end test runs against a real cluster:

1. A kind cluster with one control plane node and two worker nodes, on Kubernetes
   1.37.0.
2. Kubeflow Trainer v2.3.0, installed from the released manifests.
3. The `mpi-interactive` ClusterTrainingRuntime from the Kubeflow website guide, which
   has one launcher Pod and three node Pods, with OpenMPI and the SSH server of the
   MPI Operator.

The test creates a session with four nodes and runs the following checks:

1. `connect()` returns once the launcher Pod is running.
2. `mpirun hostname` returns the hostnames of the launcher Pod and of the three node
   Pods, so the MPI setup works.
3. A string command runs with the shell, and the `$(hostname)` substitution works.
4. A command that exits with 3 reports the exit code 3.
5. A command that writes to the standard error and exits with 1 reports the error
   output and the exit code.
6. Several commands run in the same session.
7. Closing the session deletes the TrainJob and all of its Pods.
8. Running a command in a closed session raises a `RuntimeError`.
9. A new session with the same name can be created right after the first one was
   closed.

The image of the runtime was replaced with a smaller build for the test, because the
image of the guide, `ghcr.io/kubeflow/trainer/deepspeed-runtime:v2.3.0`, is 27.7 GB
and it is based on CUDA. The test image has the same OpenMPI version, the same SSH
setup and the same Pod layout, and only the base image is different.

## Dependencies

No new dependency. The Kubernetes client already provides the exec API.

## Future Work

1. Attaching to a session that already exists.
2. Streaming the output of a long running command.
3. Sessions with a TTY for the interactive shells.
