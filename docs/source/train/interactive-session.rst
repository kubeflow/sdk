Interactive Sessions
====================

Sometimes you don't want to submit a training job and wait for it to finish.
You want to run a command on the cluster, check that a distributed setup is
wired up correctly, or drive a long running job from your notebook.

``TrainerClient.connect()`` creates a TrainJob that keeps running and returns
a session to execute commands in it.

Basic Example
-------------

.. code-block:: python

   from kubeflow.trainer import CustomTrainerContainer, TrainerClient

   client = TrainerClient()
   runtime = client.get_runtime("mpi-interactive")

   with client.connect(
       runtime=runtime,
       trainer=CustomTrainerContainer(image=runtime.trainer.image, num_nodes=4),
   ) as session:
       result = session.execute(["mpirun", "hostname"])
       print(result.stdout)

The session is a context manager. Leaving the ``with`` block deletes the
TrainJob and waits for its Pods to be deleted, so that the job name can be
reused right away.

Running Commands
----------------

``session.execute()`` runs a command in the trainer Pod of the TrainJob, which
is the launcher Pod for the MPI runtimes and the node Pod for the other
runtimes.

.. code-block:: python

   # A list of arguments is executed directly.
   result = session.execute(["mpirun", "python", "train.py"])

   # A string is executed with `sh -c`, so the shell features are available.
   result = session.execute("mpirun -n 4 python train.py --epochs 10")

   print(result.stdout)
   print(result.stderr)
   print(result.returncode)

The command output is captured in the ``ExecResult``, so the session can be used
from a notebook without a terminal. The exit code of the command is reported
as well, so failures can be handled:

.. code-block:: python

   result = session.execute("python -c 'import torch; print(torch.cuda.is_available())'")
   if result.returncode != 0:
       raise RuntimeError(result.stderr)

The same session can run as many commands as needed, as long as the TrainJob
is running.

Keeping the Pod Alive
---------------------

The trainer Pod must stay running for the session to be usable. By default the
SDK keeps it alive with the ``sleep infinity`` command. Use the
``TrainerCommand`` option to run a different command, for example to start an
interactive shell:

.. code-block:: python

   from kubeflow.trainer.options import Name, TrainerCommand

   with client.connect(
       runtime=runtime,
       trainer=CustomTrainerContainer(image=runtime.trainer.image, num_nodes=4),
       options=[Name("my-session"), TrainerCommand(["sleep", "infinity"])],
   ) as session:
       session.execute("mpirun hostname")

Requirements
------------

- The Kubernetes backend is required. The LocalProcess and Container backends
  raise a ``NotImplementedError``.
- The trainer must be a ``CustomTrainerContainer``. The ``CustomTrainer`` and
  ``BuiltinTrainer`` trainers generate their own command, so their training
  process exits when the training finishes.
- Executing commands in a Pod requires the ``pods/exec`` permission in the
  namespace of the TrainJob.
