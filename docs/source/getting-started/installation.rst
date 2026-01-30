Installation
============

Requirements
------------

- Python 3.9 or higher
- Access to a Kubernetes cluster (for Kubernetes backend)

Install from PyPI
-----------------

.. code-block:: bash

   pip install kubeflow

Optional Dependencies
---------------------

For Docker container backend:

.. code-block:: bash

   pip install kubeflow[docker]

For Podman container backend:

.. code-block:: bash

   pip install kubeflow[podman]

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/kubeflow/sdk.git
   cd sdk
   pip install -e .

Verify Installation
-------------------

.. code-block:: python

   import kubeflow
   print(kubeflow.__version__)

Next Steps
----------

Continue to :doc:`quickstart` to run your first training job.
