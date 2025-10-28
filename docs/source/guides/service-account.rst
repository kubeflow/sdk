Out-of-Cluster Usage with Service Accounts
===========================================

This guide explains how to use the Kubeflow SDK from outside a Kubernetes
cluster using Service Account tokens.

Prerequisites
-------------

1. Access to a Kubernetes cluster with Kubeflow installed
2. Permissions to create Service Accounts and ClusterRoleBindings
3. The cluster's API server endpoint
4. ``kubectl`` configured to access your cluster

Creating a Service Account
---------------------------

First, create a Service Account with the necessary permissions:

.. code-block:: bash

    # Create the Service Account
    kubectl create serviceaccount kubeflow-sdk-user -n kubeflow

    # Create ClusterRoleBinding for permissions
    kubectl create clusterrolebinding kubeflow-sdk-user-binding \
      --clusterrole=kubeflow-edit \
      --serviceaccount=kubeflow:kubeflow-sdk-user

Getting the Service Account Token
----------------------------------

For Kubernetes 1.24 and later:

.. code-block:: bash

    # Create a long-lived token (8760 hours = 1 year)
    kubectl create token kubeflow-sdk-user -n kubeflow --duration=8760h

For older Kubernetes versions:

.. code-block:: bash

    # Get the secret name
    SECRET=$(kubectl get serviceaccount kubeflow-sdk-user -n kubeflow \
      -o jsonpath='{.secrets[0].name}')

    # Extract the token
    TOKEN=$(kubectl get secret $SECRET -n kubeflow \
      -o jsonpath='{.data.token}' | base64 -d)

    echo $TOKEN

Getting the Cluster API Server URL
-----------------------------------

.. code-block:: bash

    # Get the API server URL from kubeconfig
    kubectl cluster-info | grep "Kubernetes control plane"

    # Or extract it directly
    kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}'

Getting the CA Certificate (Optional)
--------------------------------------

For secure connections with custom CA certificates:

.. code-block:: bash

    # Extract CA certificate
    kubectl get secret $SECRET -n kubeflow \
      -o jsonpath='{.data.ca\.crt}' | base64 -d > ca.crt

Configuring the SDK
-------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from kubernetes import client
    from kubeflow.trainer import TrainerClient
    from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig

    # Your cluster details
    api_server = "https://your-cluster-api-server:6443"
    token = "your-service-account-token"

    # Configure Kubernetes client
    configuration = client.Configuration()
    configuration.host = api_server
    configuration.api_key = {"authorization": f"Bearer {token}"}
    configuration.verify_ssl = True

    # Create TrainerClient with custom configuration
    backend_config = KubernetesBackendConfig(
        client_configuration=configuration,
        namespace="kubeflow"
    )

    trainer_client = TrainerClient(backend_config=backend_config)

    # Test the connection
    runtimes = trainer_client.list_runtimes()
    print(f"Available runtimes: {[r.name for r in runtimes]}")

With Custom CA Certificate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from kubernetes import client
    from kubeflow.trainer import TrainerClient
    from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig

    configuration = client.Configuration()
    configuration.host = "https://your-cluster-api-server:6443"
    configuration.api_key = {"authorization": f"Bearer {token}"}
    configuration.verify_ssl = True
    configuration.ssl_ca_cert = "/path/to/ca.crt"

    backend_config = KubernetesBackendConfig(
        client_configuration=configuration,
        namespace="kubeflow"
    )

    trainer_client = TrainerClient(backend_config=backend_config)

Using Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Store sensitive information in environment variables:

.. code-block:: python

    import os
    from kubernetes import client
    from kubeflow.trainer import TrainerClient
    from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig

    configuration = client.Configuration()
    configuration.host = os.getenv("K8S_API_SERVER")
    configuration.api_key = {
        "authorization": f"Bearer {os.getenv('K8S_SERVICE_ACCOUNT_TOKEN')}"
    }
    configuration.verify_ssl = True

    backend_config = KubernetesBackendConfig(
        client_configuration=configuration,
        namespace=os.getenv("K8S_NAMESPACE", "kubeflow")
    )

    trainer_client = TrainerClient(backend_config=backend_config)

Multi-Cluster Setup
-------------------

Managing multiple Kubernetes clusters:

.. code-block:: python

    import os
    from kubeflow.trainer import TrainerClient
    from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig

    def get_trainer_client(cluster_name: str) -> TrainerClient:
        """Create a TrainerClient for a specific cluster."""
        config = KubernetesBackendConfig(
            config_file=f"~/.kube/{cluster_name}-config",
            context=cluster_name,
            namespace="kubeflow"
        )
        return TrainerClient(backend_config=config)

    # Use different clusters
    dev_client = get_trainer_client("dev-cluster")
    staging_client = get_trainer_client("staging-cluster")
    prod_client = get_trainer_client("prod-cluster")

    # List runtimes in each cluster
    print("Dev runtimes:", [r.name for r in dev_client.list_runtimes()])
    print("Prod runtimes:", [r.name for r in prod_client.list_runtimes()])

Complete Example
----------------

Here's a complete example of submitting a training job from outside the cluster:

.. code-block:: python

    import os
    from kubernetes import client
    from kubeflow.trainer import TrainerClient, CustomTrainer
    from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig

    # Configure connection
    configuration = client.Configuration()
    configuration.host = os.getenv("K8S_API_SERVER")
    configuration.api_key = {
        "authorization": f"Bearer {os.getenv('K8S_SERVICE_ACCOUNT_TOKEN')}"
    }
    configuration.verify_ssl = True

    # Create client
    backend_config = KubernetesBackendConfig(
        client_configuration=configuration,
        namespace="kubeflow"
    )
    trainer_client = TrainerClient(backend_config=backend_config)

    # Define training function
    def train():
        import torch.distributed as dist
        dist.init_process_group(backend="gloo")
        print(f"Training on rank {dist.get_rank()}")

    # Submit training job
    job_id = trainer_client.train(
        runtime=trainer_client.get_runtime("torch-distributed"),
        trainer=CustomTrainer(
            func=train,
            num_nodes=2,
            resources_per_node={"cpu": 2, "memory": "4Gi"},
        ),
    )

    print(f"Training job submitted: {job_id}")

    # Monitor job status
    trainer_client.wait_for_job_status(job_id)

    # Get logs
    for log_line in trainer_client.get_job_logs(job_id):
        print(log_line)

Security Best Practices
-----------------------

1. **Store Tokens Securely**

   Never hardcode tokens in your code. Use environment variables or secret management tools:

   - Environment variables
   - HashiCorp Vault
   - AWS Secrets Manager
   - Azure Key Vault
   - Google Secret Manager

2. **Limit Token Lifetime**

   Create tokens with appropriate expiration times based on your use case.

3. **Use RBAC Properly**

   Grant only the minimum required permissions:

   .. code-block:: bash

       # Create a custom role with limited permissions
       kubectl create role kubeflow-trainer \
         --verb=get,list,create,delete \
         --resource=trainjobs,clustertrainingruntimes \
         -n kubeflow

       # Bind the role to the service account
       kubectl create rolebinding kubeflow-sdk-user-binding \
         --role=kubeflow-trainer \
         --serviceaccount=kubeflow:kubeflow-sdk-user \
         -n kubeflow

4. **Rotate Tokens Regularly**

   Implement a token rotation policy and update your applications accordingly.

5. **Enable Audit Logging**

   Track SDK usage in the cluster for security monitoring.

6. **Use TLS/SSL**

   Always verify SSL certificates in production (``verify_ssl=True``).

Troubleshooting
---------------

Connection Refused
~~~~~~~~~~~~~~~~~~

**Symptom**: ``ConnectionRefusedError`` or ``Unable to connect to server``

**Solutions**:

- Verify the API server URL is correct
- Check network connectivity to the cluster
- Ensure firewall rules allow access
- Verify the API server is running

Unauthorized / Forbidden
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``401 Unauthorized`` or ``403 Forbidden``

**Solutions**:

- Verify the Service Account token is valid and not expired
- Check RBAC permissions with: ``kubectl auth can-i --list --as=system:serviceaccount:kubeflow:kubeflow-sdk-user``
- Ensure the ClusterRoleBinding or RoleBinding exists
- Verify the token format (should start with ``Bearer``)

Certificate Verification Failed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``SSLError`` or certificate verification errors

**Solutions**:

- For testing only: Set ``configuration.verify_ssl = False`` (not recommended for production)
- For production: Provide the correct CA certificate with ``configuration.ssl_ca_cert``
- Extract the CA cert from your kubeconfig or cluster

Namespace Not Found
~~~~~~~~~~~~~~~~~~~

**Symptom**: ``Namespace "kubeflow" not found``

**Solutions**:

- Verify the namespace exists: ``kubectl get namespace kubeflow``
- Check the namespace in your ``KubernetesBackendConfig``
- Ensure you have permissions in that namespace

Debugging Tips
~~~~~~~~~~~~~~

Enable verbose logging:

.. code-block:: python

    import logging

    # Enable debug logging for kubernetes client
    logging.basicConfig(level=logging.DEBUG)

    # Or just for kubeflow.trainer
    logging.getLogger('kubeflow.trainer').setLevel(logging.DEBUG)

Test your configuration:

.. code-block:: python

    # Test basic connectivity
    try:
        runtimes = trainer_client.list_runtimes()
        print(f"✓ Connected successfully. Found {len(runtimes)} runtimes")
    except Exception as e:
        print(f"✗ Connection failed: {e}")

Additional Resources
--------------------

- `Kubernetes Service Accounts <https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/>`_
- `Kubernetes RBAC <https://kubernetes.io/docs/reference/access-authn-authz/rbac/>`_
- `Python Kubernetes Client <https://github.com/kubernetes-client/python>`_
- :doc:`../api-reference/backends` - Backend configuration reference
