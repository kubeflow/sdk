"""Data models for Spark application requests and responses."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ApplicationState(Enum):
    """Spark application states matching Spark Operator CRD states."""

    # Standard states from Spark Operator (v1beta2)
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"
    PENDING_RERUN = "PENDING_RERUN"
    INVALIDATING = "INVALIDATING"
    SUCCEEDING = "SUCCEEDING"
    FAILING = "FAILING"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    RESUMING = "RESUMING"
    UNKNOWN = "UNKNOWN"


class SparkConnectState(Enum):
    """Spark Connect server states matching Spark Operator SparkConnect CRD (v1alpha1).

    These states represent the lifecycle of a SparkConnect server managed by
    the Spark Operator.
    """

    # Server is being provisioned (pods being created)
    PROVISIONING = "Provisioning"
    # Server is ready and accepting connections
    READY = "Ready"
    # Server pods are not ready
    NOT_READY = "NotReady"
    # Server has failed
    FAILED = "Failed"
    # Unknown state
    UNKNOWN = "Unknown"


class RestartPolicyType(Enum):
    """Restart policy types from operator."""

    NEVER = "Never"
    ON_FAILURE = "OnFailure"
    ALWAYS = "Always"


class DeployMode(Enum):
    """Deployment modes for Spark applications."""

    CLUSTER = "cluster"
    CLIENT = "client"
    IN_CLUSTER_CLIENT = "in-cluster-client"


@dataclass
class RestartPolicy:
    """Restart policy configuration (matches operator RestartPolicy).

    Attributes:
        type: Type of restart policy
        on_failure_retries: Number of times to retry on failure
        on_failure_retry_interval: Interval in seconds between failure retries
        on_submission_failure_retries: Number of times to retry on submission failure
        on_submission_failure_retry_interval: Interval in seconds between submission retries
    """

    type: RestartPolicyType = RestartPolicyType.NEVER
    on_failure_retries: Optional[int] = None
    on_failure_retry_interval: int = 5  # Default from operator
    on_submission_failure_retries: Optional[int] = None
    on_submission_failure_retry_interval: int = 5  # Default from operator


@dataclass
class GPUSpec:
    """GPU specification for driver or executor.

    Attributes:
        name: GPU resource name (e.g., "nvidia.com/gpu", "amd.com/gpu")
        quantity: Number of GPUs to request
    """

    name: str
    quantity: int


@dataclass
class DynamicAllocation:
    """Dynamic allocation configuration (Spark 3.0+).

    Attributes:
        enabled: Whether dynamic allocation is enabled
        initial_executors: Initial number of executors
        min_executors: Minimum number of executors
        max_executors: Maximum number of executors
        shuffle_tracking_enabled: Enable shuffle tracking (default true if dynamic allocation enabled)
        shuffle_tracking_timeout: Timeout in milliseconds for shuffle tracking
    """

    enabled: bool = False
    initial_executors: Optional[int] = None
    min_executors: Optional[int] = None
    max_executors: Optional[int] = None
    shuffle_tracking_enabled: Optional[bool] = True
    shuffle_tracking_timeout: Optional[int] = None


@dataclass
class BatchSchedulerConfig:
    """Batch scheduler configuration (Volcano, Yunikorn).

    Attributes:
        queue: Resource queue name
        priority_class_name: Kubernetes PriorityClass name
    """

    queue: Optional[str] = None
    priority_class_name: Optional[str] = None


@dataclass
class PrometheusSpec:
    """Prometheus JMX exporter configuration.

    Attributes:
        jmx_exporter_jar: Path to Prometheus JMX exporter jar
        port: Port for Prometheus JMX exporter (default 8090)
        port_name: Port name (default "jmx-exporter")
        config_file: Path to custom Prometheus config file
        configuration: Prometheus configuration content
    """

    jmx_exporter_jar: str
    port: int = 8090
    port_name: str = "jmx-exporter"
    config_file: Optional[str] = None
    configuration: Optional[str] = None


@dataclass
class MonitoringSpec:
    """Monitoring configuration.

    Attributes:
        expose_driver_metrics: Whether to expose driver metrics
        expose_executor_metrics: Whether to expose executor metrics
        metrics_properties: Content of metrics.properties file
        metrics_properties_file: Path to metrics.properties file
        prometheus: Prometheus configuration
    """

    expose_driver_metrics: bool = False
    expose_executor_metrics: bool = False
    metrics_properties: Optional[str] = None
    metrics_properties_file: Optional[str] = None
    prometheus: Optional[PrometheusSpec] = None


@dataclass
class SparkUIConfiguration:
    """Spark UI service and ingress configuration.

    Attributes:
        service_port: Service port (different from target port)
        service_port_name: Service port name (default "spark-driver-ui-port")
        service_type: Kubernetes service type (default ClusterIP)
        service_annotations: Service annotations
        service_labels: Service labels
        ingress_annotations: Ingress annotations
        ingress_tls: Ingress TLS configuration
    """

    service_port: Optional[int] = None
    service_port_name: str = "spark-driver-ui-port"
    service_type: str = "ClusterIP"
    service_annotations: dict[str, str] = field(default_factory=dict)
    service_labels: dict[str, str] = field(default_factory=dict)
    ingress_annotations: dict[str, str] = field(default_factory=dict)
    ingress_tls: Optional[list[dict[str, Any]]] = None


@dataclass
class SparkApplicationRequest:
    """Request model for Spark application submission (enhanced to match operator v1beta2).

    Attributes:
        # === Basic Configuration ===
        app_name: Name of the Spark application
        main_application_file: Path to main application file (S3 or local)
        spark_version: Spark version to use
        app_type: Application type (Python, Scala, Java, R)

        # === Resource Configuration ===
        driver_cores: Number of cores for driver
        driver_memory: Memory for driver (e.g., "4g")
        executor_cores: Number of cores per executor
        executor_memory: Memory per executor (e.g., "8g")
        num_executors: Number of executors

        # === Application Configuration ===
        arguments: Application arguments
        main_class: Main class for Java/Scala applications
        python_version: Python version (for PySpark apps)
        spark_conf: Spark configuration properties
        hadoop_conf: Hadoop configuration properties
        env_vars: Environment variables
        deps: Dependencies (jars, py files, files)

        # === Advanced Configuration (NEW) ===
        mode: Deployment mode (cluster, client, in-cluster-client)
        image: Container image (overrides default)
        image_pull_policy: Image pull policy (IfNotPresent, Always, Never)
        image_pull_secrets: List of image pull secret names

        # === Lifecycle & Resilience (NEW) ===
        suspend: Suspend the application (pause execution)
        restart_policy: Restart policy configuration
        time_to_live_seconds: TTL for auto-cleanup after termination

        # === GPU Support (NEW) ===
        driver_gpu: GPU specification for driver
        executor_gpu: GPU specification for executor

        # === Dynamic Allocation (NEW) ===
        dynamic_allocation: Dynamic allocation configuration

        # === Monitoring & Observability (NEW) ===
        monitoring: Monitoring configuration
        spark_ui_options: Spark UI configuration

        # === Batch Scheduling (NEW) ===
        batch_scheduler: Batch scheduler name (volcano, yunikorn)
        batch_scheduler_options: Batch scheduler configuration

        # === Networking & Security (NEW) ===
        service_account: Kubernetes service account
        node_selector: Node selector for driver and executor
        tolerations: Kubernetes tolerations
        affinity: Kubernetes affinity rules
        host_network: Use host networking
        pod_security_context: Pod security context
        security_context: Container security context

        # === Advanced Features (NEW) ===
        driver_pod_template: Full PodTemplateSpec for driver (Spark 3.0+)
        executor_pod_template: Full PodTemplateSpec for executor (Spark 3.0+)
        volumes: Kubernetes volumes
        driver_volume_mounts: Driver volume mounts
        executor_volume_mounts: Executor volume mounts
        driver_sidecars: Sidecar containers for driver
        executor_sidecars: Sidecar containers for executor
        driver_init_containers: Init containers for driver
        executor_init_containers: Init containers for executor

        # === Labels & Annotations (NEW) ===
        labels: Kubernetes labels
        driver_labels: Driver-specific labels
        executor_labels: Executor-specific labels
        annotations: Kubernetes annotations
        driver_annotations: Driver-specific annotations
        executor_annotations: Executor-specific annotations

        # === Legacy (DEPRECATED, keeping for backward compat) ===
        queue: Queue to submit to (legacy - use namespace or batch_scheduler_options.queue)
    """

    # === Required Fields ===
    app_name: str
    main_application_file: str

    # === Basic Configuration ===
    spark_version: str = "3.5.0"
    app_type: str = "Python"
    mode: DeployMode = DeployMode.CLUSTER

    # === Resource Configuration ===
    driver_cores: int = 1
    driver_memory: str = "1g"
    executor_cores: int = 1
    executor_memory: str = "1g"
    num_executors: int = 2

    # === Application Configuration ===
    arguments: list[str] = field(default_factory=list)
    main_class: Optional[str] = None
    python_version: str = "3"
    spark_conf: dict[str, str] = field(default_factory=dict)
    hadoop_conf: dict[str, str] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    deps: Optional[dict[str, list[str]]] = None

    # === Image Configuration ===
    image: Optional[str] = None
    image_pull_policy: str = "IfNotPresent"
    image_pull_secrets: list[str] = field(default_factory=list)

    # === Lifecycle & Resilience ===
    suspend: Optional[bool] = None
    restart_policy: RestartPolicy = field(default_factory=RestartPolicy)
    time_to_live_seconds: Optional[int] = None

    # === GPU Support ===
    driver_gpu: Optional[GPUSpec] = None
    executor_gpu: Optional[GPUSpec] = None

    # === Dynamic Allocation ===
    dynamic_allocation: Optional[DynamicAllocation] = None

    # === Monitoring & Observability ===
    monitoring: Optional[MonitoringSpec] = None
    spark_ui_options: Optional[SparkUIConfiguration] = None

    # === Batch Scheduling ===
    batch_scheduler: Optional[str] = None
    batch_scheduler_options: Optional[BatchSchedulerConfig] = None

    # === Networking & Security ===
    service_account: str = "spark-operator-spark"
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict[str, Any]] = field(default_factory=list)
    affinity: Optional[dict[str, Any]] = None
    host_network: Optional[bool] = None
    pod_security_context: Optional[dict[str, Any]] = None
    security_context: Optional[dict[str, Any]] = None

    # === Pod Templates (Spark 3.0+) ===
    driver_pod_template: Optional[dict[str, Any]] = None
    executor_pod_template: Optional[dict[str, Any]] = None

    # === Volumes ===
    volumes: list[dict[str, Any]] = field(default_factory=list)
    driver_volume_mounts: list[dict[str, Any]] = field(default_factory=list)
    executor_volume_mounts: list[dict[str, Any]] = field(default_factory=list)

    # === Sidecars & Init Containers ===
    driver_sidecars: list[dict[str, Any]] = field(default_factory=list)
    executor_sidecars: list[dict[str, Any]] = field(default_factory=list)
    driver_init_containers: list[dict[str, Any]] = field(default_factory=list)
    executor_init_containers: list[dict[str, Any]] = field(default_factory=list)

    # === Labels & Annotations ===
    labels: dict[str, str] = field(default_factory=dict)
    driver_labels: dict[str, str] = field(default_factory=dict)
    executor_labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    driver_annotations: dict[str, str] = field(default_factory=dict)
    executor_annotations: dict[str, str] = field(default_factory=dict)

    # === Legacy ===
    queue: str = "poc"

    def to_dict(self) -> dict[str, Any]:
        """Convert request to dictionary for operator-compliant SparkApplication CRD.

        Returns:
            Dictionary representation matching operator's v1beta2 SparkApplication schema
        """
        # === Build metadata ===
        metadata = {"name": self.app_name}

        if self.labels:
            metadata["labels"] = self.labels.copy()
        if self.annotations:
            metadata["annotations"] = self.annotations.copy()

        # === Build spec ===
        spec: dict[str, Any] = {
            "type": self.app_type,
            "mode": self.mode.value if isinstance(self.mode, DeployMode) else self.mode,
            "mainApplicationFile": self.main_application_file,
            "sparkVersion": self.spark_version,
        }

        # Image configuration
        if self.image:
            spec["image"] = self.image
        else:
            # Default image based on app type
            if self.app_type.lower() == "python":
                spec["image"] = f"gcr.io/spark-operator/spark-py:{self.spark_version}"
            else:
                spec["image"] = f"gcr.io/spark-operator/spark:{self.spark_version}"

        spec["imagePullPolicy"] = self.image_pull_policy
        if self.image_pull_secrets:
            spec["imagePullSecrets"] = self.image_pull_secrets

        # Main class for Java/Scala
        if self.main_class:
            spec["mainClass"] = self.main_class

        # Python version
        if self.app_type.lower() == "python" and self.python_version:
            spec["pythonVersion"] = self.python_version

        # === Lifecycle & Resilience ===
        if self.suspend is not None:
            spec["suspend"] = self.suspend

        if self.time_to_live_seconds is not None:
            spec["timeToLiveSeconds"] = self.time_to_live_seconds

        # Restart policy
        restart_policy_dict = {"type": self.restart_policy.type.value}
        if self.restart_policy.on_failure_retries is not None:
            restart_policy_dict["onFailureRetries"] = self.restart_policy.on_failure_retries
        if self.restart_policy.on_failure_retry_interval:
            restart_policy_dict["onFailureRetryInterval"] = (
                self.restart_policy.on_failure_retry_interval
            )
        if self.restart_policy.on_submission_failure_retries is not None:
            restart_policy_dict["onSubmissionFailureRetries"] = (
                self.restart_policy.on_submission_failure_retries
            )
        if self.restart_policy.on_submission_failure_retry_interval:
            restart_policy_dict["onSubmissionFailureRetryInterval"] = (
                self.restart_policy.on_submission_failure_retry_interval
            )
        spec["restartPolicy"] = restart_policy_dict

        # === Configuration ===
        if self.arguments:
            spec["arguments"] = self.arguments

        if self.spark_conf:
            spec["sparkConf"] = self.spark_conf.copy()

        if self.hadoop_conf:
            spec["hadoopConf"] = self.hadoop_conf.copy()

        if self.deps:
            spec["deps"] = self.deps

        # === Batch Scheduling ===
        if self.batch_scheduler:
            spec["batchScheduler"] = self.batch_scheduler

        if self.batch_scheduler_options:
            batch_opts = {}
            if self.batch_scheduler_options.queue:
                batch_opts["queue"] = self.batch_scheduler_options.queue
            if self.batch_scheduler_options.priority_class_name:
                batch_opts["priorityClassName"] = self.batch_scheduler_options.priority_class_name
            if batch_opts:
                spec["batchSchedulerOptions"] = batch_opts

        # === Monitoring ===
        if self.monitoring:
            mon_spec = {
                "exposeDriverMetrics": self.monitoring.expose_driver_metrics,
                "exposeExecutorMetrics": self.monitoring.expose_executor_metrics,
            }
            if self.monitoring.metrics_properties:
                mon_spec["metricsProperties"] = self.monitoring.metrics_properties
            if self.monitoring.metrics_properties_file:
                mon_spec["metricsPropertiesFile"] = self.monitoring.metrics_properties_file
            if self.monitoring.prometheus:
                prom_spec = {
                    "jmxExporterJar": self.monitoring.prometheus.jmx_exporter_jar,
                    "port": self.monitoring.prometheus.port,
                    "portName": self.monitoring.prometheus.port_name,
                }
                if self.monitoring.prometheus.config_file:
                    prom_spec["configFile"] = self.monitoring.prometheus.config_file
                if self.monitoring.prometheus.configuration:
                    prom_spec["configuration"] = self.monitoring.prometheus.configuration
                mon_spec["prometheus"] = prom_spec
            spec["monitoring"] = mon_spec

        # === Spark UI ===
        if self.spark_ui_options:
            ui_opts = {}
            if self.spark_ui_options.service_port:
                ui_opts["servicePort"] = self.spark_ui_options.service_port
            if self.spark_ui_options.service_port_name:
                ui_opts["servicePortName"] = self.spark_ui_options.service_port_name
            if self.spark_ui_options.service_type:
                ui_opts["serviceType"] = self.spark_ui_options.service_type
            if self.spark_ui_options.service_annotations:
                ui_opts["serviceAnnotations"] = self.spark_ui_options.service_annotations
            if self.spark_ui_options.service_labels:
                ui_opts["serviceLabels"] = self.spark_ui_options.service_labels
            if self.spark_ui_options.ingress_annotations:
                ui_opts["ingressAnnotations"] = self.spark_ui_options.ingress_annotations
            if self.spark_ui_options.ingress_tls:
                ui_opts["ingressTLS"] = self.spark_ui_options.ingress_tls
            if ui_opts:
                spec["sparkUIOptions"] = ui_opts

        # === Dynamic Allocation ===
        if self.dynamic_allocation and self.dynamic_allocation.enabled:
            dyn_alloc = {"enabled": True}
            if self.dynamic_allocation.initial_executors is not None:
                dyn_alloc["initialExecutors"] = self.dynamic_allocation.initial_executors
            if self.dynamic_allocation.min_executors is not None:
                dyn_alloc["minExecutors"] = self.dynamic_allocation.min_executors
            if self.dynamic_allocation.max_executors is not None:
                dyn_alloc["maxExecutors"] = self.dynamic_allocation.max_executors
            if self.dynamic_allocation.shuffle_tracking_enabled is not None:
                dyn_alloc["shuffleTrackingEnabled"] = (
                    self.dynamic_allocation.shuffle_tracking_enabled
                )
            if self.dynamic_allocation.shuffle_tracking_timeout is not None:
                dyn_alloc["shuffleTrackingTimeout"] = (
                    self.dynamic_allocation.shuffle_tracking_timeout
                )
            spec["dynamicAllocation"] = dyn_alloc

        # === Volumes ===
        if self.volumes:
            spec["volumes"] = self.volumes

        # === Node Selector (spec-level) ===
        if self.node_selector:
            spec["nodeSelector"] = self.node_selector

        # === Driver Spec ===
        driver_spec = {
            "cores": self.driver_cores,
            "memory": self.driver_memory,
            "serviceAccount": self.service_account,
        }

        # Driver labels & annotations
        driver_labels = {"version": self.spark_version}
        if self.driver_labels:
            driver_labels.update(self.driver_labels)
        driver_spec["labels"] = driver_labels

        if self.driver_annotations:
            driver_spec["annotations"] = self.driver_annotations

        # Driver pod template (Spark 3.0+)
        if self.driver_pod_template:
            driver_spec["template"] = self.driver_pod_template

        # Driver GPU
        if self.driver_gpu:
            driver_spec["gpu"] = {
                "name": self.driver_gpu.name,
                "quantity": self.driver_gpu.quantity,
            }

        # Driver volumes
        if self.driver_volume_mounts:
            driver_spec["volumeMounts"] = self.driver_volume_mounts

        # Driver environment
        if self.env_vars:
            driver_spec["env"] = [{"name": k, "value": v} for k, v in self.env_vars.items()]

        # Driver sidecars & init containers
        if self.driver_sidecars:
            driver_spec["sidecars"] = self.driver_sidecars
        if self.driver_init_containers:
            driver_spec["initContainers"] = self.driver_init_containers

        # Driver tolerations, affinity, security
        if self.tolerations:
            driver_spec["tolerations"] = self.tolerations
        if self.affinity:
            driver_spec["affinity"] = self.affinity
        if self.pod_security_context:
            driver_spec["podSecurityContext"] = self.pod_security_context
        if self.security_context:
            driver_spec["securityContext"] = self.security_context
        if self.host_network is not None:
            driver_spec["hostNetwork"] = self.host_network

        spec["driver"] = driver_spec

        # === Executor Spec ===
        executor_spec = {
            "cores": self.executor_cores,
            "instances": self.num_executors,
            "memory": self.executor_memory,
        }

        # Executor labels & annotations
        executor_labels = {"version": self.spark_version}
        if self.executor_labels:
            executor_labels.update(self.executor_labels)
        executor_spec["labels"] = executor_labels

        if self.executor_annotations:
            executor_spec["annotations"] = self.executor_annotations

        # Executor pod template (Spark 3.0+)
        if self.executor_pod_template:
            executor_spec["template"] = self.executor_pod_template

        # Executor GPU
        if self.executor_gpu:
            executor_spec["gpu"] = {
                "name": self.executor_gpu.name,
                "quantity": self.executor_gpu.quantity,
            }

        # Executor volumes
        if self.executor_volume_mounts:
            executor_spec["volumeMounts"] = self.executor_volume_mounts

        # Executor environment
        if self.env_vars:
            executor_spec["env"] = [{"name": k, "value": v} for k, v in self.env_vars.items()]

        # Executor sidecars & init containers
        if self.executor_sidecars:
            executor_spec["sidecars"] = self.executor_sidecars
        if self.executor_init_containers:
            executor_spec["initContainers"] = self.executor_init_containers

        # Executor tolerations, affinity, security (reuse from driver if not overridden)
        if self.tolerations:
            executor_spec["tolerations"] = self.tolerations
        if self.affinity:
            executor_spec["affinity"] = self.affinity
        if self.pod_security_context:
            executor_spec["podSecurityContext"] = self.pod_security_context
        if self.security_context:
            executor_spec["securityContext"] = self.security_context
        if self.host_network is not None:
            executor_spec["hostNetwork"] = self.host_network

        spec["executor"] = executor_spec

        # === Build final CRD ===
        return {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "SparkApplication",
            "metadata": metadata,
            "spec": spec,
        }


@dataclass
class SparkApplicationResponse:
    """Response model for Spark application submission.

    Attributes:
        submission_id: Unique submission ID generated by gateway
        app_name: Name of the application
        status: Current status of the application
        message: Additional message
    """

    submission_id: str
    app_name: str
    status: str = "SUBMITTED"
    message: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SparkApplicationResponse":
        """Create response from API response dictionary.

        Args:
            data: Dictionary from API response

        Returns:
            SparkApplicationResponse instance
        """
        return cls(
            submission_id=data.get("submissionId", data.get("submission_id", "")),
            app_name=data.get("appName", data.get("app_name", "")),
            status=data.get("status", "SUBMITTED"),
            message=data.get("message", ""),
        )


@dataclass
class SparkConnectServerConfig:
    """Configuration for creating a SparkConnect server via Spark Operator CRD (v1alpha1).

    This allows auto-provisioning of Spark Connect servers on Kubernetes for
    interactive sessions without requiring a pre-existing cluster.

    Attributes:
        # Basic Configuration
        name: Name of the SparkConnect server (auto-generated if not provided)
        namespace: Kubernetes namespace for the server
        spark_version: Spark version to use
        image: Container image for Spark (uses operator default if not set)

        # Server Resources (driver pod)
        server_cores: Number of cores for the server
        server_memory: Memory for the server (e.g., "2g")

        # Executor Resources
        executor_cores: Number of cores per executor
        executor_memory: Memory per executor (e.g., "4g")
        num_executors: Initial number of executors

        # Dynamic Allocation
        dynamic_allocation: Dynamic allocation configuration

        # Configuration
        spark_conf: Spark configuration properties
        hadoop_conf: Hadoop configuration properties
        env_vars: Environment variables

        # Kubernetes
        service_account: Service account for Spark pods
        node_selector: Node selector for scheduling
        tolerations: Kubernetes tolerations
        volumes: Volume configurations
        volume_mounts: Volume mount configurations

        # Labels & Annotations
        labels: Kubernetes labels
        annotations: Kubernetes annotations
    """

    # Basic Configuration
    name: Optional[str] = None
    namespace: str = "default"
    spark_version: str = "3.5.0"
    image: Optional[str] = None

    # Server Resources
    server_cores: int = 1
    server_memory: str = "1g"

    # Executor Resources
    executor_cores: int = 1
    executor_memory: str = "1g"
    num_executors: int = 2

    # Dynamic Allocation
    dynamic_allocation: Optional[DynamicAllocation] = None

    # Configuration
    spark_conf: dict[str, str] = field(default_factory=dict)
    hadoop_conf: dict[str, str] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)

    # Kubernetes
    service_account: str = "spark-operator-spark"
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict[str, Any]] = field(default_factory=list)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    volume_mounts: list[dict[str, Any]] = field(default_factory=list)

    # Labels & Annotations
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)


@dataclass
class SparkConnectServerStatus:
    """Status of a SparkConnect server.

    Attributes:
        name: Name of the SparkConnect server
        namespace: Kubernetes namespace
        state: Current state of the server
        connect_url: Spark Connect URL (sc://host:port)
        service_name: Kubernetes service name
        pod_name: Server pod name
        pod_ip: Server pod IP address
        creation_time: Server creation timestamp
        message: Status message or error details
    """

    name: str
    namespace: str
    state: SparkConnectState = SparkConnectState.UNKNOWN
    connect_url: Optional[str] = None
    service_name: Optional[str] = None
    pod_name: Optional[str] = None
    pod_ip: Optional[str] = None
    creation_time: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def from_crd(cls, crd: dict[str, Any]) -> "SparkConnectServerStatus":
        """Create status from SparkConnect CRD.

        Args:
            crd: SparkConnect CRD dictionary

        Returns:
            SparkConnectServerStatus instance
        """
        metadata = crd.get("metadata", {})
        status = crd.get("status", {})
        server_status = status.get("server", {})

        # Parse state
        state_str = status.get("state", "Unknown")
        try:
            state = SparkConnectState(state_str)
        except ValueError:
            state = SparkConnectState.UNKNOWN

        # Build connect URL from service info
        connect_url = None
        service_name = server_status.get("serviceName")
        namespace = metadata.get("namespace", "default")
        if service_name:
            connect_url = f"sc://{service_name}.{namespace}.svc:15002"

        return cls(
            name=metadata.get("name", ""),
            namespace=namespace,
            state=state,
            connect_url=connect_url,
            service_name=service_name,
            pod_name=server_status.get("podName"),
            pod_ip=server_status.get("podIp"),
            creation_time=metadata.get("creationTimestamp"),
            message=status.get("message"),
        )


@dataclass
class ConnectBackendConfig:
    """Configuration for Spark Connect backend.

    This backend enables remote connectivity to existing Spark clusters via
    Spark Connect protocol (gRPC-based).

    Attributes:
        connect_url: Spark Connect URL (format: sc://host:port/;param1=value;param2=value)
        token: Bearer token for authentication (enables SSL automatically)
        use_ssl: Enable TLS/SSL for secure communication
        user_id: User identifier for session management
        session_id: Pre-defined session UUID for session sharing
        grpc_max_message_size: Maximum gRPC message size in bytes

        # Auto-provisioning (for Kubeflow-managed clusters)
        enable_auto_provision: Automatically provision Spark Connect server if not exists
        auto_provision_config: SparkApplication config for auto-provisioned server
        namespace: Kubernetes namespace for auto-provisioned server

        # Kubeflow integration
        enable_monitoring: Enable metrics collection
        artifact_staging_path: Path for staging artifacts (JARs, files, etc.)
        timeout: Default timeout for operations in seconds
    """

    connect_url: str
    token: Optional[str] = None
    use_ssl: bool = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    grpc_max_message_size: int = 128 * 1024 * 1024  # 128MB default

    # Auto-provisioning
    enable_auto_provision: bool = False
    auto_provision_config: Optional["SparkApplicationRequest"] = None
    namespace: str = "default"

    # Kubeflow integration
    enable_monitoring: bool = True
    artifact_staging_path: Optional[str] = None
    timeout: int = 300


@dataclass
class SessionMetrics:
    """Metrics for a Spark Connect session.

    Attributes:
        session_id: Session UUID
        queries_executed: Number of queries executed
        active_queries: Number of currently active queries
        artifacts_uploaded: Number of artifacts uploaded
        data_read_bytes: Total bytes read
        data_written_bytes: Total bytes written
        execution_time_ms: Total execution time in milliseconds
    """

    session_id: str
    queries_executed: int = 0
    active_queries: int = 0
    artifacts_uploaded: int = 0
    data_read_bytes: int = 0
    data_written_bytes: int = 0
    execution_time_ms: int = 0


@dataclass
class SessionInfo:
    """Information about a Spark Connect session.

    Attributes:
        session_id: Session UUID
        app_name: Application name
        user_id: User identifier
        created_at: Session creation time
        last_activity: Last activity timestamp
        state: Session state (active, idle, closed)
        metrics: Session metrics
    """

    session_id: str
    app_name: str
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    last_activity: Optional[str] = None
    state: str = "active"
    metrics: Optional[SessionMetrics] = None


@dataclass
class ApplicationStatus:
    """Status information for a Spark application.

    Attributes:
        submission_id: Submission ID
        app_id: Spark application ID
        app_name: Application name
        state: Current state
        submission_time: Time of submission
        start_time: Start time
        completion_time: Completion time
        driver_info: Driver pod information
        executor_state: Executor states
    """

    submission_id: str
    app_id: Optional[str] = None
    app_name: Optional[str] = None
    state: ApplicationState = ApplicationState.UNKNOWN
    submission_time: Optional[str] = None
    start_time: Optional[str] = None
    completion_time: Optional[str] = None
    driver_info: Optional[dict[str, Any]] = None
    executor_state: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApplicationStatus":
        """Create status from API response dictionary.

        Args:
            data: Dictionary from API response

        Returns:
            ApplicationStatus instance
        """
        # Support both Operator and Gateway response formats
        if "status" in data and "applicationState" in data.get("status", {}):
            # Operator format
            state_str = data["status"]["applicationState"].get("state", "UNKNOWN")
            app_id = data["status"].get("sparkApplicationId")
            submission_time = data["status"].get("submissionTime")
            start_time = data["status"].get("lastSubmissionAttemptTime")
            completion_time = data["status"].get("terminationTime")
            driver_info = data["status"].get("driverInfo")
            executor_state = data["status"].get("executorState")
        elif "status" in data and "appState" in data.get("status", {}):
            # Operator format (alternative field name)
            state_str = data["status"].get("appState", {}).get("state", "UNKNOWN")
            app_id = data["status"].get("sparkApplicationId")
            submission_time = None
            start_time = data["status"].get("lastSubmissionAttemptTime")
            completion_time = data["status"].get("terminationTime")
            driver_info = data["status"].get("driverInfo")
            executor_state = data["status"].get("executorState")
        else:
            # Gateway format or simple format
            state_str = data.get("status", "UNKNOWN")
            app_id = data.get("app_id")
            submission_time = data.get("submission_time")
            start_time = data.get("start_time")
            completion_time = data.get("completion_time")
            driver_info = data.get("driver_info")
            executor_state = data.get("executor_state")

        try:
            state = ApplicationState(state_str)
        except ValueError:
            state = ApplicationState.UNKNOWN

        return cls(
            submission_id=data.get(
                "submissionId", data.get("submission_id", data.get("metadata", {}).get("name", ""))
            ),
            app_id=app_id,
            app_name=data.get("metadata", {}).get("name", data.get("app_name")),
            state=state,
            submission_time=submission_time,
            start_time=start_time,
            completion_time=completion_time,
            driver_info=driver_info,
            executor_state=executor_state,
        )
