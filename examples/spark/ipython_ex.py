# ./ipython_shell.py
# then run /paste this python code .

import os

from kubeflow.spark import OperatorBackendConfig, SparkClient

config = OperatorBackendConfig(
    namespace=os.getenv("SPARK_NAMESPACE", "default"),
    service_account="spark-operator-spark",
    default_spark_image="docker.io/library/spark",
    enable_monitoring=False,
    enable_ui=False,
    context=os.getenv("KUBE_CONTEXT", "kind-spark-test"),  # Explicitly set context
)
client = BatchSparkClient(backend_config=config)

app_name = "test-spark-pi"

response = client.submit_application(
    app_name=app_name,
    main_application_file="local:///opt/spark/examples/jars/spark-examples_2.13-4.0.0.jar",
    main_class="org.apache.spark.examples.SparkPi",
    spark_version="4.0.0",
    app_type="Scala",
    driver_cores=1,
    driver_memory="512m",
    executor_cores=1,
    executor_memory="512m",
    num_executors=1,
    arguments=["100"],
    spark_conf={
        "spark.kubernetes.file.upload.path": "/tmp",  # Required for Spark 4.0
    },
)
client.get_status(app_name)


response = client.submit_application(
    app_name="my-python-pi5",
    main_application_file="local:///opt/spark/examples/src/main/python/pi.py",
    spark_version="4.0.0",
    app_type="Python",
    driver_cores=1,
    driver_memory="512m",
    executor_cores=1,
    executor_memory="512m",
    num_executors=2,
    arguments=["100"],
    spark_conf={"spark.kubernetes.file.upload.path": "/tmp"},
)
final_status = client.wait_for_completion("my-python-pi5", timeout=300)
client.get_status("my-python-pi5")
logs = list(client.get_logs("my-python-pi5"))
for line in logs:
    if "Pi is roughly" in line:
        print(f"RESULT: {line}")


response = client.submit_application(
    app_name="my-python-pi6",
    main_application_file="local:///opt/spark/examples/src/main/python/pi.py",
    spark_version="4.0.0",
    app_type="Python",
    driver_cores=1,
    driver_memory="512m",
    executor_cores=1,
    executor_memory="512m",
    num_executors=2,
    arguments=["100"],
    spark_conf={
        "spark.kubernetes.file.upload.path": "/tmp",
        "spark.eventLog.enabled": "true",
        "spark.eventLog.dir": "/tmp/spark-events",
    },
    volumes=[{"name": "spark-events", "persistentVolumeClaim": {"claimName": "spark-history-pvc"}}],
    driver_volume_mounts=[{"name": "spark-events", "mountPath": "/tmp/spark-events"}],
    executor_volume_mounts=[{"name": "spark-events", "mountPath": "/tmp/spark-events"}],
)
final_status = client.wait_for_completion("my-python-pi6", timeout=300)
client.get_status("my-python-pi6")
logs = list(client.get_logs("my-python-pi6"))
for line in logs:
    if "Pi is roughly" in line:
        print(f"RESULT: {line}")
