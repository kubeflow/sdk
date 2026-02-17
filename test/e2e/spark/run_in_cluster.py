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

"""Run Spark example scripts inside the cluster as a Job (in-cluster URL, no port-forward)."""

import contextlib
import os
import subprocess
import tempfile


def run_example_in_cluster(
    example_script_name: str,
    namespace: str,
    image: str,
    timeout_sec: int = 300,
) -> tuple[bool, str, str]:
    """Run an example script in-cluster via a Kubernetes Job.

    The Job pod has KUBERNETES_SERVICE_HOST set, so SparkClient uses the
    in-cluster URL (sc://...svc.cluster.local) and no port-forward.

    Args:
        example_script_name: Script filename (e.g. spark_connect_simple.py).
        namespace: Kubernetes namespace for the Job.
        image: Container image that has the SDK and examples (e.g. spark-e2e-runner:local).
        timeout_sec: Max time to wait for Job completion.

    Returns:
        (success, combined_stdout_stderr, job_description).
    """
    base = example_script_name.replace(".py", "").replace("_", "-")
    job_name = f"spark-e2e-{base}"[:63].rstrip("-")
    script_path = f"examples/spark/{example_script_name}"
    # Job manifest: one pod, run the example script, default SA (has e2e-sparkconnect-client Role).
    manifest = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: {timeout_sec}
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: runner
          image: {image}
          imagePullPolicy: IfNotPresent
          command:
            - python
            - {script_path}
          env:
            - name: SPARK_TEST_NAMESPACE
              value: "{namespace}"
            - name: SPARK_E2E_RUN_IN_CLUSTER
              value: "1"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(manifest)
        manifest_path = f.name

    subprocess.run(
        ["kubectl", "delete", "job", job_name, "-n", namespace, "--ignore-not-found=true"],
        capture_output=True,
        timeout=15,
    )
    apply_result = subprocess.run(
        ["kubectl", "apply", "-f", manifest_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if apply_result.returncode != 0:
        with contextlib.suppress(Exception):
            os.unlink(manifest_path)
        err = (apply_result.stderr or "").strip() or (apply_result.stdout or "").strip()
        return False, "", f"Failed to create Job: {err or apply_result.returncode}"

    wait_result = subprocess.run(
        [
            "kubectl",
            "wait",
            "--for=condition=complete",
            f"job/{job_name}",
            "-n",
            namespace,
            f"--timeout={timeout_sec}s",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_sec + 30,
    )
    wait_stderr = (wait_result.stderr or "").strip()
    if wait_result.returncode != 0:
        succeeded = False
        failed = True
    else:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "job",
                job_name,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.succeeded},{.status.failed}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = (result.stdout or "").strip() if result.returncode == 0 else "0,0"
        parts = out.split(",")
        succeeded = (parts[0] or "0") == "1"
        failed = (parts[1] or "0") != "0"

    # Get logs from the Job pod
    pod_result = subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"job-name={job_name}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    pod_name = (pod_result.stdout or "").strip() if pod_result.returncode == 0 else ""
    no_pod_extra = ""
    if not pod_name:
        pods_list = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-l", f"job-name={job_name}", "-o", "wide"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if pods_list.returncode == 0 and (pods_list.stdout or pods_list.stderr):
            no_pod_extra = (
                "\n--- Pods (job-name) ---\n" + (pods_list.stdout or "") + (pods_list.stderr or "")
            )
    logs = ""
    if pod_name:
        log_result = subprocess.run(
            ["kubectl", "logs", pod_name, "-n", namespace, "--tail=500"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        logs = (log_result.stdout or "") + (log_result.stderr or "")
        if not logs.strip():
            prev_result = subprocess.run(
                ["kubectl", "logs", pod_name, "-n", namespace, "--tail=500", "--previous"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if prev_result.returncode == 0 and (prev_result.stdout or prev_result.stderr):
                logs = (
                    "(previous container)\n"
                    + (prev_result.stdout or "")
                    + (prev_result.stderr or "")
                )

    # Job description for debugging
    desc_result = subprocess.run(
        ["kubectl", "describe", "job", job_name, "-n", namespace],
        capture_output=True,
        text=True,
        timeout=15,
    )
    job_desc = desc_result.stdout or ""
    if wait_stderr:
        job_desc = f"--- kubectl wait stderr ---\n{wait_stderr}\n\n{job_desc}"
    if pod_name:
        pod_desc_result = subprocess.run(
            ["kubectl", "describe", "pod", pod_name, "-n", namespace],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if pod_desc_result.returncode == 0 and pod_desc_result.stdout:
            job_desc = job_desc + "\n--- Pod describe ---\n" + pod_desc_result.stdout
    if no_pod_extra:
        job_desc = job_desc + no_pod_extra

    with contextlib.suppress(Exception):
        os.unlink(manifest_path)

    subprocess.run(
        ["kubectl", "delete", "job", job_name, "-n", namespace, "--ignore-not-found=true"],
        capture_output=True,
        timeout=15,
    )
    subprocess.run(
        [
            "kubectl",
            "delete",
            "sparkconnect",
            "--all",
            "-n",
            namespace,
            "--ignore-not-found=true",
            "--wait=false",
        ],
        capture_output=True,
        timeout=30,
    )

    success = succeeded and not failed
    return success, logs, job_desc
