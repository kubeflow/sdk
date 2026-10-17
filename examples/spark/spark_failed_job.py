#!/usr/bin/env python3
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

"""Simple Spark application for SDK batch failed job examples."""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a Spark session for the application.
spark = SparkSession.builder.appName("KubeflowSparkFailureExample").getOrCreate()

try:
    # Create a small DataFrame.
    df = spark.range(10).withColumn("square", col("id") * col("id"))

    print("Input DataFrame:")
    df.show()

    # Trigger a Spark action to ensure the job executes successfully
    # before intentionally failing.
    count = df.count()
    print(f"Row count: {count}")

    print("Intentionally failing the Spark application...")

    # Raise an exception to intentionally fail the Spark application.
    raise RuntimeError("Intentional failure for Spark SDK E2E testing")

finally:
    # Stop the Spark session before exiting.
    spark.stop()
