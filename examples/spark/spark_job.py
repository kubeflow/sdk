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


"""Simple Spark application for SDK batch job examples."""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("KubeflowSparkExample").getOrCreate()

# Create a small DataFrame
df = spark.range(10).withColumn("square", col("id") * col("id"))

print("Input DataFrame:")
df.show()

count = df.count()
total = df.agg({"square": "sum"}).collect()[0][0]

print(f"Row count: {count}")
print(f"Sum of squares: {total}")

spark.stop()
