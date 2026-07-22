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

"""Simple Spark job that intentionally fails for invalid input."""

import argparse

from pyspark.sql import SparkSession


def main() -> None:
    """Run a simple Spark job."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "samples",
        type=int,
        help="Number of samples.",
    )
    args = parser.parse_args()

    spark = SparkSession.builder.appName("spark-failed-job").getOrCreate()

    print(f"Running with {args.samples} samples.")

    spark.range(args.samples).count()

    spark.stop()


if __name__ == "__main__":
    main()
