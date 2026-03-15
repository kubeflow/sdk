#!/usr/bin/env python3
"""PySpark word count — counts word occurrences in sample text data.

Accepts an optional --verbose flag to print per-word counts.
"""

import sys

from pyspark.sql import SparkSession


def main() -> None:
    """Run word count on sample text and print results."""
    verbose = "--verbose" in sys.argv

    spark = SparkSession.builder.appName("PySpark-WordCount").getOrCreate()

    text_data = [
        "simple Python SDK to run Spark on Kubernetes",
        "The SDK provides SparkClient with connect and submit job APIs",
        "connect API creates new Spark Connect sessions or connects to existing servers",
        "auto provisions Spark Connect servers when configuration is provided",
        "connects to existing Spark Connect servers when URL is provided",
        "auto cleans up resources on exit",
        "batch job support submit and manage SparkApplication jobs via submit job",
    ]

    word_counts = (
        spark.sparkContext.parallelize(text_data)
        .flatMap(lambda line: line.lower().split())
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda x: x[1], ascending=False)
    )

    results = word_counts.collect()
    print(f"\nWord count complete: {len(results)} unique words")

    if verbose:
        print("=" * 40)
        for word, count in results:
            print(f"  {word}: {count}")
        print("=" * 40)

    spark.stop()


if __name__ == "__main__":
    main()
