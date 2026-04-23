# Spark Examples

This directory contains examples for using the Kubeflow Spark SDK.

## Examples

- **spark_connect_simple.py** - Basic SparkClient usage with simple API
- **spark_advanced_options.py** - Advanced configuration with Driver/Executor objects
- **demo_existing_sparkconnect.py** - Connect to existing SparkConnect cluster
- **test_connect_url.py** - Test URL-based connection to Spark Connect

## Prerequisites

Install spark dependencies:
```bash
uv pip install kubeflow[spark]
```

## Running Examples

```bash
# Run from repository root
uv run python examples/spark/spark_connect_simple.py
```
