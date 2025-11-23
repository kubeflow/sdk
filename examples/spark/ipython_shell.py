#!/usr/bin/env python3
"""
Launch IPython shell with Kubeflow SDK in dev mode.
Usage: ./ipython_shell.py

Requires IPython: pip install ipython
"""

import os
import sys

# Add SDK to path
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, sdk_path)

# Pre-import common modules

# Print welcome message
banner = f"""
{"=" * 80}
Kubeflow Spark Client - IPython Development Shell
{"=" * 80}

SDK Path: {sdk_path}

Pre-imported:
  BatchSparkClient, OperatorBackendConfig, GatewayBackendConfig,
  ApplicationState, ApplicationStatus, SparkApplicationResponse

Quick Examples:
  config = OperatorBackendConfig(namespace="default")
  client = SparkClient(backend_config=config)

Tab completion and syntax highlighting enabled!
{"=" * 80}
"""

try:
    import IPython

    IPython.embed(banner1=banner, colors="Linux")
except ImportError:
    print("IPython not installed. Install with: pip install ipython")
    print("Falling back to regular Python shell...\n")
    import code

    print(banner)
    code.interact(local=locals())
