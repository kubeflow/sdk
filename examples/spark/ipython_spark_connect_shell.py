#!/usr/bin/env python3
"""
IPython Shell Launcher for Spark Connect Demo

This script launches an IPython shell with the Kubeflow SDK pre-imported
and prints step-by-step instructions for testing Spark Connect.

Usage: python ipython_spark_connect_shell.py
"""

import os
import sys

# Add SDK to path
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, sdk_path)

# Pre-import for convenience

# Banner with instructions
banner = f"""
{"=" * 80}
Kubeflow Spark Connect - Interactive IPython Shell
{"=" * 80}

SDK Path: {sdk_path}
Spark Connect URL: sc://localhost:30000

Pre-imported modules:
  - ConnectBackendConfig
  - SparkSessionClient

{"=" * 80}
Step-by-Step Guide
{"=" * 80}

1. Create Configuration:
   config = ConnectBackendConfig(
       connect_url="sc://localhost:30000",
       use_ssl=False,
       timeout=60
   )

2. Create Client and Session:
   client = SparkSessionClient(backend_config=config)
   session = client.create_session(app_name="my-demo")

3. Run Simple SQL:
   df = session.sql("SELECT 1 AS id, 'Hello Spark Connect' AS message")
   df.show()

4. Create DataFrame from Python Data:
   sales_data = [
       (1, "Electronics", "Laptop", 1200.00, 2),
       (2, "Electronics", "Mouse", 25.00, 5),
       (3, "Clothing", "Shirt", 35.00, 3),
       (4, "Electronics", "Keyboard", 75.00, 4),
       (5, "Clothing", "Pants", 55.00, 2),
   ]
   df = session.createDataFrame(
       sales_data,
       ["id", "category", "product", "price", "quantity"]
   )
   df.show()

5. Filter and Select:
   expensive = df.filter(df.price > 50)
   expensive.show()

   df.select("category", "product", "price").show()

6. GroupBy Aggregations:
   from pyspark.sql import functions as F

   revenue_df = df.withColumn("revenue", F.col("price") * F.col("quantity"))

   category_stats = revenue_df.groupBy("category").agg(
       F.sum("revenue").alias("total_revenue"),
       F.avg("price").alias("avg_price"),
       F.count("*").alias("num_transactions")
   )
   category_stats.show()

7. Order Results:
   category_stats.orderBy(F.desc("total_revenue")).show()

8. Session Metrics:
   metrics = session.get_metrics()
   print(f"Queries executed: {{metrics.queries_executed}}")

   info = session.get_info()
   print(f"Session state: {{info.state}}")

9. Clean Up (when done):
   session.close()
   client.close()

{"=" * 80}
Ready! Start by copying and pasting the commands above.
{"=" * 80}
"""

if __name__ == "__main__":
    try:
        import IPython

        IPython.embed(banner1=banner, colors="Linux")
    except ImportError:
        print("IPython not installed. Install with: pip install ipython")
        print("Falling back to regular Python shell...\n")
        import code

        print(banner)
        code.interact(local=locals())
