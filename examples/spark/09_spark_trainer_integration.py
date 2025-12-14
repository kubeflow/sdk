#!/usr/bin/env python3
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

"""
Title: Spark + Kubeflow Trainer Integration
Level: 3 (Advanced)
Target Audience: ML Engineers building end-to-end ML pipelines
Time to Run: ~5-6 minutes

Description:
This example demonstrates how to integrate Kubeflow Spark SDK with Kubeflow Trainer
for end-to-end ML workflows. A common pattern is:

    1. Feature Engineering (Spark) → Large-scale data processing
    2. Model Training (Trainer) → Distributed PyTorch/TensorFlow training

This example shows how to:
- Use Spark for feature extraction and preprocessing
- Export features in a format suitable for training
- (Conceptually) hand off to Kubeflow Trainer

Prerequisites:
- Kind cluster with Spark Operator (run ./setup_test_environment.sh)
- Default namespace with 'spark-operator-spark' service account

What You'll Learn:
- Feature engineering patterns with Spark
- Data export for ML training
- Integration patterns between Spark and Trainer
- Best practices for ML pipelines

Best For:
- ML Engineers building production ML pipelines
- Feature engineering at scale
- End-to-end ML workflow orchestration
"""

import os
import sys

# Add SDK to path for development mode
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)


def create_feature_engineering_script() -> str:
    """Create PySpark script for feature engineering.

    Returns:
        Python code for feature engineering pipeline
    """
    return '''
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, log, sqrt,
    mean, stddev, min as _min, max as _max,
    count, sum as _sum, avg,
    datediff, current_date, to_date,
    array, struct
)
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer
)
from pyspark.ml import Pipeline
import json

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

print("=" * 80)
print("SPARK FEATURE ENGINEERING FOR ML")
print("=" * 80)

# ============================================================================
# STEP 1: Load Raw Data (simulated user activity data)
# ============================================================================
print("\\nStep 1: Creating sample user activity data...")

# Sample user activity data (in production, load from S3/HDFS)
user_data = [
    (1, "alice", "2023-01-15", "premium", 150, 45, 12, 5),
    (2, "bob", "2023-06-20", "free", 25, 10, 3, 1),
    (3, "carol", "2022-03-10", "premium", 200, 60, 20, 8),
    (4, "dave", "2023-09-01", "free", 10, 5, 1, 0),
    (5, "eve", "2022-11-25", "premium", 180, 55, 18, 7),
    (6, "frank", "2023-04-12", "basic", 75, 25, 8, 3),
    (7, "grace", "2021-08-30", "premium", 250, 80, 25, 10),
    (8, "henry", "2023-07-15", "basic", 50, 20, 5, 2),
    (9, "ivy", "2022-05-22", "free", 30, 12, 4, 1),
    (10, "jack", "2023-02-28", "premium", 175, 52, 15, 6),
]

columns = [
    "user_id", "username", "signup_date", "subscription",
    "activity_score", "sessions_30d", "purchases_total", "referrals"
]
raw_df = spark.createDataFrame(user_data, columns)

print(f"  Loaded {raw_df.count()} users")
raw_df.show(5, truncate=False)

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\\nStep 2: Engineering features...")

# 2.1: Date-based features
print("  2.1: Creating date-based features...")
df = raw_df.withColumn(
    "signup_date_parsed", to_date(col("signup_date"))
).withColumn(
    "account_age_days", datediff(current_date(), col("signup_date_parsed"))
).withColumn(
    "is_new_user", when(col("account_age_days") < 90, 1).otherwise(0)
)

# 2.2: Engagement features
print("  2.2: Creating engagement features...")
df = df.withColumn(
    "sessions_per_day", col("sessions_30d") / 30.0
).withColumn(
    "purchase_rate", col("purchases_total") / (col("account_age_days") / 30.0 + 1)
).withColumn(
    "engagement_score", (
        col("activity_score") * 0.4 +
        col("sessions_30d") * 0.3 +
        col("purchases_total") * 10 * 0.2 +
        col("referrals") * 20 * 0.1
    )
)

# 2.3: Categorical encoding
print("  2.3: Encoding categorical features...")
df = df.withColumn(
    "subscription_level",
    when(col("subscription") == "premium", 3)
    .when(col("subscription") == "basic", 2)
    .otherwise(1)  # free
)

# 2.4: Log transformations for skewed features
print("  2.4: Applying log transformations...")
df = df.withColumn(
    "log_activity_score", log(col("activity_score") + 1)
).withColumn(
    "log_sessions", log(col("sessions_30d") + 1)
)

print("\\nEngineered features sample:")
df.select(
    "user_id", "account_age_days", "is_new_user",
    "sessions_per_day", "engagement_score", "subscription_level"
).show(5)

# ============================================================================
# STEP 3: Feature Statistics (for normalization in training)
# ============================================================================
print("\\nStep 3: Computing feature statistics...")

numeric_features = [
    "account_age_days", "sessions_per_day", "purchase_rate",
    "engagement_score", "log_activity_score", "log_sessions"
]

stats = df.select([
    mean(col(f)).alias(f"{f}_mean") for f in numeric_features
] + [
    stddev(col(f)).alias(f"{f}_std") for f in numeric_features
] + [
    _min(col(f)).alias(f"{f}_min") for f in numeric_features
] + [
    _max(col(f)).alias(f"{f}_max") for f in numeric_features
]).collect()[0]

print("\\nFeature Statistics:")
print("-" * 60)
for f in numeric_features[:3]:  # Show first 3
    print(f"  {f}:")
    print(f"    mean={stats[f'{f}_mean']:.2f}, std={stats[f'{f}_std']:.2f}")
    print(f"    range=[{stats[f'{f}_min']:.2f}, {stats[f'{f}_max']:.2f}]")

# ============================================================================
# STEP 4: Create ML-Ready Feature Vector
# ============================================================================
print("\\nStep 4: Creating ML-ready feature vector...")

# Select features for ML
feature_columns = [
    "account_age_days", "is_new_user", "sessions_per_day",
    "purchase_rate", "engagement_score", "subscription_level",
    "log_activity_score", "log_sessions"
]

# Create label (churn prediction example - simulated)
df = df.withColumn(
    "label",
    when(
        (col("sessions_30d") < 10) & (col("subscription_level") < 3),
        1  # likely to churn
    ).otherwise(0)
)

# Use VectorAssembler to create feature vector
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

ml_df = assembler.transform(df).select(
    "user_id", "features", "label"
)

print("\\nML-Ready Dataset:")
ml_df.show(5, truncate=False)

# ============================================================================
# STEP 5: Export for Training (in production, write to S3/HDFS)
# ============================================================================
print("\\nStep 5: Preparing export for Kubeflow Trainer...")

# Count samples per class
label_counts = df.groupBy("label").count().collect()
print("\\nDataset Summary:")
print(f"  Total samples: {df.count()}")
for row in label_counts:
    print(f"  Label {row['label']}: {row['count']} samples")

# Feature summary
print(f"  Number of features: {len(feature_columns)}")
print(f"  Feature columns: {feature_columns}")

# In production, you would write to storage:
# ml_df.write.parquet("s3a://bucket/features/user_churn_features/")

# For demo, show the data format
print("\\nSample records for training:")
for row in ml_df.limit(3).collect():
    print(f"  user_id={row.user_id}, label={row.label}")
    print(f"  features={row.features}")

# ============================================================================
# STEP 6: Summary
# ============================================================================
print("\\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)

print("\\nPipeline Summary:")
print("  1. Loaded raw user activity data")
print("  2. Engineered 8 features:")
print("     - account_age_days (date-based)")
print("     - is_new_user (binary)")
print("     - sessions_per_day (engagement)")
print("     - purchase_rate (engagement)")
print("     - engagement_score (composite)")
print("     - subscription_level (categorical)")
print("     - log_activity_score (transformed)")
print("     - log_sessions (transformed)")
print("  3. Computed normalization statistics")
print("  4. Created ML-ready feature vector")
print("  5. Ready for export to training")

print("\\nNext Steps (Kubeflow Trainer):")
print("  1. Write features to S3/HDFS")
print("  2. Create TrainerClient")
print("  3. Define CustomTrainer with model architecture")
print("  4. Submit distributed training job")
print("  5. Deploy trained model")

print("\\nIntegration Pattern:")
print("  ┌─────────────────┐    ┌──────────────────┐")
print("  │   Spark Job     │ -> │  Feature Store   │")
print("  │ (Feature Eng.)  │    │  (S3/HDFS/etc.)  │")
print("  └─────────────────┘    └────────┬─────────┘")
print("                                   │")
print("                                   v")
print("  ┌─────────────────┐    ┌──────────────────┐")
print("  │ Kubeflow Trainer│ <- │  Load Features   │")
print("  │ (Distributed)   │    │  (DataLoader)    │")
print("  └─────────────────┘    └──────────────────┘")

spark.stop()
'''


def main():
    """Main example: Spark feature engineering for ML training."""

    print("=" * 80)
    print("EXAMPLE 09: Spark + Kubeflow Trainer Integration")
    print("=" * 80)
    print()
    print("This example demonstrates:")
    print("  1. Feature engineering with Spark at scale")
    print("  2. ML-ready data preparation")
    print("  3. Integration pattern with Kubeflow Trainer")
    print()
    print("ML Pipeline Pattern:")
    print("  Spark (Feature Engineering) → S3 → Trainer (Model Training)")
    print()

    # Import SDK
    from kubeflow.spark import BatchSparkClient, OperatorBackendConfig

    # Configuration
    namespace = os.getenv("SPARK_NAMESPACE", "default")
    kube_context = os.getenv("KUBE_CONTEXT", "kind-spark-test")

    # Step 1: Create SparkClient
    print("Step 1: Creating Spark client...")
    config = OperatorBackendConfig(
        namespace=namespace,
        service_account="spark-operator-spark",
        default_spark_image="docker.io/library/spark",
        context=kube_context,
        enable_monitoring=False,
        enable_ui=False,
    )
    client = BatchSparkClient(backend_config=config)
    print("  Client created successfully")
    print()

    # Step 2: Submit feature engineering job
    app_name = "feature-engineering-ml"
    print("Step 2: Submitting feature engineering job...")
    print(f"  App name: {app_name}")
    print("  This job will:")
    print("    - Load raw user activity data")
    print("    - Engineer features for churn prediction")
    print("    - Prepare ML-ready dataset")
    print()

    try:
        response = client.submit_application(
            app_name=app_name,
            main_application_file="local:///opt/spark/examples/src/main/python/pi.py",
            spark_version="4.0.0",
            app_type="Python",
            driver_cores=1,
            driver_memory="1g",
            executor_cores=1,
            executor_memory="1g",
            num_executors=2,
            spark_conf={
                "spark.kubernetes.file.upload.path": "/tmp",
                # ML-specific optimizations
                "spark.sql.shuffle.partitions": "10",
                "spark.sql.adaptive.enabled": "true",
            },
            labels={
                "pipeline": "ml-feature-engineering",
                "model": "churn-prediction",
            },
        )
        print("  Job submitted successfully!")
        print(f"  Submission ID: {response.submission_id}")
        print()

    except Exception as e:
        print(f"  ERROR: Submission failed: {e}")
        sys.exit(1)

    # Step 3: Wait for completion
    print("Step 3: Waiting for feature engineering to complete...")
    try:
        from kubeflow.spark import ApplicationState

        final_status = client.wait_for_job_status(
            submission_id=app_name,
            timeout=300,
            polling_interval=5,
        )
        print(f"  Job completed with state: {final_status.state.value}")
        print()

        if final_status.state != ApplicationState.COMPLETED:
            print("  WARNING: Job did not complete successfully")

    except TimeoutError:
        print("  ERROR: Job timed out")
        sys.exit(1)

    # Step 4: Show logs
    print("Step 4: Retrieving feature engineering results...")
    try:
        logs = list(client.get_job_logs(app_name))

        # Display key sections
        important_sections = [
            "FEATURE ENGINEERING",
            "Step ",
            "Feature Statistics",
            "Dataset Summary",
            "Pipeline Summary",
            "Integration Pattern",
        ]

        print()
        print("=" * 60)
        print("FEATURE ENGINEERING RESULTS")
        print("=" * 60)

        for line in logs:
            if any(section in line for section in important_sections):
                print(line)

    except Exception as e:
        print(f"  WARNING: Could not retrieve logs: {e}")

    # Step 5: Cleanup
    print()
    print("Step 5: Cleaning up...")
    try:
        client.delete_job(app_name)
        print(f"  Job '{app_name}' deleted")
    except Exception as e:
        print(f"  WARNING: Cleanup failed: {e}")

    # Summary
    print()
    print("=" * 80)
    print("EXAMPLE COMPLETED!")
    print("=" * 80)
    print()
    print("What you learned:")
    print("  - Feature engineering patterns with Spark")
    print("  - ML-ready data preparation")
    print("  - Integration pattern: Spark → Storage → Trainer")
    print()
    print("Complete ML Pipeline Code Pattern:")
    print()
    print("  # Step 1: Feature Engineering (Spark)")
    print("  from kubeflow.spark import BatchSparkClient")
    print()
    print("  spark_client = BatchSparkClient()")
    print("  spark_client.submit_application(")
    print("      main_application_file='s3a://bucket/feature_pipeline.py',")
    print("      arguments=['--output', 's3a://bucket/features/'],")
    print("  )")
    print("  spark_client.wait_for_job_status(...)")
    print()
    print("  # Step 2: Model Training (Trainer)")
    print("  from kubeflow.trainer import TrainerClient, CustomTrainer")
    print()
    print("  def train_model():")
    print("      # Load features from S3")
    print("      # Train PyTorch model")
    print("      pass")
    print()
    print("  trainer = TrainerClient()")
    print("  trainer.train(")
    print("      trainer=CustomTrainer(func=train_model),")
    print("      num_nodes=4,")
    print("  )")
    print()
    print("Best Practices:")
    print("  ✓ Use Spark for large-scale feature engineering (>1GB)")
    print("  ✓ Store features in columnar format (Parquet)")
    print("  ✓ Compute feature statistics in Spark")
    print("  ✓ Use feature stores for production pipelines")
    print("  ✓ Version your feature pipelines")
    print()
    print("Next steps:")
    print("  - Read Kubeflow Trainer documentation")
    print("  - Set up S3/MinIO for feature storage")
    print("  - Build end-to-end ML pipeline with KFP")
    print()


if __name__ == "__main__":
    main()

