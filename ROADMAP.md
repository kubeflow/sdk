# Kubeflow SDK ROADMAP

## 2026

### Core SDK
- Add Structured Logging Support: https://github.com/kubeflow/sdk/issues/85
- Integrate Kubeflow SDK with OpenTelemetry: https://github.com/kubeflow/sdk/issues/164
- Experiment Tracking with MLflow for Kubeflow SDK: https://github.com/kubeflow/sdk/issues/63
- Kubeflow SDK MCP Server: https://github.com/kubeflow/community/pull/937
- Improve Kubeflow SDK Documentation: https://github.com/kubeflow/sdk/issues/218, https://github.com/kubeflow/sdk/issues/258, https://github.com/kubeflow/sdk/issues/259, https://github.com/kubeflow/sdk/issues/260
- Provide a consistent, unified authentication mechanism: https://github.com/kubeflow/sdk/issues/281

### Trainer SDK
- Manage Kubeflow TrainJobs in a multi-cluster environment: https://github.com/kubeflow/sdk/issues/23
- Snapshot users' workspace into distributed TrainJob workload: https://github.com/kubeflow/sdk/issues/48
- Kubeflow Dynamic LLM Trainer Framework: https://github.com/kubeflow/trainer/issues/2839
- Specialized Trainer Abstractions and RuntimeConfig for the Kubeflow SDK: https://github.com/kubeflow/sdk/issues/285
- Enable GPU Support in the Kubeflow SDK Container Backend: https://github.com/kubeflow/sdk/issues/159
- Track TrainJob Progress and Expose Training Metrics: https://github.com/kubeflow/trainer/issues/2779
- Transparent GPU Checkpointing of TrainJobs with CRIU: https://github.com/kubeflow/trainer/issues/2777
- KubeflowCallback for HuggingFace Transformers: https://github.com/huggingface/transformers/issues/44486

### Optimizer SDK
- Support Local Execution for OptimizerClient: https://github.com/kubeflow/sdk/issues/153

### Spark SDK
- Integrate Kubeflow Spark Application: https://github.com/kubeflow/sdk/issues/107
- Improve SparkClient for Complex Data Engineering: Datalake reading, transformations, configurations, and Spark Session plugins (e.g., datafusion-comet for hardware acceleration): https://github.com/kubeflow/sdk/issues/470
- SparkClient Integration with Kubeflow Components: Kubeflow Pipelines for data enrichment, Spark SQL in Kubeflow Notebooks via pyspark/spark-shell: https://github.com/kubeflow/sdk/issues/471
- AI-Assisted SparkClient Development: Provide context to AI code assistants for writing, debugging, and deploying SparkClient code (via Kubeflow MCP/Agent): https://github.com/kubeflow/mcp-server/issues/5

### Hub/Model Registry SDK
- Integrate Model Registry SDK: https://github.com/kubeflow/sdk/pull/186

### Pipelines SDK
- Integrate Kubeflow Pipelines: https://github.com/kubeflow/sdk/issues/125

### Feast SDK
- Integrate Feast Feature Store with Kubeflow SDK: https://github.com/kubeflow/sdk/issues/239

## 2025

### Core SDK
- Release Kubeflow SDK 0.1.0 to PyPI: https://github.com/kubeflow/sdk/issues/45
- Include Kubeflow SDK into the KFP SDK: https://github.com/kubeflow/pipelines/issues/12027
- Improve test coverage: https://github.com/kubeflow/sdk/issues/16, https://github.com/kubeflow/sdk/issues/18
- Generate docs for Kubeflow SDK: https://github.com/kubeflow/sdk/issues/50
- Integrate Kubeflow SDK with OpenTelemetry: https://github.com/kubeflow/sdk/issues/164
- Experiment Tracking with MLflow for Kubeflow SDK: https://github.com/kubeflow/sdk/issues/63

### Trainer SDK
- Migrate Trainer SDK: https://github.com/kubeflow/sdk/issues/1
- Local Execution for Training Jobs: https://github.com/kubeflow/sdk/issues/22
- Manage Kubeflow TrainJobs in a multi-cluster environment: https://github.com/kubeflow/sdk/issues/23
- Snapshot users' workspace into distributed TrainJob workload: https://github.com/kubeflow/sdk/issues/48
- Support distributed data cache configuration via initializer SDK: https://github.com/kubeflow/trainer/issues/2655
- Support other built-in trainers besides TorchTune: https://github.com/kubeflow/trainer/issues/2752
- Support namespaced TrainingRuntime: https://github.com/kubeflow/sdk/issues/88

### Katib SDK
- Integrate Katib SDK: https://github.com/kubeflow/sdk/issues/46
- Support Local Execution for OptimizerClient: https://github.com/kubeflow/sdk/issues/153

### Model Registry SDK
- Integrate Model Registry SDK: https://github.com/kubeflow/sdk/pull/186

### Pipelines SDK
- Integrate Kubeflow Pipelines: https://github.com/kubeflow/sdk/issues/125

### Spark Application SDK
- Integrate Kubeflow Spark Application: https://github.com/kubeflow/sdk/issues/107
