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


# Common constants.
GROUP = "kubeflow.org"
VERSION = "v1beta1"
API_VERSION = f"{GROUP}/{VERSION}"

# The Kind name for the Experiment.
EXPERIMENT_KIND = "Experiment"

# The plural for the Experiment.
EXPERIMENT_PLURAL = "experiments"

# The succeeded condition for the Experiment.
EXPERIMENT_SUCCEEDED = "Succeeded"

# Label to identify Experiment's resources.
EXPERIMENT_LABEL = "katib.kubeflow.org/experiment"

# The plural for the Trials.
TRIAL_PLURAL = "trials"

# The Kind name for the Trials.
TRIAL_KIND = "Trial"

# The Kind name for the OptimizationJob.
OPTIMIZATION_JOB_KIND = "OptimizationJob"

# The default status for the OptimizationJob once users create it.
OPTIMIZATION_JOB_CREATED = "Created"

# The running status of the OptimizationJob, defined when at least one TrainJob is running.
OPTIMIZATION_JOB_RUNNING = "Running"

# The complete status of the OptimizationJob, defined when Experiment CR has succeeded condition.
OPTIMIZATION_JOB_COMPLETE = "Complete"

# The failed status of the OptimizationJob, defined when Experiment CR has failed condition.
OPTIMIZATION_JOB_FAILED = "Failed"

# Katib search space parameter types.
DOUBLE_PARAMETER = "double"
CATEGORICAL_PARAMETERS = "categorical"
