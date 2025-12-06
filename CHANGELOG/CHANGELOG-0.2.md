# [0.2.1](https://github.com/kubeflow/sdk/releases/tag/0.2.1) (2025-11-25)

## New Features

- feat(docs): Update README with announcement blog post ([#157](https://github.com/kubeflow/sdk/pull/157)) by @andreyvelich

## Bug Fixes

- fix(ci): Move permissions to the workflow root ([#177](https://github.com/kubeflow/sdk/pull/177)) by @kramaranya
- fix: pip install with --user argument fails with image running in python virtual environment ([#162](https://github.com/kubeflow/sdk/pull/162)) by @briangallagher
- fix(trainer): Remove namespace from ClusterTrainingRuntime exception messages ([#166](https://github.com/kubeflow/sdk/pull/166)) by @astefanutti
- fix(trainer): Use PyTorch static rendezvous in container backend ([#168](https://github.com/kubeflow/sdk/pull/168)) by @astefanutti
- fix(trainer): Fix listing containers with Podman backend ([#154](https://github.com/kubeflow/sdk/pull/154)) by @astefanutti

# [0.2.0](https://github.com/kubeflow/sdk/releases/tag/0.2.0) (2025-11-06)

## New Features

- feat(optimizer): Add get_best_results API to OptimizerClient ([#152](https://github.com/kubeflow/sdk/pull/152)) by @kramaranya
- feat(trainer): Add local notebook examples to E2E ([#149](https://github.com/kubeflow/sdk/pull/149)) by @Fiona-Waters
- feat(optimizer): Add get_job_logs API to OptimizerClient ([#148](https://github.com/kubeflow/sdk/pull/148)) by @kramaranya
- feat(optimizer): Add wait_for_job_status and get_best_trial APIs to OptimizerClient ([#145](https://github.com/kubeflow/sdk/pull/145)) by @kramaranya
- feat: Implement Training Options pattern for flexible TrainJob customization ([#91](https://github.com/kubeflow/sdk/pull/91)) by @abhijeet-dhumal
- feat: Add ContainerBackend with Docker and Podman ([#119](https://github.com/kubeflow/sdk/pull/119)) by @Fiona-Waters
- feat(trainer): add s3 initializers, add `ignore_patterns` to hf initializers ([#131](https://github.com/kubeflow/sdk/pull/131)) by @rudeigerc
- feat(ci): add workflow to approve ok-to-test label ([#138](https://github.com/kubeflow/sdk/pull/138)) by @aniketpati1121
- feat(trainer): Add CustomTrainerContainer to create TrainJobs from image ([#127](https://github.com/kubeflow/sdk/pull/127)) by @andreyvelich
- feat: Hyperparameter Optimization APIs in Kubeflow SDK ([#124](https://github.com/kubeflow/sdk/pull/124)) by @andreyvelich
- feat(trainer): KEP-2655: Support provisioning of cache with Kubeflow SDK ([#112](https://github.com/kubeflow/sdk/pull/112)) by @akshaychitneni
- feat: Support LoraConfig in TorchTune BuiltinTrainer ([#102](https://github.com/kubeflow/sdk/pull/102)) by @Electronic-Waste
- feat(docs): KEP-46-Hyperparameter Optimization in Kubeflow SDK ([#123](https://github.com/kubeflow/sdk/pull/123)) by @kramaranya

## Bug Fixes

- fix(ci): Update url for installing docker for use with local notebooks ([#151](https://github.com/kubeflow/sdk/pull/151)) by @Fiona-Waters
- fix(trainer): Remove --user flag from packages install in local subprocess ([#147](https://github.com/kubeflow/sdk/pull/147)) by @andreyvelich
- fix(trainer): Fix empty image for Runtime trainer ([#143](https://github.com/kubeflow/sdk/pull/143)) by @andreyvelich
- fix: Update Kubeflow SDK diagram ([#146](https://github.com/kubeflow/sdk/pull/146)) by @kramaranya
- fix(trainer): Fix S3 initializer implementation ([#144](https://github.com/kubeflow/sdk/pull/144)) by @andreyvelich
- fix: Support custom images in ClusterTrainingRuntime for container backend ([#140](https://github.com/kubeflow/sdk/pull/140)) by @Fiona-Waters
- fix: add --user when install python packages ([#136](https://github.com/kubeflow/sdk/pull/136)) by @briangallagher
- fix(ci): Fix first-time PR welcome workflow ([#117](https://github.com/kubeflow/sdk/pull/117)) by @kramaranya
- fix(ci): Skip release workflow on forks ([#113](https://github.com/kubeflow/sdk/pull/113)) by @kramaranya
- fix(scripts): Use previous stable tag for changelog ([#103](https://github.com/kubeflow/sdk/pull/103)) by @kramaranya

## Maintenance

- chore: Add HPO support to readme and SDK diagram ([#141](https://github.com/kubeflow/sdk/pull/141)) by @kramaranya
- chore(ci): Add pre-commit configuration and CI workflow ([#134](https://github.com/kubeflow/sdk/pull/134)) by @aniketpati1121
- chore(docs): added AGENTS.MD ([#106](https://github.com/kubeflow/sdk/pull/106)) by @hawkaii
- chore(docs): Add Spark Operator to the future supported projects ([#109](https://github.com/kubeflow/sdk/pull/109)) by @andreyvelich
