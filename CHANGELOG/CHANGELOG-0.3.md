# Changelog

## [0.3.0](https://github.com/kubeflow/sdk/releases/tag/0.3.0) (2026-01-19)
### 🚀 Features

- feat(ci): Switch to UV for Dependabot ([#231](https://github.com/kubeflow/sdk/pull/231) by @andreyvelich)
- feat: added git cliff for generating changelogs ([#226](https://github.com/kubeflow/sdk/pull/226) by @jaiakash)
- feat(docs): Add Kubeflow SDK YouTube demos ([#229](https://github.com/kubeflow/sdk/pull/229) by @andreyvelich)
- feat(docs): KEP- Spark Client for Kubeflow SDK ([#163](https://github.com/kubeflow/sdk/pull/163) by @Shekharrajak)
- feat(trainer): add get_job_events API to retrieve TrainJob events ([#220](https://github.com/kubeflow/sdk/pull/220) by @sksingh2005)
- feat(trainer): support NVIDIA MIG device resources in TrainJob device… ([#204](https://github.com/kubeflow/sdk/pull/204) by @LabsJS)
- feat: Add custom instructions for GitHub Copilot ([#212](https://github.com/kubeflow/sdk/pull/212) by @osamaahmed17)
- feat:  Add callbacks to the wait_for_job_status() API ([#205](https://github.com/kubeflow/sdk/pull/205) by @osamaahmed17)
- feat(trainer): Allow to reference runtime by name ([#214](https://github.com/kubeflow/sdk/pull/214) by @andreyvelich)
- feat(trainer): Support optional image for CustomTrainer ([#216](https://github.com/kubeflow/sdk/pull/216) by @andreyvelich)
- feat: Add dependabot to Kubeflow SDK ([#194](https://github.com/kubeflow/sdk/pull/194) by @kramaranya)
- feat(docs): Update README with announcement blog post ([#157](https://github.com/kubeflow/sdk/pull/157) by @andreyvelich)

### 🐛 Bug Fixes

- fix: include full pr name for change log ([#236](https://github.com/kubeflow/sdk/pull/236) by @jaiakash)
- fix: Upgrade urllib3 to v2.6.3 ([#230](https://github.com/kubeflow/sdk/pull/230) by @Fiona-Waters)
- fix(trainer): Fix parsing for TrainJob events ([#228](https://github.com/kubeflow/sdk/pull/228) by @andreyvelich)
- fix: Upgrade urllib3 to  v2.6.1 ([#193](https://github.com/kubeflow/sdk/pull/193) by @Fiona-Waters)
- fix(trainer): expose CustomTrainerContainer for import ([#185](https://github.com/kubeflow/sdk/pull/185) by @AndEsterson)
- fix: update permissions for welcome workflow to avoid 403 error ([#181](https://github.com/kubeflow/sdk/pull/181) by @aniketpati1121)
- fix(ci): Move permissions to the workflow root ([#177](https://github.com/kubeflow/sdk/pull/177) by @kramaranya)
- fix: pip install with --user argument fails with image running in python virtual environment ([#162](https://github.com/kubeflow/sdk/pull/162) by @briangallagher)
- fix(trainer): Remove namespace from ClusterTrainingRuntime exception messages ([#166](https://github.com/kubeflow/sdk/pull/166) by @astefanutti)
- fix(trainer): Use PyTorch static rendezvous in container backend ([#168](https://github.com/kubeflow/sdk/pull/168) by @astefanutti)
- fix(trainer): Fix listing containers with Podman backend ([#154](https://github.com/kubeflow/sdk/pull/154) by @astefanutti)

### ⚙️ Miscellaneous Tasks

- chore(deps): bump kubernetes from 33.1.0 to 35.0.0 ([#235](https://github.com/kubeflow/sdk/pull/235) by @dependabot[bot])
- chore(deps): bump pytest from 8.4.1 to 8.4.2 ([#234](https://github.com/kubeflow/sdk/pull/234) by @dependabot[bot])
- chore(deps): bump the python-minor group with 6 updates ([#233](https://github.com/kubeflow/sdk/pull/233) by @dependabot[bot])
- chore: Nominate @kramaranya as Kubeflow SDK approver ([#206](https://github.com/kubeflow/sdk/pull/206) by @andreyvelich)
- chore(ci): bump softprops/action-gh-release from 1 to 2 ([#209](https://github.com/kubeflow/sdk/pull/209) by @dependabot[bot])
- chore(ci): bump actions/upload-artifact from 4 to 6 ([#208](https://github.com/kubeflow/sdk/pull/208) by @dependabot[bot])
- chore(ci): bump actions/download-artifact from 6 to 7 ([#207](https://github.com/kubeflow/sdk/pull/207) by @dependabot[bot])
- chore(ci): bump amannn/action-semantic-pull-request from 5.5.3 to 6.1.1 ([#210](https://github.com/kubeflow/sdk/pull/210) by @dependabot[bot])
- chore(ci): bump actions/github-script from 7 to 8 ([#201](https://github.com/kubeflow/sdk/pull/201) by @dependabot[bot])
- chore(ci): bump actions/download-artifact from 4 to 6 ([#200](https://github.com/kubeflow/sdk/pull/200) by @dependabot[bot])
- chore(ci): bump actions/stale from 5 to 10 ([#199](https://github.com/kubeflow/sdk/pull/199) by @dependabot[bot])
- chore(ci): bump actions/setup-python from 5 to 6 ([#198](https://github.com/kubeflow/sdk/pull/198) by @dependabot[bot])
- chore(ci): bump actions/checkout from 4 to 6 ([#202](https://github.com/kubeflow/sdk/pull/202) by @dependabot[bot])
- chore(docs): Add new items to the roadmap ([#187](https://github.com/kubeflow/sdk/pull/187) by @kramaranya)


### New Contributors
* @Shekharrajak made their first contribution in [#163](https://github.com/kubeflow/sdk/pull/163)
* @sksingh2005 made their first contribution in [#220](https://github.com/kubeflow/sdk/pull/220)
* @LabsJS made their first contribution in [#204](https://github.com/kubeflow/sdk/pull/204)
* @osamaahmed17 made their first contribution in [#212](https://github.com/kubeflow/sdk/pull/212)
* @AndEsterson made their first contribution in [#185](https://github.com/kubeflow/sdk/pull/185)
