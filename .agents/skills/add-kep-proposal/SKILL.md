---
name: add-kep-proposal
description: >-
  Create a Kubeflow Enhancement Proposal (KEP). Use when asked to propose a
  significant feature, write a KEP, or draft a proposal for the Kubeflow SDK.
---

# Create a Kubeflow Enhancement Proposal (KEP)

Significant features and enhancements require a KEP under `proposals/`. This
skill walks through creating one that follows the project's conventions.

## Before you start

1. Read an existing KEP for the expected structure:

```
proposals/107-spark-client/README.md
proposals/107-spark-client/kep.yaml
```

2. Check if a GitHub issue already exists for the feature. KEPs reference
   their tracking issue.

3. Determine the KEP number. Use the next available number after checking:

```bash
ls proposals/
```

## Step 1: Create the directory

```bash
mkdir -p proposals/<number>-<short-name>/
```

Use a descriptive short name matching the feature (e.g., `107-spark-client`,
`125-kfp-client`).

## Step 2: Create `kep.yaml`

This is the structured metadata file. Follow this template:

```yaml
title: <Feature Title>
kep-number: <number>
authors:
  - "@github-handle"
status: provisional
creation-date: "<YYYY-MM-DD>"
reviewers:
  - "@reviewer1"
  - "@reviewer2"
approvers:
  - "@approver1"
see-also:
  - https://github.com/kubeflow/sdk/issues/<number>
replaces: []

stage: alpha

latest-milestone: "TBD"

milestone:
  alpha:
    kubeflow-sdk: "TBD"
  beta:
    kubeflow-sdk: "TBD"
  stable:
    kubeflow-sdk: "TBD"
```

**Status values:** `provisional`, `implementable`, `implemented`, `deferred`,
`rejected`, `withdrawn`, `replaced`.

**Stage values:** `alpha`, `beta`, `stable`.

## Step 3: Create `README.md`

This is the proposal document. Use this outline:

```markdown
# KEP-<number>: <Feature Title>

## Authors

- Name - [@github-handle](https://github.com/handle)

Ref: https://github.com/kubeflow/sdk/issues/<number>

## Summary

<1-2 paragraphs describing the feature>

## Motivation

<Why is this needed? What problem does it solve?>

## Goals

1. <Goal 1>
2. <Goal 2>

## Non-Goals

- <What this KEP does NOT cover>

## Design Details

### API

<Python API signatures, class definitions, usage examples>

### Implementation

<Architecture, backend integration, CRD usage if applicable>

## Alternatives Considered

<Other approaches and why they were rejected>
```

Include code examples showing the proposed Python API. Reference existing
SDK patterns (e.g., `TrainerClient`, `SparkClient`) for consistency.

## Step 4: Add diagrams (optional)

If the feature has a complex flow, add architecture diagrams:

```
proposals/<number>-<short-name>/
├── README.md
├── kep.yaml
├── high-level-arch.svg      # Optional
└── detailed-workflow.svg    # Optional
```

## Step 5: Open a PR

Use the PR title format:

```
feat: add <feature-name> proposal (#<issue-number>)
```

Or:

```
feat(docs): Add <feature-name> KEP (#<issue-number>)
```

## Checklist

- [ ] Directory created under `proposals/<number>-<short-name>/`
- [ ] `kep.yaml` with all required fields
- [ ] `README.md` with Summary, Motivation, Goals, Non-Goals, Design Details
- [ ] Links to tracking GitHub issue
- [ ] Authors and reviewers listed
- [ ] Python API examples included in Design Details
- [ ] PR title follows Conventional Commits
