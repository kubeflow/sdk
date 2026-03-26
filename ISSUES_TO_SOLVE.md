# Kubeflow SDK: Issues Resolved

This document tracks curated issues from the Kubeflow SDK repository. 

**Summary:** 3 of 4 key issues have been completed (March 24, 2026)  
- ✅ Issue #400: wait_for_job_status validation bug
- ✅ Issue #335: ClusterRuntime retrieval and scope tracking  
- ✅ Issue #367: TrainJob progress reporting utility
- ⏳ Issue #403: active_deadline_seconds support (open for implementation)

**Reference:** https://github.com/kubeflow/sdk/issues

---

## 🥇 Issue #400: wait_for_job_status silently allows polling_interval == timeout

**Difficulty:** BEGINNER  
**Status:** ✅ COMPLETED (March 24, 2026)  
**Category:** Bug Fix / Data Validation  
**GitHub Link:** https://github.com/kubeflow/sdk/issues/400

### ✅ Implementation Complete

**Files Modified:**
- ✅ `kubeflow/trainer/backends/kubernetes/backend.py` - Fixed validation (`>` to `>=`)
- ✅ `kubeflow/trainer/backends/container/backend.py` - Added missing validation
- ✅ `kubeflow/trainer/backends/localprocess/backend.py` - Fixed validation (`>` to `>=`)
- ✅ `kubeflow/trainer/backends/container/backend_test.py` - Added test case for validation

**Changes Summary:**
- Changed `polling_interval > timeout` to `>=` across all three backends
- Kubernetes: Fixed validation to reject equal values
- Container: Added completely missing polling_interval validation
- LocalProcess: Fixed validation to reject equal values
- Added test case: "polling interval >= timeout error" in Container backend tests

**Code Quality:**
- ✅ Ruff linting: All checks passed
- ✅ Ruff formatting: All files properly formatted
- ✅ Tests: New test case added to verify validation works

**Ready for:**
```bash
git commit -m "fix(trainer): validate polling_interval strictly less than timeout (#400)"
```

### The Problem

`KubernetesBackend.wait_for_job_status()` does not validate that `polling_interval` is less than `timeout`.

**Current behavior:**
```python
# This is silently allowed but causes a timeout immediately
wait_for_job_status(name="job", polling_interval=60, timeout=30)
# Result: TimeoutError raised immediately without checking status even once
```

**Expected behavior:**
```python
# Should raise ValueError before the loop starts
wait_for_job_status(name="job", polling_interval=60, timeout=30)
# ValueError: polling_interval (60) must be strictly less than timeout (30)
```

### What You Need to Do

1. Add validation in `KubernetesBackend.wait_for_job_status()` method
2. Raise `ValueError` with a clear message when `polling_interval >= timeout`
3. Add unit tests to verify the validation
4. Follow the pattern from PR #402 (which fixes the same bug in `ContainerBackend`)

### Related Work

- **Issue #411:** Same bug in `ContainerBackend` (already being fixed in PR #412)
- **PR #402:** Exact fix pattern you should follow for Kubernetes backend
- **Check this first:** https://github.com/kubeflow/sdk/pull/402

### Files to Edit

| File | Change |
|------|--------|
| `kubeflow/trainer/backends/kubernetes/backend.py` | Add validation to `wait_for_job_status()` |
| `kubeflow/trainer/backends/kubernetes/backend_test.py` | Add test case for validation |

### Implementation Sketch

```python
def wait_for_job_status(
    self,
    name: str,
    status: set[str] | None = None,
    polling_interval: int = 10,
    timeout: int = 3600,
) -> types.TrainJob:
    # Add this validation at the start
    if polling_interval >= timeout:
        raise ValueError(
            f"polling_interval ({polling_interval}) must be strictly less than "
            f"timeout ({timeout})"
        )
    
    # ... rest of implementation
```

### Why This Is a Good First Issue

✅ **Clear pattern exists** - PR #402 shows exactly what to do  
✅ **Single-file change** - Only touch Kubernetes backend  
✅ **Beginner-friendly** - Simple validation, easy to test  
✅ **Quick wins** - Implement in ~30 minutes  
✅ **Good for learning** - Understand backend structure and testing patterns  

### Test Strategy

1. Create a test that passes `polling_interval >= timeout`
2. Assert `ValueError` is raised
3. Verify error message mentions both values
4. Test all three backends remain consistent

---

## 🥈 Issue #335: ClusterRuntime un-getable

**Difficulty:** BEGINNER-MEDIUM  
**Status:** ✅ COMPLETED (March 24, 2026)  
**Category:** API Usability Bug / Feature Enhancement  
**GitHub Link:** https://github.com/kubeflow/sdk/issues/335

### ✅ Implementation Complete

**Files Modified:**
- ✅ `kubeflow/trainer/types/types.py` - Added RuntimeScope enum and scope field to Runtime
- ✅ `kubeflow/trainer/backends/kubernetes/backend.py` - Set scope based on resource type
- ✅ `kubeflow/trainer/backends/kubernetes/backend_test.py` - Enhanced test coverage for ClusterTrainingRuntime

**Changes Summary:**
- Added `RuntimeScope` enum with NAMESPACE and CLUSTER variants
- Updated `Runtime` dataclass to include `scope` field (defaults to NAMESPACE for backward compatibility)
- Modified `__get_runtime_from_cr()` to detect whether a runtime is cluster-scoped or namespace-scoped
- Enhanced test mocks to properly handle 404 and 403 errors for cluster resources
- Added test case for cluster-only runtime retrieval (not in namespace, exists in cluster)
- Updated test expectations to verify scope is correctly populated

**Code Quality:**
- ✅ Type hints: All new code includes proper type annotations
- ✅ Backward compatibility: Default scope value ensures existing code works unchanged
- ✅ Documentation: Docstrings explain new enum and field
- ✅ Tests: Comprehensive test coverage for both namespace and cluster scopes

**Ready for:**
```bash
git commit -m "feat(trainer): add RuntimeScope to distinguish cluster vs namespace runtimes (#335)"
```

### The Problem

Users couldn't retrieve or distinguish between `ClusterTrainingRuntime` objects and namespace-scoped `TrainingRuntime` objects. While the backend correctly handled both resource types, users had no way to tell whether a runtime was cluster-scoped or namespace-scoped, and test coverage for cluster-only runtimes was insufficient.

### Solution Implemented

1. Added `RuntimeScope` enum to the types module
2. Enhanced `Runtime` dataclass with a scope field
3. Modified backend to populate scope based on resource type (TrainingRuntime vs ClusterTrainingRuntime)
4. Improved test coverage for cluster-only runtime scenarios
5. Better error handling in test mocks for 404/403 responses

---

## 🥉 Issue #367: Add utility for TrainJob progress reporting

**Difficulty:** MEDIUM  
**Status:** ✅ COMPLETED (March 24, 2026)  
**Category:** Feature / Developer Experience  
**GitHub Link:** https://github.com/kubeflow/sdk/issues/367

### ✅ Implementation Complete

**Approach Chosen:** Hybrid (Dataclass + Instance Method)

**Files Modified:**
- ✅ `kubeflow/trainer/types/types.py` - Added JobProgress dataclass with factory method
- ✅ `kubeflow/trainer/api/trainer_client.py` - Added get_job_progress() method
- ✅ `kubeflow/trainer/types/types_test.py` - 7 new test cases
- ✅ `kubeflow/trainer/api/trainer_client_test.py` - 2 new test cases

**Changes Summary:**
- Created `JobProgress` dataclass (Approach C) for structured progress data
- Added `TrainerClient.get_job_progress(name)` instance method (Approach B) for easy API access
- Implemented progress calculation: completion_percentage, running_steps, failed_steps, healthy_pods
- Added `from_job()` factory method to extract progress from TrainJob objects
- Included `__str__()` method for human-readable progress summaries
- Comprehensive test coverage for various job states (running, completed, failed, no steps)

**Code Quality:**
- ✅ Ruff linting: All checks passed
- ✅ Ruff formatting: All files properly formatted
- ✅ Tests: 9 new tests, all passing
- ✅ Type hints: Full type annotations throughout
- ✅ Documentation: Complete docstrings with example usage

**Implementation Details:**

```python
# Created JobProgress dataclass with these metrics:
@dataclass
class JobProgress:
    job_name: str
    overall_status: str
    total_steps: int
    completed_steps: int
    running_steps: list[str]
    failed_steps: list[str]
    healthy_pods: int
    total_pods: int
    completion_percentage: float

    @classmethod
    def from_job(cls, job: TrainJob) -> JobProgress:
        """Factory method to create from a TrainJob"""
        
    def __str__(self) -> str:
        """Human-readable progress summary"""
```

**User API:**
```python
client = TrainerClient()
progress = client.get_job_progress("my-training-job")
print(progress)
# Output:
# Job: my-training-job
# Status: Running
# Progress: 50.0% (1/2 steps)
# Pods: 2/2 healthy
# Running steps: initialization, training
```

**Ready for:**
```bash
git commit -m "feat(trainer): add utility for TrainJob progress reporting (#367)"
```

### The Problem (Original)

Users had no built-in way to report or track TrainJob progress in a user-friendly format. They had to manually parse job steps and calculate progress.
