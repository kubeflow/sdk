class KubeflowError(Exception):
    """Base class for all Kubeflow SDK errors."""
    pass


class NameResolutionError(KubeflowError):
    """Raised when a resource cannot be found by name."""
    pass


class CompilationError(KubeflowError):
    """Raised when pipeline or job compilation fails."""
    pass


class RunFailedError(KubeflowError):
    """Raised when a run or job reaches a failed state."""
    pass


class KubeflowTimeoutError(KubeflowError):
    """Raised when a wait operation exceeds its timeout."""
    pass
