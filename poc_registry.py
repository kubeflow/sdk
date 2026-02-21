import abc
from typing import Dict, Type, Any

# 1. The Registry: Now with Type Hinting as requested by review
BACKEND_REGISTRY: Dict[str, Type['BaseBackend']] = {}

def register_backend(name: str):
    """Decorator to register new backends dynamically with validation."""
    def wrapper(cls: Type[Any]):
        # Validation: Ensure we don't overwrite existing backends
        if name in BACKEND_REGISTRY:
            raise ValueError(f"Backend '{name}' is already registered.")
        
        # Validation: Ensure the class inherits from BaseBackend
        if not issubclass(cls, BaseBackend):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseBackend")
            
        BACKEND_REGISTRY[name] = cls
        return cls
    return wrapper

# 2. The Base Class template
class BaseBackend(abc.ABC):
    @abc.abstractmethod
    def train(self) -> str:
        pass

# 3. Dynamic Implementation
@register_backend("kubernetes")
class KubernetesBackend(BaseBackend):
    def train(self) -> str:
        return "Training on Kubernetes Cluster..."

@register_backend("local")
class LocalBackend(BaseBackend):
    def train(self) -> str:
        return "Training on Local Process..."

# 4. The Refactored Client
class TrainerClient:
    def __init__(self, backend_type: str):
        backend_class = BACKEND_REGISTRY.get(backend_type)
        if not backend_class:
            raise ValueError(f"Backend '{backend_type}' not found in registry!")
        self.backend = backend_class()

    def train(self) -> str:
        # Returning value instead of printing for better API design
        return self.backend.train()

# --- Test the Logic ---
if __name__ == "__main__":
    print("Available Backends:", list(BACKEND_REGISTRY.keys()))
    client = TrainerClient("kubernetes")
    print(f"Result: {client.train()}")