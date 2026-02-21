import abc

# 1. The Registry: This is the core idea for Project 10
BACKEND_REGISTRY = {}

def register_backend(name):
    """Decorator to register new backends dynamically."""
    def wrapper(cls):
        BACKEND_REGISTRY[name] = cls
        return cls
    return wrapper

# 2. The Base Class template
class BaseBackend(abc.ABC):
    @abc.abstractmethod
    def train(self):
        pass

# 3. Dynamic Implementation (How we will add new LLM backends)
@register_backend("kubernetes")
class KubernetesBackend(BaseBackend):
    def train(self):
        return "Training on Kubernetes Cluster..."

@register_backend("local")
class LocalBackend(BaseBackend):
    def train(self):
        return "Training on Local Process..."

# 4. The Refactored Client that uses the Registry
class TrainerClient:
    def __init__(self, backend_type):
        backend_class = BACKEND_REGISTRY.get(backend_type)
        if not backend_class:
            raise ValueError(f"Backend {backend_type} not found!")
        self.backend = backend_class()

    def train(self):
        print(self.backend.train())

# --- Test the Logic ---
if __name__ == "__main__":
    print("Available Backends:", list(BACKEND_REGISTRY.keys()))
    client = TrainerClient("kubernetes")
    client.train()