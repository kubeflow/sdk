import unittest
import sys
import os

# This line ensures Python can find your 'examples' folder
sys.path.append(os.getcwd())

from examples.poc_registry import BACKEND_REGISTRY, register_backend, TrainerClient, BaseBackend

class TestBackendRegistry(unittest.TestCase):
    
    def setUp(self):
        BACKEND_REGISTRY.clear()

    def test_successful_registration(self):
        @register_backend("mock-llm")
        class MockLLM(BaseBackend):
            def train(self): return "Training started"
        
        self.assertIn("mock-llm", BACKEND_REGISTRY)
        client = TrainerClient("mock-llm")
        self.assertEqual(client.train(), "Training started")

    def test_duplicate_registration_error(self):
        @register_backend("duplicate")
        class First(BaseBackend): pass
        
        with self.assertRaises(ValueError):
            @register_backend("duplicate")
            class Second(BaseBackend): pass

    def test_invalid_backend_type_error(self):
        with self.assertRaises(TypeError):
            @register_backend("invalid")
            class NotABackend: pass

if __name__ == "__main__":
    unittest.main()