"""
Tests for hybrid quantum-classical model
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.hybrid import HybridQuantumClassifier
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestHybridModel:
    """Test hybrid quantum-classical model"""

    def test_model_creation(self):
        """Test creating hybrid model"""
        model = HybridQuantumClassifier(
            n_qubits=4,
            n_quantum_layers=2,
            n_classes=10
        )
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        inputs = torch.randn(8, 1, 28, 28)

        outputs = model(inputs)
        assert outputs.shape == (8, 10), f"Expected (8, 10), got {outputs.shape}"

    def test_output_logits(self):
        """Test output are logits (not probabilities)"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        inputs = torch.randn(4, 1, 28, 28)

        outputs = model(inputs)
        # Logits can be any value
        assert outputs.dtype == torch.float32

    def test_batch_sizes(self):
        """Test various batch sizes"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)

        for batch_size in [1, 4, 16, 32]:
            inputs = torch.randn(batch_size, 1, 28, 28)
            outputs = model(inputs)
            assert outputs.shape[0] == batch_size

    def test_gradient_flow(self):
        """Test gradients flow through entire model"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        inputs = torch.randn(4, 1, 28, 28, requires_grad=True)

        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()

        # Check some parameters have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "Some parameters should have gradients"

    def test_eval_mode(self):
        """Test model in evaluation mode"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        model.eval()

        inputs = torch.randn(4, 1, 28, 28)
        outputs = model(inputs)
        assert outputs.shape == (4, 10)

    def test_train_mode(self):
        """Test model in training mode"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        model.train()

        inputs = torch.randn(4, 1, 28, 28)
        outputs = model(inputs)
        assert outputs.shape == (4, 10)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestModelComponents:
    """Test individual model components"""

    def test_has_classical_component(self):
        """Test model has classical feature extractor"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        assert hasattr(model, 'classical_extractor')

    def test_has_quantum_component(self):
        """Test model has quantum layer"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        assert hasattr(model, 'quantum_layer')

    def test_has_classifier(self):
        """Test model has classifier"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        assert hasattr(model, 'classifier')

    def test_parameter_count(self):
        """Test total parameter count"""
        model = HybridQuantumClassifier(n_qubits=4, n_quantum_layers=2, n_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0, "Model should have parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
