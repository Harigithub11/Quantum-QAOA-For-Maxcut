"""
Hybrid Classical-Quantum Model for MNIST Classification
Combines ResNet18 Feature Extractor + Quantum Circuit + Classification Head
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classical import ResNet18FeatureExtractor
from models.quantum import QuantumLayer


class HybridQuantumClassifier(nn.Module):
    """
    Hybrid quantum-classical neural network for MNIST digit classification.

    Architecture:
        Input (28x28 grayscale)
        → ResNet18 Feature Extractor (512-dim)
        → Dimensionality Reduction (4-dim)
        → Quantum Circuit (4-qubit, 4 expectation values)
        → Classification Head (10 classes)
    """

    def __init__(
        self,
        # Classical component parameters
        pretrained_resnet: bool = True,
        freeze_resnet: bool = True,
        feature_dim: int = 4,
        dropout: float = 0.2,
        # Quantum component parameters
        n_qubits: int = 4,
        n_layers: int = 3,
        quantum_device: str = "default.qubit",
        diff_method: str = "parameter-shift",
        # Classification head parameters
        n_classes: int = 10,
        hidden_dim: Optional[int] = 16
    ):
        """
        Initialize hybrid quantum-classical model.

        Args:
            pretrained_resnet: Use pre-trained ResNet18 weights
            freeze_resnet: Freeze ResNet layers (feature extraction mode)
            feature_dim: Dimension of features fed to quantum circuit (must be 4)
            dropout: Dropout probability for classical layers
            n_qubits: Number of qubits in quantum circuit (must be 4)
            n_layers: Number of variational layers in quantum circuit
            quantum_device: PennyLane device for quantum simulation
            diff_method: Differentiation method for quantum gradients
            n_classes: Number of output classes (10 for MNIST)
            hidden_dim: Hidden dimension for classification head (None = direct mapping)
        """
        super().__init__()

        # Validate dimensions
        assert feature_dim == n_qubits, "Feature dimension must equal number of qubits"
        assert feature_dim == 4, "Current architecture requires 4-dimensional features"

        self.feature_dim = feature_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        # Component 1: Classical Feature Extractor (ResNet18)
        self.classical_extractor = ResNet18FeatureExtractor(
            pretrained=pretrained_resnet,
            freeze_layers=freeze_resnet,
            feature_dim=feature_dim,
            dropout=dropout
        )

        # Component 2: Quantum Circuit
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=quantum_device,
            diff_method=diff_method
        )

        # Component 3: Classification Head
        # Maps quantum expectation values (4D) to class probabilities (10D)
        if hidden_dim is not None:
            # Two-layer classifier with hidden dimension
            self.classifier = nn.Sequential(
                nn.Linear(n_qubits, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
            )
        else:
            # Direct linear mapping
            self.classifier = nn.Linear(n_qubits, n_classes)

        self._initialize_classifier()

        print(f"\nHybrid Quantum-Classical Classifier initialized:")
        print(f"  Input: (batch_size, 1, 28, 28)")
        print(f"  Classical features: {feature_dim}-dimensional")
        print(f"  Quantum processing: {n_qubits} qubits, {n_layers} layers")
        print(f"  Output: {n_classes} classes")
        print(f"  Total trainable parameters: {self.count_trainable_parameters():,}")

    def _initialize_classifier(self):
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through hybrid model.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Logits tensor of shape (batch_size, n_classes)
        """
        # Step 1: Classical feature extraction
        classical_features = self.classical_extractor(x)  # (batch_size, 4)

        # Step 2: Quantum processing
        quantum_output = self.quantum_layer(classical_features)  # (batch_size, 4)

        # Ensure consistent dtype (quantum output may be float64)
        quantum_output = quantum_output.float()

        # Step 3: Classification
        logits = self.classifier(quantum_output)  # (batch_size, 10)

        return logits

    def forward_with_features(self, x):
        """
        Forward pass that also returns intermediate features for visualization.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Dictionary containing:
                - logits: Output logits (batch_size, n_classes)
                - classical_features: Features from ResNet (batch_size, 4)
                - quantum_output: Quantum expectation values (batch_size, 4)
                - predictions: Predicted class indices (batch_size,)
                - probabilities: Class probabilities (batch_size, n_classes)
        """
        # Forward pass through each component
        classical_features = self.classical_extractor(x)
        quantum_output = self.quantum_layer(classical_features).float()
        logits = self.classifier(quantum_output)

        # Compute predictions and probabilities
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        return {
            'logits': logits,
            'classical_features': classical_features,
            'quantum_output': quantum_output,
            'predictions': predictions,
            'probabilities': probabilities
        }

    def predict(self, x):
        """
        Make predictions on input images.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Predicted class indices (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x):
        """
        Get class probabilities for input images.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Class probabilities (batch_size, n_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model specifications
        """
        info = {
            'architecture': 'Hybrid Quantum-Classical CNN',
            'classical_backbone': 'ResNet18',
            'quantum_circuit': f'{self.n_qubits}-qubit variational circuit',
            'n_quantum_layers': self.n_layers,
            'feature_dimension': self.feature_dim,
            'n_classes': self.n_classes,
            'total_parameters': self.count_total_parameters(),
            'trainable_parameters': self.count_trainable_parameters(),
            'frozen_parameters': self.count_total_parameters() - self.count_trainable_parameters(),
            'quantum_parameters': self.quantum_layer.qcircuit.n_params,
            'classifier_parameters': sum(p.numel() for p in self.classifier.parameters())
        }
        return info

    def unfreeze_classical_layers(self, num_layers: Optional[int] = None):
        """
        Unfreeze ResNet layers for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        self.classical_extractor.unfreeze_layers(num_layers)
        print(f"Updated trainable parameters: {self.count_trainable_parameters():,}")

    def get_quantum_circuit_diagram(self):
        """Get visual representation of quantum circuit."""
        return self.quantum_layer.get_circuit_diagram()


def create_hybrid_model(
    pretrained: bool = True,
    freeze_resnet: bool = True,
    n_quantum_layers: int = 3,
    quantum_device: str = "default.qubit",
    diff_method: str = "parameter-shift",
    dropout: float = 0.2
) -> HybridQuantumClassifier:
    """
    Factory function to create hybrid model with default configuration.

    Args:
        pretrained: Use pre-trained ResNet18
        freeze_resnet: Freeze ResNet layers
        n_quantum_layers: Number of variational layers in quantum circuit
        quantum_device: PennyLane device
        diff_method: Quantum differentiation method
        dropout: Dropout probability

    Returns:
        Initialized HybridQuantumClassifier
    """
    return HybridQuantumClassifier(
        pretrained_resnet=pretrained,
        freeze_resnet=freeze_resnet,
        n_layers=n_quantum_layers,
        quantum_device=quantum_device,
        diff_method=diff_method,
        dropout=dropout
    )


if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    if os.name == 'nt':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    print("=" * 70)
    print("Testing Hybrid Quantum-Classical Model")
    print("=" * 70)

    # Create hybrid model
    print("\n1. Creating hybrid model...")
    model = HybridQuantumClassifier(
        pretrained_resnet=True,
        freeze_resnet=True,
        n_layers=3,
        dropout=0.2
    )

    # Display model info
    print("\n2. Model Information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 4
    sample_images = torch.randn(batch_size, 1, 28, 28)
    print(f"  Input shape: {sample_images.shape}")

    # Basic forward pass
    logits = model(sample_images)
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # Forward pass with features
    print("\n4. Testing forward pass with features...")
    outputs = model.forward_with_features(sample_images)
    print(f"  Classical features shape: {outputs['classical_features'].shape}")
    print(f"  Quantum output shape: {outputs['quantum_output'].shape}")
    print(f"  Predictions shape: {outputs['predictions'].shape}")
    print(f"  Probabilities shape: {outputs['probabilities'].shape}")
    print(f"  Sample prediction: class {outputs['predictions'][0].item()}")
    print(f"  Sample probabilities: {outputs['probabilities'][0][:3].detach().numpy()}")

    # Test predictions
    print("\n5. Testing prediction methods...")
    predictions = model.predict(sample_images)
    probabilities = model.predict_proba(sample_images)
    print(f"  Predictions: {predictions.numpy()}")
    print(f"  Max probabilities: {probabilities.max(dim=1)[0].numpy()}")

    # Test gradient computation
    print("\n6. Testing gradient computation...")
    model.train()
    loss = logits.sum()
    loss.backward()

    # Check gradients in different components
    classical_grad = any(p.grad is not None for p in model.classical_extractor.parameters() if p.requires_grad)
    quantum_grad = model.quantum_layer.weights.grad is not None
    classifier_grad = any(p.grad is not None for p in model.classifier.parameters())

    print(f"  Classical extractor gradients: {classical_grad}")
    print(f"  Quantum layer gradients: {quantum_grad}")
    print(f"  Classifier gradients: {classifier_grad}")

    if quantum_grad:
        print(f"  Quantum gradient norm: {model.quantum_layer.weights.grad.norm().item():.6f}")

    # Test with actual MNIST data
    print("\n7. Testing with MNIST data...")
    try:
        from src.data.dataset import MNISTDataModule

        data_module = MNISTDataModule(batch_size=8)
        data_module.setup()

        images, labels = data_module.get_sample_batch()
        print(f"  MNIST batch: {images.shape}, Labels: {labels.shape}")

        with torch.no_grad():
            model.eval()
            outputs = model.forward_with_features(images)

        print(f"  Predictions: {outputs['predictions'][:8].numpy()}")
        print(f"  Actual labels: {labels[:8].numpy()}")
        print(f"  Accuracy: {(outputs['predictions'] == labels).float().mean().item():.2%}")

        print("\n  ✓ Integration with MNIST data successful!")
    except Exception as e:
        print(f"  Note: Could not test with MNIST data - {e}")

    # Model summary
    print("\n8. Component Parameter Breakdown:")
    print(f"  ResNet18 frozen: {model.classical_extractor.count_total_parameters() - model.classical_extractor.count_trainable_parameters():,}")
    print(f"  Feature reduction: {model.classical_extractor.count_trainable_parameters():,}")
    print(f"  Quantum circuit: {model.quantum_layer.qcircuit.n_params:,}")
    print(f"  Classifier head: {sum(p.numel() for p in model.classifier.parameters()):,}")
    print(f"  Total trainable: {model.count_trainable_parameters():,}")

    print("\n" + "=" * 70)
    print("✓ Hybrid Model test successful!")
    print("=" * 70)
