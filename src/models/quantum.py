"""
4-Qubit Variational Quantum Circuit for MNIST Classification
Uses PennyLane with PyTorch integration
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class QuantumCircuit:
    """
    4-qubit variational quantum circuit with angle encoding.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        device: str = "default.qubit",
        diff_method: str = "parameter-shift"
    ):
        """
        Initialize quantum circuit.

        Args:
            n_qubits: Number of qubits (must be 4 for this architecture)
            n_layers: Number of variational layers
            device: PennyLane device to use
            diff_method: Differentiation method ('parameter-shift' or 'backprop')
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device
        self.diff_method = diff_method

        # Create PennyLane device
        self.dev = qml.device(device, wires=n_qubits)

        # Calculate number of parameters
        # Each layer has n_qubits rotation gates
        self.n_params = n_qubits * n_layers

        print(f"Quantum Circuit initialized:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Layers: {n_layers}")
        print(f"  Parameters: {self.n_params}")
        print(f"  Device: {device}")
        print(f"  Diff method: {diff_method}")

    def _circuit(self, inputs, weights):
        """
        Define the quantum circuit architecture.

        Args:
            inputs: Classical input features (4D vector)
            weights: Trainable quantum parameters
        """
        # Step 1: Angle Encoding
        # Map classical features to quantum states using RY rotations
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # Step 2: Variational Layers
        for layer in range(self.n_layers):
            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RY(weights[layer * self.n_qubits + i], wires=i)

            # Entanglement: Linear connectivity (ring topology)
            # 0 → 1 → 2 → 3 → 0
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        # Step 3: Measurement
        # Return expectation values of Pauli-Z on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def create_qnode(self):
        """
        Create a QNode (quantum node) with the circuit.

        Returns:
            QNode that can be called like a function
        """
        qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method=self.diff_method
        )
        return qnode


class QuantumLayer(nn.Module):
    """
    PyTorch-compatible quantum layer.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        device: str = "default.qubit",
        diff_method: str = "parameter-shift"
    ):
        """
        Initialize quantum layer as PyTorch module.

        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            device: PennyLane device
            diff_method: Differentiation method
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Create quantum circuit
        self.qcircuit = QuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=device,
            diff_method=diff_method
        )

        # Create QNode
        self.qnode = self.qcircuit.create_qnode()

        # Initialize trainable quantum parameters
        # Using nn.Parameter makes them trainable by PyTorch
        n_params = self.qcircuit.n_params
        self.weights = nn.Parameter(
            torch.randn(n_params, requires_grad=True) * 0.01
        )

        print(f"QuantumLayer initialized with {n_params} trainable parameters")

    def forward(self, x):
        """
        Forward pass through quantum circuit.

        Args:
            x: Input tensor of shape (batch_size, n_qubits)

        Returns:
            Output tensor of shape (batch_size, n_qubits)
        """
        batch_size = x.shape[0]

        # Process each sample in the batch through quantum circuit
        outputs = []
        for i in range(batch_size):
            # Get single sample (4D feature vector)
            sample = x[i]

            # Pass through quantum circuit
            # Returns list of expectation values
            result = self.qnode(sample, self.weights)

            # Stack results into tensor
            outputs.append(torch.stack(result))

        # Stack all outputs: (batch_size, n_qubits)
        output = torch.stack(outputs)

        return output

    def get_circuit_diagram(self, sample_input: Optional[torch.Tensor] = None):
        """
        Get a visual representation of the quantum circuit.

        Args:
            sample_input: Sample input for circuit (4D vector)

        Returns:
            String representation of the circuit
        """
        if sample_input is None:
            sample_input = torch.zeros(self.n_qubits)

        # Draw circuit
        drawer = qml.draw(self.qnode)
        circuit_str = drawer(sample_input, self.weights)

        return circuit_str

    def get_circuit_specs(self):
        """
        Get specifications of the quantum circuit.

        Returns:
            Dictionary with circuit specifications
        """
        specs = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_parameters": self.qcircuit.n_params,
            "device": self.qcircuit.device_name,
            "diff_method": self.qcircuit.diff_method,
            "gates_per_layer": {
                "RY_rotations": self.n_qubits,
                "CNOT_entanglement": self.n_qubits
            },
            "total_gates": self.n_qubits + self.n_layers * (2 * self.n_qubits)
        }
        return specs


def create_quantum_layer(
    n_qubits: int = 4,
    n_layers: int = 3,
    device: str = "default.qubit",
    diff_method: str = "parameter-shift"
) -> QuantumLayer:
    """
    Factory function to create a quantum layer.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        device: PennyLane device
        diff_method: Differentiation method

    Returns:
        Initialized QuantumLayer
    """
    return QuantumLayer(
        n_qubits=n_qubits,
        n_layers=n_layers,
        device=device,
        diff_method=diff_method
    )


if __name__ == "__main__":
    # Test the quantum circuit
    print("=" * 60)
    print("Testing Quantum Circuit")
    print("=" * 60)

    # Create quantum layer
    quantum_layer = QuantumLayer(n_qubits=4, n_layers=3)

    # Test with sample input
    print("\nTesting forward pass...")
    batch_size = 8
    sample_input = torch.randn(batch_size, 4)  # Random 4D features

    print(f"Input shape: {sample_input.shape}")

    # Forward pass
    output = quantum_layer(sample_input)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Sample output: {output[0].detach().numpy()}")

    # Test gradient computation
    print("\nTesting gradient computation...")
    output_sum = output.sum()
    output_sum.backward()

    print(f"Gradients computed: {quantum_layer.weights.grad is not None}")
    if quantum_layer.weights.grad is not None:
        print(f"Gradient shape: {quantum_layer.weights.grad.shape}")
        print(f"Gradient norm: {quantum_layer.weights.grad.norm().item():.6f}")

    # Display circuit
    print("\nQuantum Circuit Diagram:")
    print(quantum_layer.get_circuit_diagram(sample_input[0]))

    # Display specs
    print("\nCircuit Specifications:")
    specs = quantum_layer.get_circuit_specs()
    for key, value in specs.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("✓ Quantum circuit test successful!")
    print("=" * 60)
