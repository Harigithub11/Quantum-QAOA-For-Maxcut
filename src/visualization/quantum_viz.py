"""
Quantum Circuit Visualization Utilities
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
import torch

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not installed. Quantum visualization will be limited.")


def draw_quantum_circuit(
    quantum_layer,
    sample_input: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    style: str = "black_white"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw quantum circuit using PennyLane's matplotlib drawer.

    Args:
        quantum_layer: QuantumLayer instance
        sample_input: Sample input tensor (4D vector)
        save_path: Path to save figure (optional)
        style: Drawing style ('black_white', 'pennylane', 'sketch', etc.)

    Returns:
        Tuple of (figure, axes)
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane is required for quantum circuit visualization")

    if sample_input is None:
        sample_input = torch.zeros(quantum_layer.n_qubits)

    # Set drawing style
    qml.drawer.use_style(style)

    # Create matplotlib-based circuit drawing
    fig, ax = qml.draw_mpl(quantum_layer.qnode)(
        sample_input,
        quantum_layer.weights
    )

    fig.set_size_inches(12, 6)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Circuit diagram saved to: {save_path}")

    return fig, ax


def visualize_parameter_evolution(
    parameter_history: list,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize how quantum parameters evolve during training.

    Args:
        parameter_history: List of parameter tensors at different epochs
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # Convert to numpy
    params_array = np.array([p.detach().cpu().numpy() for p in parameter_history])
    n_epochs = len(parameter_history)
    n_params = params_array.shape[1]

    # Plot 1: All parameters over time
    ax = axes[0]
    for i in range(min(n_params, 12)):  # Plot first 12 parameters
        ax.plot(params_array[:, i], label=f'θ_{i}', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Parameter Value', fontsize=11)
    ax.set_title('Quantum Parameter Evolution', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2: Parameter magnitude distribution
    ax = axes[1]
    final_params = params_array[-1]
    ax.bar(range(n_params), np.abs(final_params), color='steelblue', edgecolor='navy')
    ax.set_xlabel('Parameter Index', fontsize=11)
    ax.set_ylabel('|θ|', fontsize=11)
    ax.set_title('Final Parameter Magnitudes', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Parameter change over time
    ax = axes[2]
    param_changes = np.diff(params_array, axis=0)
    mean_change = np.mean(np.abs(param_changes), axis=1)
    ax.plot(mean_change, color='darkgreen', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Mean |Δθ|', fontsize=11)
    ax.set_title('Average Parameter Change per Epoch', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Plot 4: Parameter heatmap
    ax = axes[3]
    im = ax.imshow(params_array.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Parameter Index', fontsize=11)
    ax.set_title('Parameter Evolution Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Parameter Value')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter evolution plot saved to: {save_path}")

    return fig


def visualize_quantum_states(
    quantum_layer,
    sample_inputs: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize quantum output states for different inputs.

    Args:
        quantum_layer: QuantumLayer instance
        sample_inputs: Batch of input samples
        labels: Optional labels for samples
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    # Get quantum outputs
    with torch.no_grad():
        outputs = quantum_layer(sample_inputs).cpu().numpy()

    n_samples = min(len(sample_inputs), 16)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for idx in range(n_samples):
        ax = axes[idx]

        # Bar plot of expectation values
        expectation_vals = outputs[idx]
        colors = ['red' if val < 0 else 'blue' for val in expectation_vals]

        ax.bar(range(len(expectation_vals)), expectation_vals, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Qubit', fontsize=9)
        ax.set_ylabel('⟨Z⟩', fontsize=9)

        title = f'Sample {idx}'
        if labels is not None:
            title += f' (Label: {labels[idx].item()})'
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(n_samples, 16):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Quantum states visualization saved to: {save_path}")

    return fig


def plot_entanglement_analysis(
    quantum_layer,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze and visualize entanglement in the quantum circuit.

    Args:
        quantum_layer: QuantumLayer instance
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get circuit specs
    specs = quantum_layer.get_circuit_specs()

    # Create connectivity matrix for CNOT gates
    n_qubits = specs['n_qubits']
    n_layers = specs['n_layers']

    # Ring topology: 0→1, 1→2, 2→3, 3→0
    connectivity = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        target = (i + 1) % n_qubits
        connectivity[i, target] = n_layers  # Count across all layers

    # Plot connectivity matrix
    im = ax.imshow(connectivity, cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(n_qubits))
    ax.set_yticks(range(n_qubits))
    ax.set_xlabel('Target Qubit', fontsize=12)
    ax.set_ylabel('Control Qubit', fontsize=12)
    ax.set_title('Qubit Entanglement Connectivity\n(CNOT gates per qubit pair)',
                 fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(n_qubits):
        for j in range(n_qubits):
            text = ax.text(j, i, int(connectivity[i, j]),
                          ha="center", va="center", color="black", fontsize=12)

    plt.colorbar(im, ax=ax, label='Number of CNOT gates')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Entanglement analysis saved to: {save_path}")

    return fig


def create_circuit_summary(quantum_layer) -> str:
    """
    Create a text summary of the quantum circuit.

    Args:
        quantum_layer: QuantumLayer instance

    Returns:
        String with circuit summary
    """
    specs = quantum_layer.get_circuit_specs()

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              QUANTUM CIRCUIT SPECIFICATIONS                   ║
╠══════════════════════════════════════════════════════════════╣
║  Number of Qubits:              {specs['n_qubits']:4d}                      ║
║  Number of Layers:              {specs['n_layers']:4d}                      ║
║  Trainable Parameters:          {specs['n_parameters']:4d}                      ║
║  Device:                        {specs['device']:<20}        ║
║  Differentiation Method:        {specs['diff_method']:<20}   ║
╠══════════════════════════════════════════════════════════════╣
║  Gates per Layer:                                             ║
║    - RY Rotations:              {specs['gates_per_layer']['RY_rotations']:4d}                      ║
║    - CNOT Entanglement:         {specs['gates_per_layer']['CNOT_entanglement']:4d}                      ║
║  Total Gates in Circuit:        {specs['total_gates']:4d}                      ║
╠══════════════════════════════════════════════════════════════╣
║  Architecture:                                                ║
║    Input (4D) → Angle Encoding → Variational Layers          ║
║    → Pauli-Z Measurement → Output (4D)                        ║
║                                                               ║
║  Entanglement Topology: Ring (0→1→2→3→0)                     ║
╚══════════════════════════════════════════════════════════════╝
    """

    return summary


if __name__ == "__main__":
    print("Quantum Visualization Module")
    print("=" * 60)
    print("\nThis module provides visualization utilities for quantum circuits.")
    print("\nAvailable functions:")
    print("  - draw_quantum_circuit(): Draw circuit diagram")
    print("  - visualize_parameter_evolution(): Plot parameter changes")
    print("  - visualize_quantum_states(): Visualize output states")
    print("  - plot_entanglement_analysis(): Analyze entanglement")
    print("  - create_circuit_summary(): Generate text summary")
    print("\nRequires PennyLane to be installed for full functionality.")
    print("=" * 60)
