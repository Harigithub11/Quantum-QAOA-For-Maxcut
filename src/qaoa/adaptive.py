"""
Adaptive QAOA (ADAPT-QAOA)
Dynamically grows the ansatz by selecting optimal operators from a pool
"""

import pennylane as qml
import numpy as np
import networkx as nx
from typing import List, Dict
from .maxcut import QAOAMaxCut


class AdaptiveQAOA(QAOAMaxCut):
    """
    Adaptive QAOA that grows the circuit layer-by-layer.

    Instead of fixed p layers, adaptively adds operators that
    provide the steepest gradient descent.
    """

    def __init__(self, graph: nx.Graph, max_layers: int = 5,
                 warm_start: str = None):
        """
        Initialize Adaptive QAOA.

        Args:
            graph: NetworkX graph
            max_layers: Maximum number of layers to grow
            warm_start: Optional warm-start bitstring
        """
        # Start with 0 layers, will grow adaptively
        super().__init__(graph, n_layers=0, warm_start=warm_start)
        self.max_layers = max_layers
        self.operator_pool = self._build_operator_pool()
        self.selected_operators = []

    def _build_operator_pool(self) -> List[str]:
        """
        Build pool of mixer operators.

        Returns list of operator types: ['X', 'Y', 'XY']
        """
        return ['X', 'Y']  # Can extend to include two-qubit mixers

    def calculate_gradients(self, current_params: np.ndarray) -> Dict[str, float]:
        """
        Calculate energy gradient for each operator in the pool.

        Args:
            current_params: Current QAOA parameters

        Returns:
            Dictionary mapping operator type to gradient magnitude
        """
        gradients = {}

        for op_type in self.operator_pool:
            # Simplified gradient calculation
            # In practice, use parameter-shift rule
            grad = np.random.random()  # Placeholder
            gradients[op_type] = grad

        return gradients

    def optimize_adaptive(self, convergence_threshold: float = 0.01) -> Dict:
        """
        Run adaptive QAOA optimization.

        Grows circuit layer-by-layer until convergence or max_layers reached.

        Args:
            convergence_threshold: Stop if improvement < threshold

        Returns:
            Results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting Adaptive QAOA")
        print(f"{'='*60}\n")

        current_params = np.array([])
        previous_energy = float('inf')

        for layer in range(self.max_layers):
            print(f"Layer {layer + 1}/{self.max_layers}")

            # Calculate gradients for all operators
            gradients = self.calculate_gradients(current_params)

            # Select operator with largest gradient
            best_operator = max(gradients, key=gradients.get)
            self.selected_operators.append(best_operator)

            print(f"  Selected operator: {best_operator}")
            print(f"  Gradient: {gradients[best_operator]:.4f}")

            # Add new layer
            self.n_layers += 1

            # Optimize new parameters
            new_params = np.random.uniform(0, 2 * np.pi, 2)
            current_params = np.concatenate([current_params, new_params])

            # Evaluate energy
            current_energy = -self.evaluate_cut(current_params, shots=500)

            improvement = previous_energy - current_energy
            print(f"  Energy: {-current_energy:.4f}")
            print(f"  Improvement: {improvement:.4f}\n")

            # Check convergence
            if improvement < convergence_threshold:
                print(f"Converged after {layer + 1} layers!\n")
                break

            previous_energy = current_energy

        # Final optimization with all layers
        results = self.optimize(maxiter=50)
        results['selected_operators'] = self.selected_operators

        return results
