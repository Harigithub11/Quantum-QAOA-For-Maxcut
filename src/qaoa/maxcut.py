"""
QAOA MaxCut Solver
Implements Quantum Approximate Optimization Algorithm for MaxCut problem
"""

import pennylane as qml
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize
import time


class QAOAMaxCut:
    """
    QAOA solver for the MaxCut problem.

    Combines quantum circuits with classical optimization to find
    approximate solutions to the MaxCut problem on graphs.
    """

    def __init__(self, graph: nx.Graph, n_layers: int = 3,
                 warm_start: Optional[str] = None):
        """
        Initialize QAOA solver.

        Args:
            graph: NetworkX graph
            n_layers: Number of QAOA layers (p parameter)
            warm_start: Optional bitstring for warm-starting
        """
        self.graph = graph
        self.n_qubits = len(graph.nodes())
        self.n_layers = n_layers
        self.warm_start = warm_start
        self.edges = list(graph.edges())

        # Create quantum device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        # Build cost Hamiltonian
        self.cost_hamiltonian = self._build_cost_hamiltonian()

        # Optimization history
        self.history = {
            'params': [],
            'energies': [],
            'bitstrings': [],
            'cut_sizes': [],
            'times': []
        }

        self.start_time = None

    def _build_cost_hamiltonian(self) -> List[Tuple]:
        """
        Build MaxCut cost Hamiltonian.

        For each edge (i,j), add term: -0.5 * (1 - Z_i Z_j)
        This encourages qubits to have opposite values.
        """
        hamiltonian = []
        for i, j in self.edges:
            hamiltonian.append((0.5, qml.Identity(0)))  # Constant term
            hamiltonian.append((-0.5, qml.PauliZ(i) @ qml.PauliZ(j)))
        return hamiltonian

    def cost_layer(self, gamma: float):
        """Apply cost Hamiltonian layer."""
        for i, j in self.edges:
            qml.CNOT(wires=[i, j])
            qml.RZ(gamma, wires=j)
            qml.CNOT(wires=[i, j])

    def mixer_layer(self, beta: float):
        """Apply mixer Hamiltonian layer (standard X mixer)."""
        for qubit in range(self.n_qubits):
            qml.RX(2 * beta, wires=qubit)

    def qaoa_circuit(self, params: np.ndarray):
        """
        Build complete QAOA circuit (without measurements).

        Args:
            params: Parameters [gamma_1, beta_1, ..., gamma_p, beta_p]
        """
        # Initial state preparation
        if self.warm_start is not None:
            # Warm-start: prepare |bitstring⟩
            for i, bit in enumerate(self.warm_start):
                if bit == '1':
                    qml.PauliX(wires=i)
        else:
            # Standard: prepare |+⟩^n (equal superposition)
            for qubit in range(self.n_qubits):
                qml.Hadamard(wires=qubit)

        # Apply p layers of cost + mixer
        for layer in range(self.n_layers):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            self.cost_layer(gamma)
            self.mixer_layer(beta)

    def cost_function(self, params: np.ndarray) -> float:
        """
        Evaluate QAOA cost function (expectation value of cost Hamiltonian).
        """
        # Simplified: return negative cut size to minimize
        return -self.evaluate_cut(params)

    def evaluate_cut(self, params: np.ndarray, shots: int = 1000) -> float:
        """
        Evaluate cut size by sampling from quantum circuit.

        Args:
            params: QAOA parameters
            shots: Number of measurement shots

        Returns:
            Average cut size
        """
        # Create sampler circuit using counts
        @qml.qnode(self.dev, interface='autograd')
        def sampler(params):
            self.qaoa_circuit(params)
            return qml.counts(wires=range(self.n_qubits))

        # Get counts dictionary
        counts_dict = sampler(params, shots=shots)

        # Calculate average cut size from counts
        total_cut = 0
        total_shots = 0
        for bitstring, count in counts_dict.items():
            # Convert tuple of measurements to bitstring
            if isinstance(bitstring, tuple):
                bitstring_str = ''.join(['1' if int(bit) == 1 else '0' for bit in bitstring])
            else:
                bitstring_str = bitstring
            cut = self._calculate_cut_from_bitstring(bitstring_str)
            total_cut += cut * count
            total_shots += count

        return total_cut / total_shots if total_shots > 0 else 0

    def _calculate_cut_from_bitstring(self, bitstring: str) -> int:
        """Calculate cut size from a bitstring."""
        cut_size = 0
        for i, j in self.edges:
            if bitstring[i] != bitstring[j]:
                cut_size += 1
        return cut_size

    def optimize(self, maxiter: int = 100, method: str = 'COBYLA') -> Dict:
        """
        Run classical optimization to find optimal QAOA parameters.

        Args:
            maxiter: Maximum optimization iterations
            method: Classical optimizer (COBYLA, BFGS, etc.)

        Returns:
            Results dictionary
        """
        self.start_time = time.time()

        # Initialize parameters randomly
        np.random.seed(42)
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * self.n_layers)

        # Define objective function for scipy
        def objective(params):
            cost = -self.evaluate_cut(params, shots=500)

            # Store history
            elapsed = time.time() - self.start_time
            self.history['params'].append(params.copy())
            self.history['energies'].append(-cost)  # Store positive cut size
            self.history['times'].append(elapsed)

            return cost

        # Run optimization
        print(f"\n{'='*60}")
        print(f"Starting QAOA Optimization")
        print(f"{'='*60}")
        print(f"Graph: {self.n_qubits} nodes, {len(self.edges)} edges")
        print(f"Layers: {self.n_layers}")
        print(f"Warm-start: {'Yes' if self.warm_start else 'No'}")
        print(f"{'='*60}\n")

        result = minimize(objective, initial_params, method=method,
                         options={'maxiter': maxiter, 'disp': True})

        # Get final solution
        optimal_params = result.x
        final_bitstring = self._get_best_bitstring(optimal_params)
        final_cut = self._calculate_cut_from_bitstring(final_bitstring)

        elapsed_time = time.time() - self.start_time

        # Get number of iterations (different attributes for different optimizers)
        n_iterations = getattr(result, 'nit', getattr(result, 'nfev', len(self.history['energies'])))

        results = {
            'optimal_params': optimal_params,
            'best_bitstring': final_bitstring,
            'cut_size': final_cut,
            'max_possible_cut': len(self.edges),
            'approximation_ratio': final_cut / len(self.edges),
            'n_iterations': n_iterations,
            'elapsed_time': elapsed_time,
            'history': self.history
        }

        print(f"\n{'='*60}")
        print(f"QAOA Optimization Complete")
        print(f"{'='*60}")
        print(f"Best Cut Size: {final_cut}/{len(self.edges)}")
        print(f"Approximation Ratio: {results['approximation_ratio']:.4f}")
        print(f"Iterations: {n_iterations}")
        print(f"Time: {elapsed_time:.2f}s")
        print(f"{'='*60}\n")

        return results

    def _get_best_bitstring(self, params: np.ndarray, shots: int = 2000) -> str:
        """Sample the circuit and return the most frequent bitstring."""
        @qml.qnode(self.dev, interface='autograd')
        def sampler(params):
            self.qaoa_circuit(params)
            return qml.counts(wires=range(self.n_qubits))

        # Get counts dictionary
        counts_dict = sampler(params, shots=shots)

        # Find most frequent bitstring
        best_bitstring = None
        max_count = 0
        for bitstring, count in counts_dict.items():
            if count > max_count:
                max_count = count
                # Convert tuple of measurements to bitstring
                if isinstance(bitstring, tuple):
                    best_bitstring = ''.join(['1' if int(bit) == 1 else '0' for bit in bitstring])
                else:
                    best_bitstring = bitstring

        return best_bitstring
