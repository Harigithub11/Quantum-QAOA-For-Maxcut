"""
Graph Datasets for QAOA MaxCut Benchmarking
Provides standard benchmark graphs from literature
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple, List
import json
import os


class MaxCutDatasets:
    """Collection of benchmark graphs for MaxCut problem"""

    @staticmethod
    def get_dataset_info() -> Dict[str, Dict]:
        """Return information about available datasets"""
        return {
            'gset': {
                'name': 'Gset Benchmark',
                'description': 'Standard MaxCut benchmark from Stanford GraphBase',
                'source': 'Goemans & Williamson (1995)',
                'sizes': [20, 30, 40, 50],
                'difficulty': 'Medium to Hard'
            },
            'random': {
                'name': 'Random Graphs',
                'description': 'Erdős-Rényi random graphs',
                'source': 'Generated',
                'sizes': [10, 15, 20],
                'difficulty': 'Medium'
            },
            'cycles': {
                'name': 'Cycle Graphs',
                'description': 'Simple cycle graphs (greedy often suboptimal)',
                'source': 'Graph Theory',
                'sizes': [5, 7, 9, 11],
                'difficulty': 'Easy to Medium'
            },
            'complete_bipartite': {
                'name': 'Complete Bipartite',
                'description': 'K_{n,m} graphs (optimal solution is all edges)',
                'source': 'Graph Theory',
                'sizes': [(3,3), (4,4), (5,5)],
                'difficulty': 'Medium (testing bipartite detection)'
            },
            'petersen': {
                'name': 'Petersen Graph',
                'description': 'Famous hard graph from graph theory',
                'source': 'Graph Theory',
                'sizes': [10],
                'difficulty': 'Hard'
            },
            'karate_club': {
                'name': 'Karate Club Network',
                'description': 'Real social network (Zachary, 1977)',
                'source': 'Social Network Analysis',
                'sizes': [34],
                'difficulty': 'Medium'
            },
            'grid': {
                'name': '2D Grid Graphs',
                'description': '2D lattice structure',
                'source': 'Graph Theory',
                'sizes': [(3,3), (4,4), (5,5)],
                'difficulty': 'Medium'
            }
        }

    @staticmethod
    def load_cycle_graph(n: int) -> nx.Graph:
        """
        Load cycle graph (n nodes in a ring)
        Known property: Odd cycles make greedy suboptimal
        """
        graph = nx.cycle_graph(n)
        graph.name = f"Cycle-{n}"
        return graph

    @staticmethod
    def load_complete_bipartite(m: int, n: int) -> nx.Graph:
        """
        Load complete bipartite graph K_{m,n}
        Optimal solution: All m*n edges (100% cut)
        """
        graph = nx.complete_bipartite_graph(m, n)
        graph.name = f"K_{{{m},{n}}}"
        return graph

    @staticmethod
    def load_petersen_graph() -> nx.Graph:
        """
        Load Petersen graph (famous hard graph)
        10 nodes, 15 edges, highly symmetric
        """
        graph = nx.petersen_graph()
        graph.name = "Petersen"
        return graph

    @staticmethod
    def load_karate_club() -> nx.Graph:
        """
        Load Zachary's Karate Club network
        Real-world social network: 34 nodes, 78 edges
        Famous benchmark for community detection
        """
        graph = nx.karate_club_graph()
        graph.name = "Karate Club"
        return graph

    @staticmethod
    def load_random_graph(n: int, p: float = 0.5, seed: int = 42) -> nx.Graph:
        """
        Load random Erdős-Rényi graph
        """
        graph = nx.gnp_random_graph(n, p, seed=seed)
        graph.name = f"Random-{n}-p{p}"
        return graph

    @staticmethod
    def load_grid_graph(m: int, n: int) -> nx.Graph:
        """
        Load 2D grid graph
        m x n lattice structure
        """
        graph = nx.grid_2d_graph(m, n)
        # Convert node labels to integers
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping)
        graph.name = f"Grid-{m}x{n}"
        return graph

    @staticmethod
    def load_gset_graph(n: int, density: float = 0.5, seed: int = 42) -> nx.Graph:
        """
        Generate Gset-like benchmark graph
        (Approximation of Stanford GraphBase benchmarks)
        """
        # Create weighted random graph
        np.random.seed(seed)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        # Add edges with probability density
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < density:
                    # Add edge with random weight (Gset graphs are weighted)
                    weight = np.random.randint(1, 10)
                    graph.add_edge(i, j, weight=weight)

        graph.name = f"Gset-{n}"
        return graph

    @staticmethod
    def get_graph_stats(graph: nx.Graph) -> Dict:
        """Calculate graph statistics"""
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())
        density = nx.density(graph) if n_nodes > 1 else 0

        # Calculate degree statistics
        degrees = [d for _, d in graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        # Check properties
        is_connected = nx.is_connected(graph)
        is_bipartite = nx.is_bipartite(graph)

        # Theoretical max cut for bipartite
        theoretical_max = n_edges if is_bipartite else None

        return {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': density,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'is_connected': is_connected,
            'is_bipartite': is_bipartite,
            'theoretical_max_cut': theoretical_max,
            'difficulty_estimate': _estimate_difficulty(n_nodes, density)
        }


def _estimate_difficulty(n_nodes: int, density: float) -> str:
    """Estimate problem difficulty"""
    if n_nodes <= 5:
        return "Easy (Small graph)"
    elif n_nodes <= 10:
        if density < 0.3:
            return "Easy (Sparse)"
        elif density > 0.7:
            return "Medium (Dense)"
        else:
            return "Medium (Moderate density)"
    elif n_nodes <= 20:
        if density < 0.3:
            return "Medium (Sparse, medium size)"
        else:
            return "Hard (Dense, medium size)"
    else:
        return "Very Hard (Large graph)"


class DatasetLoader:
    """Load graphs from various file formats"""

    @staticmethod
    def from_edge_list(filepath: str) -> nx.Graph:
        """
        Load graph from edge list file
        Format: Each line is "node1 node2" or "node1 node2 weight"
        """
        graph = nx.Graph()

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    weight = float(parts[2]) if len(parts) > 2 else 1.0
                    graph.add_edge(u, v, weight=weight)

        graph.name = os.path.basename(filepath).replace('.txt', '')
        return graph

    @staticmethod
    def from_adjacency_matrix(filepath: str) -> nx.Graph:
        """
        Load graph from adjacency matrix file
        Format: Space-separated matrix
        """
        matrix = np.loadtxt(filepath)
        graph = nx.from_numpy_array(matrix)
        graph.name = os.path.basename(filepath).replace('.txt', '')
        return graph

    @staticmethod
    def from_json(filepath: str) -> nx.Graph:
        """
        Load graph from JSON file
        Format: {"nodes": [...], "edges": [[u, v], ...]}
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        graph = nx.Graph()

        if 'nodes' in data:
            graph.add_nodes_from(data['nodes'])

        if 'edges' in data:
            for edge in data['edges']:
                if len(edge) == 2:
                    graph.add_edge(edge[0], edge[1])
                elif len(edge) == 3:
                    graph.add_edge(edge[0], edge[1], weight=edge[2])

        graph.name = os.path.basename(filepath).replace('.json', '')
        return graph


def create_benchmark_suite() -> List[Tuple[str, nx.Graph]]:
    """
    Create a comprehensive benchmark suite
    Returns list of (name, graph) tuples
    """
    datasets = MaxCutDatasets()
    benchmark = []

    # Cycle graphs (good for showing greedy weakness)
    for n in [5, 7, 9]:
        graph = datasets.load_cycle_graph(n)
        benchmark.append((f"Cycle-{n}", graph))

    # Complete bipartite (test bipartite detection)
    for m, n in [(3, 3), (4, 4)]:
        graph = datasets.load_complete_bipartite(m, n)
        benchmark.append((f"K_{m},{n}", graph))

    # Random graphs
    for n in [8, 10, 12]:
        graph = datasets.load_random_graph(n, p=0.5, seed=42)
        benchmark.append((f"Random-{n}", graph))

    # Grid graphs
    for m, n in [(3, 3), (4, 4)]:
        graph = datasets.load_grid_graph(m, n)
        benchmark.append((f"Grid-{m}x{n}", graph))

    # Famous graphs
    benchmark.append(("Petersen", datasets.load_petersen_graph()))
    benchmark.append(("Karate Club", datasets.load_karate_club()))

    return benchmark


if __name__ == "__main__":
    # Demo
    print("="*80)
    print("QAOA MaxCut Benchmark Datasets")
    print("="*80)

    datasets = MaxCutDatasets()

    # Show available datasets
    print("\nAvailable Dataset Categories:")
    for key, info in datasets.get_dataset_info().items():
        print(f"\n{info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Source: {info['source']}")
        print(f"  Sizes: {info['sizes']}")
        print(f"  Difficulty: {info['difficulty']}")

    # Load and show stats for some graphs
    print("\n" + "="*80)
    print("Example Graphs and Statistics:")
    print("="*80)

    test_graphs = [
        datasets.load_cycle_graph(7),
        datasets.load_complete_bipartite(3, 3),
        datasets.load_petersen_graph(),
        datasets.load_karate_club()
    ]

    for graph in test_graphs:
        stats = datasets.get_graph_stats(graph)
        print(f"\n{graph.name}:")
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Edges: {stats['edges']}")
        print(f"  Density: {stats['density']:.3f}")
        print(f"  Avg Degree: {stats['avg_degree']:.2f}")
        print(f"  Bipartite: {stats['is_bipartite']}")
        if stats['theoretical_max_cut']:
            print(f"  Theoretical Max Cut: {stats['theoretical_max_cut']} (100%)")
        print(f"  Difficulty: {stats['difficulty_estimate']}")
