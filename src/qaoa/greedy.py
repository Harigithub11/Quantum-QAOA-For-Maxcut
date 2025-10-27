"""
Classical Greedy Heuristic for MaxCut
Provides warm-start solution for QAOA
"""

import networkx as nx
import numpy as np
from typing import Tuple, Set


def greedy_maxcut(graph: nx.Graph) -> Tuple[Set[int], Set[int], int]:
    """
    Greedy algorithm for MaxCut problem.

    Iteratively places each vertex into the partition that maximizes
    the immediate increase in cut size.

    Args:
        graph: NetworkX graph

    Returns:
        partition_A: Set of nodes in first partition
        partition_B: Set of nodes in second partition
        cut_size: Size of the cut (number of edges crossing)
    """
    partition_A = set()
    partition_B = set()

    # Sort nodes for deterministic behavior
    nodes = sorted(graph.nodes())

    for node in nodes:
        # Count edges to each partition
        edges_to_A = sum(1 for neighbor in graph.neighbors(node)
                        if neighbor in partition_A)
        edges_to_B = sum(1 for neighbor in graph.neighbors(node)
                        if neighbor in partition_B)

        # Place node in partition that maximizes cut
        if edges_to_B >= edges_to_A:
            partition_A.add(node)
        else:
            partition_B.add(node)

    # Calculate cut size
    cut_size = calculate_cut_size(graph, partition_A, partition_B)

    return partition_A, partition_B, cut_size


def calculate_cut_size(graph: nx.Graph, partition_A: Set[int],
                       partition_B: Set[int]) -> int:
    """Calculate the number of edges crossing the partition."""
    cut_size = 0
    for u, v in graph.edges():
        if (u in partition_A and v in partition_B) or \
           (u in partition_B and v in partition_A):
            cut_size += 1
    return cut_size


def partition_to_bitstring(graph: nx.Graph, partition_A: Set[int],
                          partition_B: Set[int]) -> str:
    """
    Convert partition to bitstring.

    Args:
        graph: NetworkX graph
        partition_A: Nodes in partition A (assigned '0')
        partition_B: Nodes in partition B (assigned '1')

    Returns:
        Bitstring representation (e.g., '010110')
    """
    n_nodes = len(graph.nodes())
    bitstring = ['0'] * n_nodes

    for node in partition_B:
        bitstring[node] = '1'

    return ''.join(bitstring)


def bitstring_to_partition(bitstring: str) -> Tuple[Set[int], Set[int]]:
    """
    Convert bitstring to partition.

    Args:
        bitstring: Binary string (e.g., '010110')

    Returns:
        partition_A: Nodes with bit '0'
        partition_B: Nodes with bit '1'
    """
    partition_A = {i for i, bit in enumerate(bitstring) if bit == '0'}
    partition_B = {i for i, bit in enumerate(bitstring) if bit == '1'}

    return partition_A, partition_B
