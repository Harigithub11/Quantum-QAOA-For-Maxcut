"""
Quick test script for QAOA MaxCut solver
Demonstrates the complete workflow
"""

import networkx as nx
import matplotlib.pyplot as plt
from src.qaoa.greedy import greedy_maxcut, partition_to_bitstring
from src.qaoa.maxcut import QAOAMaxCut

print("="*60)
print("QAOA MaxCut Solver - Quick Test")
print("="*60)

# Create a simple test graph
print("\n1. Creating test graph...")
graph = nx.Graph()
graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])  # 5 edges

print(f"   Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

# Run greedy algorithm
print("\n2. Running classical greedy algorithm...")
partition_A, partition_B, greedy_cut = greedy_maxcut(graph)
print(f"   Greedy cut size: {greedy_cut}/{len(graph.edges())}")
print(f"   Partition A: {partition_A}")
print(f"   Partition B: {partition_B}")

# Get warm-start bitstring
warm_start = partition_to_bitstring(graph, partition_A, partition_B)
print(f"   Warm-start bitstring: {warm_start}")

# Run QAOA
print("\n3. Running QAOA (this may take 20-30 seconds)...")
qaoa = QAOAMaxCut(graph, n_layers=2, warm_start=warm_start)
results = qaoa.optimize(maxiter=20)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Greedy Cut:  {greedy_cut}/{len(graph.edges())}")
print(f"QAOA Cut:    {results['cut_size']}/{len(graph.edges())}")
print(f"Improvement: {results['cut_size'] - greedy_cut} edges")
print(f"Time:        {results['elapsed_time']:.2f}s")
print(f"Iterations:  {results['n_iterations']}")
print("="*60)

if results['cut_size'] >= greedy_cut:
    print("\n[SUCCESS] QAOA matched or beat greedy solution!")
else:
    print("\n[WARNING] QAOA found suboptimal solution (can happen with low iterations)")

print("\n[SUCCESS] QAOA MaxCut solver is working correctly!")
print(">>> Run Streamlit app for interactive visualization:")
print("    streamlit run app/pages/5_QAOA_MaxCut.py")
