"""
QAOA MaxCut Benchmark Evaluation
Tests QAOA on standard benchmark datasets and reports accuracy
"""

import pandas as pd
import numpy as np
import time
from src.qaoa.datasets import MaxCutDatasets, create_benchmark_suite
from src.qaoa.greedy import greedy_maxcut, partition_to_bitstring
from src.qaoa.maxcut import QAOAMaxCut


def evaluate_graph(graph, n_layers=3, maxiter=50, verbose=True):
    """
    Evaluate QAOA vs Greedy on a single graph

    Returns:
        dict with results
    """
    if verbose:
        print(f"\nEvaluating: {graph.name}")
        print(f"  Nodes: {len(graph.nodes())}, Edges: {len(graph.edges())}")

    # Run greedy algorithm
    greedy_start = time.time()
    partition_A_greedy, partition_B_greedy, greedy_cut = greedy_maxcut(graph)
    greedy_time = time.time() - greedy_start

    greedy_ratio = greedy_cut / len(graph.edges()) if len(graph.edges()) > 0 else 0

    if verbose:
        print(f"  Greedy: {greedy_cut}/{len(graph.edges())} = {greedy_ratio:.1%} in {greedy_time:.4f}s")

    # Run QAOA
    warm_start = partition_to_bitstring(graph, partition_A_greedy, partition_B_greedy)
    qaoa = QAOAMaxCut(graph, n_layers=n_layers, warm_start=warm_start)

    qaoa_start = time.time()
    results = qaoa.optimize(maxiter=maxiter)
    qaoa_time = time.time() - qaoa_start

    qaoa_ratio = results['approximation_ratio']

    if verbose:
        print(f"  QAOA:   {results['cut_size']}/{len(graph.edges())} = {qaoa_ratio:.1%} in {qaoa_time:.2f}s")

    # Calculate improvement
    improvement_edges = results['cut_size'] - greedy_cut
    improvement_pct = (improvement_edges / greedy_cut * 100) if greedy_cut > 0 else 0

    if verbose:
        if improvement_edges > 0:
            print(f"  Result: QUANTUM ADVANTAGE! +{improvement_edges} edges (+{improvement_pct:.1f}%)")
        elif improvement_edges == 0:
            print(f"  Result: Tied (both found same solution)")
        else:
            print(f"  Result: Greedy was better by {-improvement_edges} edges")

    return {
        'graph_name': graph.name,
        'nodes': len(graph.nodes()),
        'edges': len(graph.edges()),
        'greedy_cut': greedy_cut,
        'greedy_ratio': greedy_ratio,
        'greedy_time': greedy_time,
        'qaoa_cut': results['cut_size'],
        'qaoa_ratio': qaoa_ratio,
        'qaoa_time': qaoa_time,
        'improvement_edges': improvement_edges,
        'improvement_pct': improvement_pct,
        'qaoa_iterations': results['n_iterations'],
        'best_bitstring': results['best_bitstring']
    }


def run_benchmark_suite(n_layers=3, maxiter=50):
    """
    Run complete benchmark suite
    """
    print("="*80)
    print("QAOA MAXCUT BENCHMARK EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  QAOA Layers: {n_layers}")
    print(f"  Max Iterations: {maxiter}")
    print(f"  Warm-start: Enabled (Greedy)")
    print("="*80)

    # Create benchmark suite
    benchmark = create_benchmark_suite()
    results = []

    # Evaluate each graph
    for name, graph in benchmark:
        try:
            result = evaluate_graph(graph, n_layers=n_layers, maxiter=maxiter, verbose=True)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Create results dataframe
    df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    print(f"\nTotal Graphs Tested: {len(results)}")
    print(f"QAOA Better: {sum(df['improvement_edges'] > 0)}")
    print(f"Tied: {sum(df['improvement_edges'] == 0)}")
    print(f"Greedy Better: {sum(df['improvement_edges'] < 0)}")

    print(f"\nAverage Greedy Ratio: {df['greedy_ratio'].mean():.1%}")
    print(f"Average QAOA Ratio: {df['qaoa_ratio'].mean():.1%}")
    print(f"Average Improvement: {df['improvement_pct'].mean():.2f}%")

    print(f"\nAverage Greedy Time: {df['greedy_time'].mean():.4f}s")
    print(f"Average QAOA Time: {df['qaoa_time'].mean():.2f}s")
    print(f"Speedup Factor: {df['qaoa_time'].mean() / df['greedy_time'].mean():.1f}x slower")

    # Best improvements
    print("\n" + "="*80)
    print("TOP 5 QUANTUM ADVANTAGES")
    print("="*80)

    top5 = df.nlargest(5, 'improvement_pct')[['graph_name', 'nodes', 'edges',
                                                'greedy_ratio', 'qaoa_ratio',
                                                'improvement_edges', 'improvement_pct']]
    print(top5.to_string(index=False))

    # Detailed results table
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    print(f"\n{'Graph':<15} {'Size':>8} {'Greedy':>10} {'QAOA':>10} {'Improvement':>12} {'Status':>10}")
    print("-"*80)

    for _, row in df.iterrows():
        status = "BETTER" if row['improvement_edges'] > 0 else ("TIED" if row['improvement_edges'] == 0 else "WORSE")
        improvement_str = f"+{row['improvement_edges']}" if row['improvement_edges'] >= 0 else str(row['improvement_edges'])

        print(f"{row['graph_name']:<15} {row['nodes']:>3}N {row['edges']:>3}E "
              f"{row['greedy_cut']:>3}/{row['edges']:<3} "
              f"{row['qaoa_cut']:>3}/{row['edges']:<3} "
              f"{improvement_str:>4} ({row['improvement_pct']:>5.1f}%) "
              f"{status:>10}")

    # Save results
    output_file = 'qaoa_benchmark_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return df


def quick_benchmark():
    """Quick benchmark on small graphs"""
    print("="*80)
    print("QUICK QAOA BENCHMARK (Small Graphs)")
    print("="*80)

    datasets = MaxCutDatasets()

    # Small test suite
    test_graphs = [
        ("Pentagon", datasets.load_cycle_graph(5)),
        ("Heptagon", datasets.load_cycle_graph(7)),
        ("K_3,3", datasets.load_complete_bipartite(3, 3)),
        ("Grid-3x3", datasets.load_grid_graph(3, 3)),
        ("Random-8", datasets.load_random_graph(8, p=0.5, seed=17)),
        ("Petersen", datasets.load_petersen_graph())
    ]

    results = []
    for name, graph in test_graphs:
        result = evaluate_graph(graph, n_layers=3, maxiter=40, verbose=True)
        results.append(result)

    df = pd.DataFrame(results)

    # Summary
    print("\n" + "="*80)
    print("QUICK BENCHMARK SUMMARY")
    print("="*80)

    qaoa_wins = sum(df['improvement_edges'] > 0)
    ties = sum(df['improvement_edges'] == 0)

    print(f"\nQAOA Better: {qaoa_wins}/{len(results)} ({qaoa_wins/len(results)*100:.0f}%)")
    print(f"Tied: {ties}/{len(results)}")
    print(f"Average Improvement: {df['improvement_pct'].mean():.2f}%")

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick benchmark
        df = quick_benchmark()
    else:
        # Full benchmark
        print("\nRunning full benchmark suite...")
        print("This may take 5-10 minutes depending on graph sizes.")
        print("\nFor a quick test, run: python benchmark_qaoa.py quick\n")

        response = input("Continue with full benchmark? (y/n): ")
        if response.lower() == 'y':
            df = run_benchmark_suite(n_layers=3, maxiter=50)
        else:
            print("\nRunning quick benchmark instead...\n")
            df = quick_benchmark()

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
