# MaxCut Benchmark Datasets

## Standard Small Benchmarks (Perfect for QAOA)

### Small Gset-like Graphs
- `small_gset_10nodes.txt` - 10 nodes, random weighted graph
- `small_gset_12nodes.txt` - 12 nodes, random weighted graph
- `small_gset_15nodes.txt` - 15 nodes, random weighted graph
- `small_gset_20nodes.txt` - 20 nodes, random weighted graph

**Properties:**
- 50% edge density
- Random weights (1-9)
- Good for testing QAOA performance
- Similar to Gset benchmark structure but smaller

### Known Optimal Benchmarks
- `k5_5_bipartite.txt` - Complete bipartite K_{5,5}
  - **Optimal cut: 25 edges (100%)**
  - Tests if algorithm recognizes bipartite structure

### Greedy-Suboptimal Benchmarks
- `cycle_9.txt` - 9-node cycle
- `cycle_11.txt` - 11-node cycle
- `cycle_13.txt` - 13-node cycle
  - **Property:** Odd cycles make greedy suboptimal
  - **Perfect for showing quantum advantage!**

### Famous Graphs
- `petersen.txt` - Petersen graph (10 nodes, 15 edges)
  - Famous hard graph from graph theory
  - Highly symmetric

### Real-World Networks
- `karate_club.txt` - Zachary's Karate Club (34 nodes, 78 edges)
  - Real social network from 1977 study
  - Standard benchmark in network analysis

## File Format

All files use simple edge list format:
```
# Comment line
node1 node2 [weight]
node1 node3 [weight]
...
```

## Sources

- Gset: Stanford University (Helmberg & Rendl)
- Standard graphs: NetworkX / Graph Theory
- Karate Club: Zachary (1977)

## For QAOA Testing

Recommended order:
1. **cycle_9.txt** - Show quantum advantage over greedy
2. **k5_5_bipartite.txt** - Validate optimality detection
3. **small_gset_12nodes.txt** - Realistic benchmark
4. **petersen.txt** - Famous hard graph
5. **karate_club.txt** - Real-world application

All graphs are small enough for classical QAOA simulation!
