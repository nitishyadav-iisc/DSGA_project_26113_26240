# Property-Based Testing of NetworkX Graph Algorithms

**Course:** Data Structure and Graph Algorithms (DSGA)

## Authors

| Author             | SR No.                    |
|--------------------|-------------------------- |
| Aswin S            | 13-19-01-19-52-25-1-26240 |
| Nitish Kumar Yadav | 13-19-01-19-52-25-1-26113 |

---

## Project Overview

This project applies **property-based testing** (using the [Hypothesis](https://hypothesis.readthedocs.io/) framework) to verify the correctness of classical graph algorithms implemented in [NetworkX](https://networkx.org/). Rather than writing example-based tests with hand-picked inputs, property-based tests specify universal invariants that must hold for *all* valid inputs. Hypothesis then automatically generates hundreds of randomised graph instances to stress-test those invariants.

Two core NetworkX algorithms are tested:

1. **Prim's Minimum Spanning Tree** (`nx.minimum_spanning_tree` with `algorithm="prim"`)
2. **Dijkstra's Shortest Path** (`nx.dijkstra_path`, `nx.dijkstra_path_length`, `nx.single_source_dijkstra_path_length`)

---

## Test Framework & Graph Generation

All tests use **Hypothesis composite strategies** to generate random connected, undirected, positively-weighted graphs:

- **Node count** is drawn uniformly from a configurable range (typically 2–15 or 2–30).
- **Connectivity** is guaranteed by first constructing a random spanning path (a permutation of nodes chained together with weighted edges).
- **Extra edges** are added probabilistically, producing graphs that range from near-tree (sparse) to near-complete (dense).
- **Edge weights** are strictly positive floats (e.g., 0.01–100.0), ensuring Dijkstra's non-negative weight precondition is satisfied and reducing degenerate tie scenarios.

---

## Algorithms Tested

### 1. Prim's Minimum Spanning Tree Algorithm

**NetworkX API:** `nx.minimum_spanning_tree(G, algorithm="prim")`

Prim's algorithm computes a Minimum Spanning Tree (MST) of a connected, weighted, undirected graph. Starting from an arbitrary root, it greedily grows a subtree by always picking the lightest edge crossing the cut between visited and unvisited vertices. The following properties are tested:

#### Property Tests

| # | Property | Description |
|---|----------|-------------|
| 1 | **Edge Count (n − 1)** | A spanning tree of a connected graph with *n* nodes must have exactly *n − 1* edges. Fewer edges means the tree is not spanning; more edges means it contains a cycle and is not a tree. |
| 2 | **Connectivity & Spanning** | The MST must be connected (`nx.is_connected`) and its vertex set must equal the vertex set of the original graph. This ensures the result truly *spans* the input. |
| 3 | **Subgraph Property** | Every edge in the MST must exist in the original graph. Prim's algorithm selects edges exclusively from the input, so the MST must be a subgraph of *G*. |
| 4 | **Acyclicity** | The MST must contain no cycles. This is verified by checking that `nx.cycle_basis(mst)` returns an empty list. A tree is, by definition, a connected acyclic graph. |
| 5 | **Weight Optimality (Metamorphic — Prim vs Kruskal)** | The total MST weight computed by Prim's algorithm must equal the total MST weight computed by Kruskal's algorithm on the same graph. Both are provably optimal, and while the specific edge sets may differ when ties exist, the total weight of any MST is unique for a given graph. |

---

### 2. Dijkstra's Shortest Path Algorithm

**NetworkX APIs:** `nx.dijkstra_path()`, `nx.dijkstra_path_length()`, `nx.single_source_dijkstra_path_length()`

Dijkstra's algorithm computes the shortest (minimum total weight) path between nodes in a graph with non-negative edge weights. It uses a priority queue to greedily finalize the closest unvisited node at each step. The following properties are tested:

#### Property Tests

| # | Property | Description |
|---|----------|-------------|
| 1 | **Self-Distance is Zero** | The shortest-path distance from any node to itself is exactly 0. The empty path has cost 0, and with strictly positive edge weights, no cycle can reduce this below zero. |
| 2 | **Non-Negative Distances** | All single-source shortest-path distances must be ≥ 0 when all edge weights are non-negative. This follows from the fact that Dijkstra's relaxation only adds non-negative weights to non-negative estimates. |
| 3 | **Symmetry (Undirected Graphs)** | In an undirected graph, `dist(u, v) == dist(v, u)`. Every edge is traversable in both directions at equal cost, so the shortest path reversed has the same total weight. |
| 4 | **Triangle Inequality** | For all nodes *u*, *v*, *w*: `dist(u, v) ≤ dist(u, w) + dist(w, v)`. If this were violated, concatenating the paths *u→w* and *w→v* would yield a cheaper *u→v* path, contradicting optimality. |
| 5 | **Subpath Optimality (Optimal Substructure)** | Every prefix of a shortest path is itself a shortest path. If the shortest path from *u* to *v* passes through intermediate node *m*, then the sub-path *u→m* must also be optimal. This is the Bellman optimality principle that underpins Dijkstra's correctness. |
| 6 | **Path Validity** | The returned path must start at the source, end at the target, and every consecutive pair of nodes along the path must be connected by an edge in the original graph. |
| 7 | **Path–Length Consistency** | The length reported by `nx.dijkstra_path_length` must equal the manually computed sum of edge weights along the path returned by `nx.dijkstra_path`. This ensures the path-finding and distance-calculation logic are synchronized. |
| 8 | **Edge Existence on Paths** | Every edge traversed by every shortest path returned by `nx.single_source_dijkstra_path` must exist in the original graph, ensuring the algorithm does not invent non-existent transitions. |

---

## How to Run

### Prerequisites

```bash
pip install networkx hypothesis pytest
```

### Running the Tests

```bash
# Run all tests in verbose mode
pytest -v prim_djikstra_property_test.py
pytest -v test_shortest_path_properties.py
```

---

## Project Structure

```
DSGA_project_26113_26240/
├── README.md                          # This file
├── LICENSE                            # Project license
├── prim_djikstra_property_test.py     # Property tests for Prim's MST & Dijkstra's shortest path
└── test_shortest_path_properties.py   # Additional property tests for Dijkstra's shortest path
```

---

## License

See [LICENSE](LICENSE) for details.
