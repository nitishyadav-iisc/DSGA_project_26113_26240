"""
# =============================================================================
# NetworkX Property-Based Tests  —  Copilot Edition
# =============================================================================
# Team Members : [Your Name(s) Here]
# Algorithms   : Prim's Minimum Spanning Tree · Dijkstra's Shortest Path
# Framework    : Hypothesis (property-based testing) + pytest
# NetworkX ver : 3.2+
# =============================================================================
"""

import math

import networkx as nx
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# =============================================================================
# Graph Generation Strategies
# =============================================================================


@st.composite
def connected_weighted_undirected_graph(
    draw, min_nodes: int = 2, max_nodes: int = 15
):
    """
    Hypothesis composite strategy that generates random connected, undirected,
    positively-weighted graphs.

    Construction
    ────────────
    1. Draw a node count n in [min_nodes, max_nodes].
    2. Build a guaranteed spanning path by chaining nodes in a random
       permutation: perm[0]-perm[1]-…-perm[n-1].  This ensures connectivity
       regardless of n.
    3. For every remaining candidate edge independently flip a fair coin and,
       if heads, draw a positive weight and add the edge.

    Edge weights are drawn from (0.01, 100.0] to keep all weights strictly
    positive, avoiding degenerate tie situations that complicate comparison.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Step 2 – guarantee connectivity via a random-order spanning path
    perm = draw(st.permutations(range(n)))
    for i in range(n - 1):
        w = draw(
            st.floats(
                min_value=0.01, max_value=100.0,
                allow_nan=False, allow_infinity=False,
            )
        )
        G.add_edge(perm[i], perm[i + 1], weight=w)

    # Step 3 – randomly add extra edges
    for u in range(n):
        for v in range(u + 1, n):
            if not G.has_edge(u, v) and draw(st.booleans()):
                w = draw(
                    st.floats(
                        min_value=0.01, max_value=100.0,
                        allow_nan=False, allow_infinity=False,
                    )
                )
                G.add_edge(u, v, weight=w)

    return G


@st.composite
def connected_weighted_graph_with_node_pair(
    draw, min_nodes: int = 2, max_nodes: int = 15
):
    """
    Wraps connected_weighted_undirected_graph and additionally draws two nodes
    (possibly equal) from the graph, returning the triple (G, u, v).

    Tests that require distinct nodes use assume(u != v) internally.
    """
    G = draw(
        connected_weighted_undirected_graph(
            min_nodes=min_nodes, max_nodes=max_nodes
        )
    )
    nodes = list(G.nodes())
    u = draw(st.sampled_from(nodes))
    v = draw(st.sampled_from(nodes))
    return G, u, v


# =============================================================================
# Prim's Minimum Spanning Tree — Property-Based Tests
# =============================================================================


@given(G=connected_weighted_undirected_graph())
def test_prims_edge_count(G):
    """
    Property: The MST of a connected n-node graph has exactly n−1 edges.

    Mathematical basis
    ──────────────────
    A tree on n vertices has exactly n−1 edges.  This follows from the
    tree characterisation theorem: a connected acyclic graph on n vertices
    has exactly n−1 edges.  Fewer edges leaves at least one vertex
    unreachable; more edges introduces at least one cycle.

    Test strategy
    ─────────────
    Generate random connected weighted undirected graphs (2–15 nodes,
    sparse through dense) and verify that Prim's MST satisfies the
    edge-count invariant.  Adjacency density varies because extra edges
    are added probabilistically, so the test exercises both near-tree
    and near-complete graphs.

    Assumptions / preconditions
    ───────────────────────────
    Graph is connected (guaranteed by the strategy) and has at least
    2 nodes.

    Failure interpretation
    ──────────────────────
    • Edge count < n−1 → algorithm dropped a vertex (not spanning).
    • Edge count > n−1 → algorithm kept a redundant edge (not a tree).
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert mst.number_of_edges() == G.number_of_nodes() - 1


@given(G=connected_weighted_undirected_graph())
def test_prims_connectivity_and_spanning(G):
    """
    Property: The MST is connected and its vertex set equals the original
    vertex set.

    Mathematical basis
    ──────────────────
    A spanning tree of G must include every vertex (spanning) and be
    connected (tree).  Prim's algorithm initialises from an arbitrary root
    and iterates the greedy-cut step until every vertex is in the grown
    subtree.  For a connected input, this terminates with all n vertices
    included and every pair reachable through tree edges.

    Test strategy
    ─────────────
    Assert (1) nx.is_connected(mst) and (2) set(mst.nodes) == set(G.nodes).
    Both conditions are independently verifiable without knowledge of
    internal algorithm state.

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected weighted graph with at least 2 nodes.

    Failure interpretation
    ──────────────────────
    • Not connected → algorithm returned a forest, not a spanning tree.
    • Vertex sets differ → a vertex was silently dropped or spuriously added.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert nx.is_connected(mst)
    assert set(mst.nodes()) == set(G.nodes())


@given(G=connected_weighted_undirected_graph())
def test_prims_edges_are_subgraph(G):
    """
    Property: Every edge in the MST exists in the original graph
    (the MST is a subgraph of G).

    Mathematical basis
    ──────────────────
    Prim's algorithm selects edges exclusively from the input graph's
    edge set at each greedy step.  It cannot fabricate edges that are
    absent from G.  Therefore the MST must be a spanning subgraph of G:
    MST ⊆ G as a subgraph.

    Test strategy
    ─────────────
    Iterate over every (u, v) edge of the MST and assert the same edge
    exists in G.  For undirected graphs, G.has_edge(u, v) checks both
    directions.

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected weighted graph.

    Failure interpretation
    ──────────────────────
    A missing edge in G means the algorithm invented a connection that
    does not exist, violating the definition of a spanning subgraph and
    making the MST impossible to realise in the original network.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    for u, v in mst.edges():
        assert G.has_edge(u, v), (
            f"MST contains edge ({u},{v}) which does not exist in G"
        )


@given(G=connected_weighted_undirected_graph())
def test_prims_acyclicity(G):
    """
    Property: The MST contains no cycles.

    Mathematical basis
    ──────────────────
    Acyclicity is the second half of the tree definition.  Together with
    connectivity, it implies exactly n−1 edges (and vice-versa).  Prim's
    algorithm enforces acyclicity via the cut property: at each step it
    selects the minimum-weight edge crossing a cut between the visited
    subtree and the unvisited vertices.  Adding such an edge can never
    form a cycle, because one endpoint is already in the tree while the
    other is not yet connected to it.

    Test strategy
    ─────────────
    Compute the cycle basis of the MST using nx.cycle_basis and assert
    the result is an empty list (no cycles detected).

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected weighted graph.  Dense graphs (many potential
    cycles in G) stress the selection logic the most.

    Failure interpretation
    ──────────────────────
    A non-empty cycle basis means the algorithm kept a redundant edge
    that could be removed while preserving connectivity, so the result
    is not a minimum *tree*.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert nx.cycle_basis(mst) == []


@given(G=connected_weighted_undirected_graph())
def test_prims_weight_equals_kruskal(G):
    """
    Property (metamorphic): The total MST weight produced by Prim's
    algorithm equals the total weight produced by Kruskal's algorithm
    on the same graph.

    Mathematical basis
    ──────────────────
    Both Prim's and Kruskal's algorithms are provably optimal: each
    computes a minimum spanning tree of the input graph.  The total weight
    of any MST for a given graph is unique — this is the MST weight
    uniqueness theorem.  Although the specific edge sets can differ when
    edge weights have ties, the total weight must be the same for both
    solutions.  This cross-algorithm comparison avoids the need to
    enumerate all spanning trees (which is exponential in complexity).

    Test strategy
    ─────────────
    Compute both MSTs on the identical randomly generated graph, sum
    their edge weights, and compare with a relative tolerance of 1e-6
    to account for floating-point summation-order differences.

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected weighted graph with strictly positive weights.
    Positive weights make ties rare, strengthening the comparison.

    Failure interpretation
    ──────────────────────
    A weight discrepancy means at least one algorithm returned a
    suboptimal spanning tree rather than a genuine minimum spanning tree.
    """
    prim_mst    = nx.minimum_spanning_tree(G, algorithm="prim")
    kruskal_mst = nx.minimum_spanning_tree(G, algorithm="kruskal")

    prim_w    = sum(d["weight"] for _, _, d in prim_mst.edges(data=True))
    kruskal_w = sum(d["weight"] for _, _, d in kruskal_mst.edges(data=True))

    assert math.isclose(prim_w, kruskal_w, rel_tol=1e-6), (
        f"Prim weight {prim_w:.6f} ≠ Kruskal weight {kruskal_w:.6f}"
    )


# =============================================================================
# Dijkstra's Shortest Path — Property-Based Tests
# =============================================================================


@given(G=connected_weighted_undirected_graph())
def test_dijkstra_self_distance_is_zero(G):
    """
    Property: The shortest-path distance from any node to itself is
    exactly 0.

    Mathematical basis
    ──────────────────
    The empty path (staying at the source node with zero hops) has total
    weight 0.  With all edge weights strictly positive, no cycle can
    reduce this below zero.  Dijkstra's algorithm initialises dist[source]
    = 0 and only relaxes via positive-weight edges, so self-distance
    cannot become negative or non-zero.

    Test strategy
    ─────────────
    For every node in the generated graph call
    nx.dijkstra_path_length(G, node, node) and assert the result is 0.
    The test loops over all nodes so that every vertex in every generated
    graph is exercised.

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected graph with non-negative edge weights (weights
    are > 0 in our strategy, which is strictly stronger).

    Failure interpretation
    ──────────────────────
    A non-zero self-distance would indicate a corrupted distance
    initialisation or the algorithm traversing a negative-weight cycle
    that should not be reachable with non-negative weights.
    """
    for node in G.nodes():
        dist = nx.dijkstra_path_length(G, node, node, weight="weight")
        assert dist == 0, (
            f"Self-distance for node {node} is {dist}, expected 0"
        )


@given(data=connected_weighted_graph_with_node_pair())
def test_dijkstra_all_distances_non_negative(data):
    """
    Property: All single-source shortest-path distances are ≥ 0 when
    all edge weights are non-negative.

    Mathematical basis
    ──────────────────
    With non-negative edge weights the shortest path from source to any
    reachable node is a sum of non-negative terms, so the total is
    necessarily ≥ 0.  Dijkstra's algorithm preserves this invariant
    throughout: it never relaxes a node to a negative distance because
    the relaxation rule d[v] = d[u] + w(u,v) adds a non-negative weight
    to a non-negative current estimate.

    Test strategy
    ─────────────
    Use a random source node and compute the full single-source distance
    dictionary via nx.single_source_dijkstra_path_length.  Assert every
    entry in the dictionary is ≥ 0.

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected graph; all edge weights ≥ 0 (our strategy
    ensures weights > 0, which is stricter).

    Failure interpretation
    ──────────────────────
    A negative distance entry implies incorrect relaxation logic or an
    inadvertent negative-weight edge, both of which are fundamental
    correctness violations.
    """
    G, source, _ = data
    lengths = nx.single_source_dijkstra_path_length(G, source, weight="weight")
    for node, dist in lengths.items():
        assert dist >= 0, (
            f"Negative distance {dist:.6f} from source {source} to node {node}"
        )


@given(data=connected_weighted_graph_with_node_pair())
def test_dijkstra_symmetry_on_undirected_graph(data):
    """
    Property: On an undirected graph, dist(u, v) == dist(v, u).

    Mathematical basis
    ──────────────────
    In an undirected graph every edge (u, v, w) is traversable in both
    directions with the same cost w.  Therefore, if
    P = (u = x₀, x₁, …, xₖ = v) is an optimal path, the reversed path
    (v = xₖ, …, x₀ = u) has the same set of edges in the opposite order
    and thus identical total weight.  Symmetry is a necessary condition
    of any correct shortest-path algorithm on undirected graphs.

    Test strategy
    ─────────────
    Draw two distinct nodes u, v from a connected undirected weighted
    graph and assert dist(u,v) ≈ dist(v,u) with a tight relative
    tolerance to account for floating-point rounding.

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected graph; u ≠ v (enforced by assume).

    Failure interpretation
    ──────────────────────
    Asymmetry implies the implementation is treating the undirected graph
    as directed, or applying direction-sensitive edge weight logic that
    should not exist on an undirected input.
    """
    G, u, v = data
    assume(u != v)

    dist_uv = nx.dijkstra_path_length(G, u, v, weight="weight")
    dist_vu = nx.dijkstra_path_length(G, v, u, weight="weight")

    assert math.isclose(dist_uv, dist_vu, rel_tol=1e-9), (
        f"Symmetry violated: dist({u},{v}) = {dist_uv:.9f} "
        f"≠ dist({v},{u}) = {dist_vu:.9f}"
    )


@given(data=connected_weighted_graph_with_node_pair())
def test_dijkstra_triangle_inequality(data):
    """
    Property: For all nodes u, v, w in a connected graph,
    dist(u, v) ≤ dist(u, w) + dist(w, v)  (triangle inequality).

    Mathematical basis
    ──────────────────
    The triangle inequality is a direct consequence of path optimality.
    Proof by contradiction: suppose dist(u,v) > dist(u,w) + dist(w,v)
    for some intermediate node w.  Then concatenating the optimal path
    u→w with the optimal path w→v gives a valid path from u to v with
    total length dist(u,w) + dist(w,v) < dist(u,v), contradicting the
    optimality of dist(u,v).  Therefore the inequality must hold for any
    correct implementation of a shortest-path algorithm.

    Test strategy
    ─────────────
    For a pair of nodes (u, v) sampled from a randomly generated
    connected graph, iterate over every third node w and check the
    inequality.  A small absolute tolerance (1e-9) handles
    floating-point imprecision.

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected weighted graph with non-negative weights.

    Failure interpretation
    ──────────────────────
    A violation means the algorithm computed at least one incorrect
    shortest-path distance — a detour through w is strictly cheaper than
    the claimed optimal d(u,v), so the algorithm missed a shorter route.
    """
    G, u, v = data
    dist_uv = nx.dijkstra_path_length(G, u, v, weight="weight")

    for w in G.nodes():
        dist_uw = nx.dijkstra_path_length(G, u, w, weight="weight")
        dist_wv = nx.dijkstra_path_length(G, w, v, weight="weight")
        assert dist_uv <= dist_uw + dist_wv + 1e-9, (
            f"Triangle inequality violated: "
            f"dist({u},{v}) = {dist_uv:.9f}  >  "
            f"dist({u},{w}) + dist({w},{v}) = {dist_uw:.9f} + {dist_wv:.9f}"
        )


@given(data=connected_weighted_graph_with_node_pair())
def test_dijkstra_subpath_optimality(data):
    """
    Property (metamorphic): Every prefix of a shortest path from u to v
    is itself a shortest path from u to the prefix's terminal node
    (optimal substructure / Bellman principle).

    Mathematical basis
    ──────────────────
    Dijkstra's correctness rests on the Bellman optimality principle:
    if the path P[u→v] = (u = x₀, x₁, …, xₖ = v) is optimal, then for
    any index j < k the sub-path P[u→xⱼ] = (x₀, …, xⱼ) must also be
    optimal from u to xⱼ.

    Proof by contradiction: suppose there exists a cheaper path Q from u
    to xⱼ (i.e. cost(Q) < cost(P[u→xⱼ])).  Then the path Q followed by
    P[xⱼ→v] is a valid u-to-v path with total cost
      cost(Q) + cost(P[xⱼ→v])
      < cost(P[u→xⱼ]) + cost(P[xⱼ→v])
      = cost(P[u→v])
    contradicting the optimality of P[u→v].  ∎

    Test strategy
    ─────────────
    1. Compute the shortest path path = nx.dijkstra_path(G, u, v).
    2. For each prefix [u, …, m] (i.e. end_idx from 1 to len(path)−1):
       a. Sum the edge weights along that prefix.
       b. Compare with nx.dijkstra_path_length(G, u, m).
       c. Assert both values agree within a tight floating-point tolerance.

    Assumptions / preconditions
    ───────────────────────────
    Connected undirected graph; u ≠ v (enforced by assume) so that the
    path has at least one edge and there is something non-trivial to check.

    Failure interpretation
    ──────────────────────
    A discrepancy means the returned path is locally non-optimal at some
    intermediate step: a cheaper route to an intermediate node exists that
    the algorithm failed to find, which means the overall path is also
    suboptimal.
    """
    G, u, v = data
    assume(u != v)

    path = nx.dijkstra_path(G, u, v, weight="weight")

    for end_idx in range(1, len(path)):
        prefix = path[: end_idx + 1]
        prefix_weight = sum(
            G[prefix[i]][prefix[i + 1]]["weight"]
            for i in range(len(prefix) - 1)
        )
        optimal_dist = nx.dijkstra_path_length(
            G, u, prefix[-1], weight="weight"
        )
        assert math.isclose(
            prefix_weight, optimal_dist, rel_tol=1e-9, abs_tol=1e-9
        ), (
            f"Subpath optimality violated: "
            f"prefix u={u}→{prefix[-1]} via path has weight {prefix_weight:.9f} "
            f"but dijkstra reports optimal distance {optimal_dist:.9f}"
        )
