import networkx as nx
from hypothesis import given, strategies as st, settings
import random

# Configure Hypothesis for more complex graph generation
settings.register_profile("ci", deadline=1000, max_examples=50)
settings.load_profile("ci")

@st.composite
def weighted_graph_and_nodes(draw, directed=False):
    """
    Hypothesis strategy to generate a weighted graph and two distinct nodes.
    
    Graph Generation Details:
    1. Generates a random number of nodes (2 to 30).
    2. Starts with a random tree to guarantee connectivity between all nodes.
    3. Adds a random number of additional edges to create cycles and multiple path options.
    4. Assigns positive floating-point weights to all edges (Dijkstra requires non-negative weights).
    5. Returns the graph along with two randomly selected distinct nodes (source and target).
    """
    num_nodes = draw(st.integers(min_value=2, max_value=30))
    
    # Create initial connected structure
    base_G = nx.random_tree(num_nodes)
    if directed:
        G = nx.DiGraph(base_G)
    else:
        G = nx.Graph(base_G)

    # Add complexity with extra edges
    num_extra_edges = draw(st.integers(min_value=0, max_value=num_nodes))
    nodes = list(G.nodes())
    for _ in range(num_extra_edges):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    # Assign random positive weights
    for u, v in G.edges():
        G.edges[u, v]['weight'] = draw(st.floats(min_value=0.1, max_value=50.0))

    source, target = random.sample(nodes, 2)
    return G, source, target

@given(data=weighted_graph_and_nodes(directed=False))
def test_shortest_path_validity(data):
    """
    Property: Path Validity (Postcondition)
    
    Description: The path returned must be a valid simple path in G starting 
    at the source and ending at the target.
    
    Importance: This is the most basic requirement of the algorithm. If the 
    returned list of nodes doesn't correspond to existing edges in the graph, 
    the algorithm is fundamentally broken.
    """
    G, u, v = data
    path = nx.dijkstra_path(G, source=u, target=v, weight='weight')
    
    assert path[0] == u, "Path must start at source"
    assert path[-1] == v, "Path must end at target"
    
    # Verify every step in the path is a valid edge in the graph
    for i in range(len(path) - 1):
        assert G.has_edge(path[i], path[i+1]), f"Edge {path[i]}->{path[i+1]} does not exist"

@given(data=weighted_graph_and_nodes(directed=False))
def test_path_length_consistency(data):
    """
    Property: Path Length Consistency (Postcondition)
    
    Description: The length returned by nx.dijkstra_path_length must equal 
    the sum of the weights of the edges in the path returned by nx.dijkstra_path.
    
    Importance: Ensures that the path-finding logic and the distance-calculation 
    logic are synchronized and using the same weight definitions.
    """
    G, u, v = data
    path = nx.dijkstra_path(G, source=u, target=v, weight='weight')
    calculated_dist = nx.dijkstra_path_length(G, source=u, target=v, weight='weight')
    
    # Sum weights manually
    manual_dist = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
    
    assert manual_dist == calculated_dist, "Sum of edge weights must equal reported path length"

@given(data=weighted_graph_and_nodes(directed=False))
def test_shortest_path_symmetry(data):
    """
    Property: Symmetry (Invariant for Undirected Graphs)
    
    Description: In an undirected graph, the distance from u to v must be 
    identical to the distance from v to u.
    
    Importance: Shortest path distances in undirected graphs form a metric space. 
    Symmetry is a requirement of any metric. A failure here suggests 
    the algorithm might be sensitive to node ordering or traversal direction.
    """
    G, u, v = data
    dist_uv = nx.dijkstra_path_length(G, source=u, target=v, weight='weight')
    dist_vu = nx.dijkstra_path_length(G, source=v, target=u, weight='weight')
    
    assert dist_uv == dist_vu, "Distance must be symmetric in undirected graphs"

@given(data=weighted_graph_and_nodes(directed=False))
def test_path_to_self_boundary(data):
    """
    Property: Path to Self (Boundary Condition)
    
    Description: The shortest path from any node u to itself is always [u], 
    and its length is 0.
    
    Importance: Tests the edge case where source equals target. Robust 
    algorithms must handle zero-length traversals without errors or 
    incorrect cycle detection.
    """
    G, u, _ = data
    path = nx.dijkstra_path(G, source=u, target=u, weight='weight')
    length = nx.dijkstra_path_length(G, source=u, target=u, weight='weight')
    
    assert path == [u], "Path to self must be just the node itself"
    assert length == 0, "Distance to self must be zero"

@given(data=weighted_graph_and_nodes(directed=False))
def test_subpath_optimality(data):
    """
    Property: Subpath Optimality (Invariant)
    
    Description: Any subpath of a shortest path is itself a shortest path. 
    If the shortest path from u to v is [u, ..., i, j, ..., v], then the 
    distance from i to j found by the algorithm must match the weight 
    sum of the segment [i, ..., j] in the original path.
    
    Importance: This is the "Optimal Substructure" property required for 
    dynamic programming and greedy algorithms like Dijkstra to work. 
    If this fails, the algorithm cannot guarantee a global optimum.
    """
    G, u, v = data
    full_path = nx.dijkstra_path(G, source=u, target=v, weight='weight')
    
    if len(full_path) < 3:
        return # Not enough nodes to test a proper subpath
        
    # Pick two nodes from the path as sub-source and sub-target
    idx1 = random.randint(0, len(full_path) - 2)
    idx2 = random.randint(idx1 + 1, len(full_path) - 1)
    
    sub_u, sub_v = full_path[idx1], full_path[idx2]
    
    # The distance calculated independently for the subpath
    independent_sub_dist = nx.dijkstra_path_length(G, source=sub_u, target=sub_v, weight='weight')
    
    # The distance as it appears in the original full path
    original_sub_dist = sum(G[full_path[i]][full_path[i+1]]['weight'] for i in range(idx1, idx2))
    
    assert independent_sub_dist == original_sub_dist, "Subpath of a shortest path must be optimal"