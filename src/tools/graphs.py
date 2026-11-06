import networkx as nx

def is_eulerian_digraph(G):
    if not nx.is_strongly_connected(G):
        return False
    for node in G.nodes():
        if G.in_degree(node) != G.out_degree(node):
            return False
    return True