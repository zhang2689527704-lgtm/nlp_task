import networkx as nx

G = nx.cycle_graph(5)
centrality = nx.eigenvector_centrality_numpy(G)
print(centrality)


# import networkx as nx

# G = nx.path_graph(4)

# centrality = nx.eigenvector_centrality_numpy(G)

# print("Eigenvector centrality:", centrality)
