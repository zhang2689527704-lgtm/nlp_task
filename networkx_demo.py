import networkx as nx

G = nx.cycle_graph(5)
centrality = nx.eigenvector_centrality_numpy(G)
print(centrality)
#{0: 0.44721359549995787, 1: 0.44721359549995787, 2: 0.447213595499958, 3: 0.4472135954999581, 4: 0.447213595499958}

# import networkx as nx

# G = nx.path_graph(4)

# centrality = nx.eigenvector_centrality_numpy(G)

# print("Eigenvector centrality:", centrality)
