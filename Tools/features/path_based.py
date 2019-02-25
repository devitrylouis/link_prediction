import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse.linalg import inv

def path_features(g, data, IDs, name = 'shortest_path', compute_katz = False):
    # Select training edges
    training_edges = [(l[0], l[1]) for l in data]

    # Allocate list
    if compute_katz: katz_scores = []

    # Compute link prediction related variables
    shortest_path = map(lambda x: shortest_path_custom(g, x), training_edges)

    # Katz beta
    if compute_katz:
        katz_matrix = katz_measure(beta = 0.5)

        for sample in training_edges:
            index_source = IDs.index(sample[0])
            index_target = IDs.index(sample[1])

            katz_scores.append(katz_matrix[index_source, index_target])

        return(pd.DataFrame(np.vstack(shortest_path, katz_scores).T, columns = [name, 'katz_score']))
    else:
        return(pd.DataFrame(shortest_path, columns = [name]))

def shortest_path_custom(g, edge):
    try:

        return(1 - len(nx.shortest_path(g, source=edge[0], target=edge[1])))
    except:
        return(1)

def betweenness_centrality(g):
    betweenness = nx.edge_betweenness_centrality(g, normalized=True, weight=None)
    be = [betweenness[edge] for edge in training_edges]
    return(be)

def katz_measure(beta = 0.5):
    A = nx.adjacency(g)
    katz_matrix = inv(np.identity(A.shape[0]) - beta*A) - np.identity(A.shape[0])

    return katz_matrix
