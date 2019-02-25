import numpy as np
import networkx as nx
import pandas as pd

def neighborhood_features(g, data):
    # Select training edges
    training_edges = [(l[0], l[1]) for l in data]

    # Compute link prediction related variables
    resource_alloc = [l[2] for l in list(nx.resource_allocation_index(g, ebunch=training_edges))]
    jaccard = [l[2] for l in list(nx.jaccard_coefficient(g, ebunch=training_edges))]
    adamic_adar = [l[2] for l in list(nx.adamic_adar_index(g, ebunch=training_edges))]
    preferential = [l[2] for l in list(nx.preferential_attachment(g, ebunch=training_edges))]
    common_neighbors = map(lambda x: common_neighbors_custom(g, x), training_edges)

    # Stack features
    graph_features = np.vstack([resource_alloc, jaccard, adamic_adar, preferential, common_neighbors]).T

    return(pd.DataFrame(graph_features, columns = ['ressource_allocation', 'jaccard', 'adamic_adar', 'preferential', 'common_neighbors']))

def common_neighbors_custom(g, edge):
    return(len(list(nx.common_neighbors(g, edge[0], edge[1]))))
