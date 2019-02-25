import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path

def flow_features(g, data):
    result_max_flow = map(lambda x: max_flow(g, x), data)
    result_min_cut = map(lambda x: min_cut(g, x), data)
    return(pd.DataFrame(np.vstack((result_max_flow, result_min_cut)).T, columns = ['max_flow', 'min_cut']))

def min_cut(g, edge):
    return(nx.minimum_cut(g, edge[0], edge[1], flow_func=shortest_augmenting_path))

def max_flow(g, edge):
    return(nx.maximum_flow_value(g, edge[0], edge[1], flow_func=shortest_augmenting_path))
