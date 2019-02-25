import numpy as np
import pandas as pd
import networkx as nx

def node_degrees(g, data):

    out_nodes = [l[0] for l in data]
    in_nodes = [l[1] for l in data]

    in_degrees_in_nodes = g.in_degree(in_nodes)
    training_edges = [(l[0], l[1]) for l in data]
    result_inin = []
    for sample in training_edges:
        try:
            in_in = in_degrees_in_nodes[sample[1]]
            result_inin.append(in_in)
        except:
            result_inin.append(0)

    in_degrees_out_nodes = g.in_degree(out_nodes)
    training_edges = [(l[0], l[1]) for l in data]
    result_inout = []

    for sample in training_edges:
        try:
            in_out = in_degrees_out_nodes[sample[0]]
            result_inout.append(in_out)
        except:
            result_inout.append(0)

    out_degrees_in_nodes = g.out_degree(in_nodes)
    training_edges = [(l[0], l[1]) for l in data]
    result_outin = []
    for sample in training_edges:
        try:
            out_in = out_degrees_in_nodes[sample[1]]
            result_outin.append(out_in)
        except:
            result_outin.append(0)

    out_degrees_out_nodes = g.out_degree(out_nodes)
    training_edges = [(l[0], l[1]) for l in data]
    result_outout = []
    for sample in training_edges:
        try:
            out_out = out_degrees_out_nodes[sample[0]]
            result_outout.append(out_out)
        except:
            result_outout.append(0)
    output = np.vstack((result_inin, result_inout, result_outin, result_outout)).T

    return pd.DataFrame(data=output, index=None, columns=['in_in', 'in_out', 'out_in', 'out_out'], dtype=None, copy=False)
