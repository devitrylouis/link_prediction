import numpy as np
import networkx as nx
import pandas as pd

def construct_digraph(data, node_info):

    # Create empty directed graph
    g = nx.DiGraph(directed=True)

    # Add nodes and edges
    nodes = [element[0] for element in node_info]
    for i, n in enumerate(nodes):
        g.add_node(n)

    edges = [(element[0],element[1]) for element in data if element[2]=="1"]
    for i, e in enumerate(edges):
        g.add_edge(e[0], e[1], capacity = 1)

    return(g)

def construct_graph(data, node_info):

    # Create empty directed graph
    g = nx.Graph(directed=True)

    # Add nodes and edges
    nodes = [element[0] for element in node_info]
    for i, n in enumerate(nodes):
        g.add_node(n)

    edges = [(element[0],element[1]) for element in data if element[2]=="1"]
    for i, e in enumerate(edges):
        g.add_edge(e[0], e[1])

    return(g)
