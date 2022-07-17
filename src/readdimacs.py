"""
reads dimacs and returns networkx graph
"""

import networkx as nx
import os
import math


def read(filename):
    with open(filename, mode="r") as f:
        for line in f.readlines():
            if line.startswith("p "):
                _, mode, n, m = line.split()
                assert mode == "min"
                g = nx.DiGraph(mode=mode)
                n = int(n)
                # if "GRIDGRAPH" in filename:
                #    k=0
                # else:
                k = 1
                nodes = list(range(k, n + k))
                g.add_nodes_from(nodes, demand=0)
            elif line.startswith("n "):
                _, ID, demand = line.split()
                g.nodes[int(ID)]["demand"] = int(demand)
            elif line.startswith("a "):
                _, src, dest, low, cap, cost = line.split()
                g.add_edge(int(src), int(dest), capacity=int(cap), weight=int(cost))
                assert int(low) == 0

    return g

# path_to_data = r"../data/generated_data"
# g = read(path_to_data+r"/0_GRIDGRAPH-31006323.min")
# for v in g.nodes:
#    g.nodes[v]["demand"] = -g.nodes[v]["demand"]

# x = nx.shortest_path_length(g,1,1002)
# print(x)
# pos = {}
# degree = [g.degree(v) for v in g.nodes]

# nx.draw(g, pos = pos, node_size=100)

# testing
# filename = r"..\data\lemon_data\gridgen\generated\0_GRIDGEN-8.min"

# g = read(filename)
# nx.draw(g)
