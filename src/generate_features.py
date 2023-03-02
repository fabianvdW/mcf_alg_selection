# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 08:46:33 2021

@author: Philipp
"""
import sys
import readdimacs as rd
import numpy as np
import networkx as nx
import line_profiler


# To node depth mst
# features_of_graph = [number of nodes, number of arcs, min cost, max cost,
#                      sum cost, mean cost, std cost, min cap, max cap, sum cap, 
#                      mean cap, std cap, max of min of shortest path from supply to demand,
#                      supply, num of supply nodes, number of arcs between source and sink]
# features of mst = [sum cost, mean cost,std cost, sum cap, mean cap, std cap]

def calculate_features(g, mst):
    features = []
    if not mst:
        features.append(g.number_of_nodes())
        features.append(g.size())

    weights = [g[u][v]["weight"] for u, v in g.edges]
    if not mst:
        features.append(min(weights))
        features.append(max(weights))
    features.append(sum(weights))
    features.append(np.mean(weights) / max(weights))
    features.append(np.std(weights) / max(weights))

    capacities = [g[u][v]["capacity"] for u, v in g.edges]
    if not mst:
        features.append(min(capacities))
        features.append(max(capacities))
    features.append(sum(capacities))
    features.append(np.mean(capacities) / max(capacities))
    features.append(np.std(capacities) / max(capacities))

    if not mst:
        demand = [g.nodes[v]["demand"] for v in g.nodes]
        demand_nodes = [v for v in g.nodes if g.nodes[v]["demand"] < 0]
        supply_nodes = [v for v in g.nodes if g.nodes[v]["demand"] > 0]
        supply = sum([abs(d) for d in demand]) / 2

        # min_shortest_path = []
        # for s in supply_nodes:
        #     demand_nodes_length = []
        #     for d in demand_nodes:
        #         try:
        #             demand_nodes_length.append(nx.shortest_path_length(g, s, d, weight="weight"))
        #         except:
        #             pass
        #     min_shortest_path.append(min(demand_nodes_length))

        # features.append(max(min_shortest_path))
        features.append(supply)
        features.append(len(supply_nodes))

        source = supply_nodes[0]
        sink = demand_nodes[0]
        arcs_source_sink = nx.shortest_path_length(g, source, sink)
        features.append(arcs_source_sink)

    return features


if __name__ == "__main__":
    graph = rd.read(sys.stdin)
    features_a = calculate_features(graph, False)
    mst = nx.minimum_spanning_tree(graph.to_undirected())
    features_b = calculate_features(mst, True)
    print(" ".join(map(lambda x: str(x), features_a + features_b)))
