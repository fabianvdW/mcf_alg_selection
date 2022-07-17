# -*- coding: utf-8 -*-
"""
Gridgraph generator with networkx
m = rows
n = columns
one source node s
one sink node t
Node(i,j) is connected to Node(i+1,j) and Node(i,j+1)
Source node s is connected to all nodes of first column
Sink node t is reachable from all nodes of the last column
"""
import random
import networkx as nx


def generate(m, n, supply, cost, cap, amount=1):
    for i in range(amount):
        y = str(i)
        G = nx.DiGraph()
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                G.add_node(j + (i - 1) * n, demand=0)

        G.add_node(0)  # s
        G.add_node(m * n + 1)  # t
        for k in range(1, m * n + 1):
            if (m * n - k) > n:
                G.add_edge(k, k + n)
            if (k % n) == 0:
                G.add_edge(k, m * n + 1)
            else:
                G.add_edge(k, k + 1)
            if (k % n) == 1:
                G.add_edge(0, k)
        print(G.edges())

        G.nodes[0]["demand"] = supply
        G.nodes[m * n + 1]["demand"] = -supply

        costs = {edge: random.randint(1, cost) for edge in G.edges}
        capacities = {edge: random.randint(1, cap) for edge in G.edges}

        nx.set_edge_attributes(G, costs, "cost")
        nx.set_edge_attributes(G, capacities, "capacity")

        dimacs_filename = y + "_GRIDGRAPH" + str(m) + "_" + str(n) + "_" + str(supply) + ".min"

        with open("../data/generated_data/" + dimacs_filename, "w") as f:
            # write the header
            f.write("p min {} {}\n".format(G.number_of_nodes(), G.number_of_edges()))
            f.write("n 0 {}\n".format(G.nodes[0]["demand"]))
            f.write("n {} {}\n".format(m * n + 1, G.nodes[m * n + 1]["demand"]))
            # now write all edges
            for u, v in G.edges():
                # print(G[u][v])
                f.write("a {} {} 0 {} {}\n".format(u, v, G[u][v]["capacity"], G[u][v]["cost"]))


generate(100, 100, 100, 100, 100)
