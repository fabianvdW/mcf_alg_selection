# -*- coding: utf-8 -*-
"""
Gridgraph generator without networkx
m = rows
n = columns
one source node s
one sink node t
Node(i,j) is connected to Node(i+1,j) and Node(i,j+1)
Source node s is connected to all nodes of first column
Sink node t is reachable from all nodes of the last column
"""

import random

random.seed(42)


# To Do: der name der datei nur vom seed abhängig
def generate(m, n, supply, cost, cap, param_list, amount=1):
    for i in range(amount):
        seed = 3 * 10 ** 7 + random.randint(10 ** 6, 10 ** 7 - 1)
        dimacs_filename = str(i) + "_GRIDGRAPH-" + str(seed) + ".min"

        edges = []
        for k in range(2, m * n + 2):
            if (m * n - (k - 1)) >= n:
                edges.append((k, k + n))
            if ((k - 1) % n) == 0:
                edges.append((k, m * n + 2))
            else:
                edges.append((k, k + 1))
            if ((k - 1) % n) == 1:
                edges.append((1, k))

        # print(edges)
        # print([seed,m,n,supply,cost,cap])

        params_str = ' '.join(map(str, [seed, m, n, supply, cost, cap]))
        param_list.append(params_str + "\n")

        with open("../data/generated_data/" + dimacs_filename, "w") as f:
            # write the header
            f.write("p min {} {}\n".format(m * n + 2, len(edges)))
            f.write("n 1 {}\n".format(supply))
            f.write("n {} {}\n".format(m * n + 2, -supply))
            # now write all edges
            for edge in edges:
                # print(G[u][v])
                f.write("a {} {} 0 {} {}\n".format(edge[0], edge[1], random.randint(1, cap), random.randint(1, cost)))

# generate(10,20,40,100,100,[])
