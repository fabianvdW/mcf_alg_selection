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
import sys


def generate(m, n, supply, cost, cap, seed):
    random.seed(seed)
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

    # write the header
    print("p min {} {}".format(m * n + 2, len(edges)))
    print("n 1 {}".format(supply))
    print("n {} {}".format(m * n + 2, -supply))
    # now write all edges
    for edge in edges:
        print("a {} {} 0 {} {}".format(edge[0], edge[1], random.randint(1, cap), random.randint(1, cost)))


if __name__ == "__main__":
    generate(*map(lambda x: int(x), sys.argv[1:7]))
