# -*- coding: utf-8 -*-
"""
Create MCF-instances with different params with
Netgen, Gridgen, Goto and Gridgraph
"""


path_to_project = '../'
path_to_data = path_to_project + 'data/generated_data/'

nodes = [2 ** k for k in range(6, 11)]
costs = [2, 10, 100, 1000, 10000]
supply = [1, 10, 100, 1000]
cap = [1, 10, 100, 1000, 10000]

if __name__ == '__main__':

    param_list = []

    # 18000 instances ~ 1.6GB
    for n in nodes:
        for cost in costs:
            for s in [s * (n ** 0.5) for s in supply]:
                for c in cap:
                    for d in [8 * n, n * (n ** 0.25), n * (n ** 0.5)]:
                        for sd in [1, n ** 0.25, n ** 0.5]:
                            net.create(4, n, cost, int(s), c, int(d), int(sd), param_list, path_to_data)
    print("Netgen Done")

    # 18000 instances ~1.5GB
    for n in nodes:
        for cost in costs:
            for s in [s * (n ** 0.5) for s in supply]:
                for c in cap:
                    for d in [8, (n ** 0.25), (n ** 0.5)]:
                        for sd in [1, n ** 0.25, n ** 0.5]:
                            for w in [n ** 0.5, (n / 2) ** 0.5]:
                                for twa in [0, 1]:
                                    ggen.create(n, cost, int(s), c, int(d), int(sd), int(w), twa, param_list,
                                                path_to_data)
    print("Gridgen Done")

    # 27000 instances, x feasible ~ 1.6GB
    width = [5, 10, 20, 30, 50, 70, 100]
    for m in width:
        for n in width:
            for s in [s * ((m * n) ** 0.5) for s in supply]:
                for c in cap:
                    create = True
                    if s > m * c:
                        create = False
                    if c == 1000 and s > 23000:
                        create = False
                    if c == 100 and s > 2500:
                        create = False
                    if c == 10 and s > 350:
                        create = False
                    if c == 1 and s > 25:
                        create = False
                    if create:
                        for cost in costs:
                            ggraph.generate(m, n, int(s), cost, c, param_list, 10)

    print("Gridgraph Done")

    cap = cap[1:]
    costs = costs[1:]
    # 18000 instances ~6GB
    for n in nodes:
        for cost in costs:
            for c in cap:
                for d in [8 * n, n * (n ** 0.5)]:
                    goto.create(112, int(n), cost, c, int(d), param_list, path_to_data)

    with open(path_to_data + "params.txt", "w") as f:
        f.writelines(param_list)

    print("GO-TO Done")
