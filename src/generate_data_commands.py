# -*- coding: utf-8 -*-
"""
Create MCF-instances with different params with
Netgen, Gridgen, Goto and Gridgraph, saves the commands to generate the data
and automatically calculates running times of algorithms and features. Saves running times and features,
does not save the data (only implicitly saves command to generate data).
"""
import math
import time
import random
from src.util import *


def get_seed():
    return int(time.time() * 100 * 100 * random.random())


def generate_gridgraph():
    costs = [2, 10, 100, 1000, 10000]
    supply = [1, 10, 100, 1000]
    cap = [1, 10, 100, 1000, 10000]
    width = [5, 10, 20, 30, 50, 70, 200]
    i = 0
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
                            command_dict[
                                f"GRIDGRAPH_{i}"] = f"python generate_gridgraph.py {m} {n} {int(s)} {cost} {c} {get_seed()}"
                            i += 1


def generate_netgen():

    min_cost, max_cost = 1, 10 ** 4
    min_cap, max_cap = 1, 10 ** 3
    min_nodes, max_nodes = [10, 10 ** 3]
    seed = get_seed()
    nodes = random.randint(min_nodes, max_nodes)
    arcs = random.randint(nodes * 2, nodes * (nodes - 1))
    supply = random.randint(1, 100 * nodes)
    a, b, c = random.random(), random.random(), random.random()
    supply_nodes = min(1, math.floor(nodes * a / (a + b + c)))
    demand_nodes = min(1, math.floor(nodes * b / (a + b + c)))

    params = [seed, 1, nodes, supply_nodes, demand_nodes, arcs, min_cost, max_cost, supply, 0, 0, 100, 100,
              min_cap, max_cap]
    params_str = ' '.join(map(str, params))
    path_to_netgen = os.path.join(PATH_TO_PROJECT, "netgen")
    command = f'"{path_to_netgen}{os.path.sep}netgen"'
    return command, params_str

if __name__ == "__main__":
    import subprocess
    command, input = generate_netgen()
    print(input)
    instance_data = subprocess.run(command, capture_output=True, input=input.encode("utf-8")).stdout
    print(instance_data.decode("utf-8"))