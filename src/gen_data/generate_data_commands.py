# -*- coding: utf-8 -*-
"""
Create MCF-instances with different params with
Netgen, Gridgen, Goto and Gridgraph, saves the commands to generate the data
and automatically calculates running times of algorithms and features. Saves running times and features,
does not save the data (only implicitly saves command to generate data).
"""
import os
import time
import math
import random
from util import *

MIN_COST, MAX_COST = 1, 10**4
MIN_CAP, MAX_CAP = 1, 10**3
MIN_NODES, MAX_NODES = [15, 5 * 10**3]
MAX_ARCS = 200000


def get_seed():
    return int(time.time() * 100 * 100 * random.random())


def generate_goto():
    seed = get_seed()
    nodes = random.randint(MIN_NODES, MAX_NODES)
    max_arcs = min([MAX_ARCS, int(nodes ** (5 / 3))])
    arcs = random.randint(nodes * 6, max_arcs)
    path_to_goto = os.path.join(PATH_TO_PROJECT, "goto")
    command = f'"{path_to_goto}{os.path.sep}goto"'
    return command, f"{nodes} {arcs} {MAX_CAP} {MAX_COST} {seed}"


def generate_gridgen():
    seed = get_seed()
    nodes = random.randint(MIN_NODES, MAX_NODES)
    width = random.randint(5, int(math.sqrt(nodes)))
    supply = random.randint(1, 100 * nodes)
    a, b, c = random.random(), random.random(), random.random()
    supply_nodes = max(1, math.floor(nodes * a / (a + b + c)))
    demand_nodes = max(1, math.floor(nodes * b / (a + b + c)))
    arcs = random.randint(nodes * 2, min(MAX_ARCS, nodes * (nodes - 1)))
    average_degree = int(arcs / nodes)

    params = [1, seed, nodes, width, supply_nodes, demand_nodes, average_degree, supply, 1, MIN_COST, MAX_COST, 1, MIN_CAP, MAX_CAP]
    params_str = " ".join(map(str, params))
    path_to_gridgen = os.path.join(PATH_TO_PROJECT, "gridgen")
    command = f'"{path_to_gridgen}{os.path.sep}gridgen"'
    return command, params_str


def generate_gridgraph():
    seed = get_seed()
    nodes = random.randint(MIN_NODES, MAX_NODES)
    width = random.randint(5, int(math.sqrt(nodes)))
    height = int(nodes / width)
    supply = random.randint(1, 2 * MAX_CAP)

    return f"python generate_gridgraph.py {width} {height} {supply} {MAX_COST} {MAX_CAP} {seed}", ""


def generate_netgen():
    seed = get_seed()
    nodes = random.randint(MIN_NODES, MAX_NODES)
    arcs = random.randint(nodes * 2, min(MAX_ARCS, nodes * (nodes - 1)))
    supply = random.randint(1, 100 * nodes)
    supply_nodes, demand_nodes = 1, 1

    params = [seed, 1, nodes, supply_nodes, demand_nodes, arcs, MIN_COST, MAX_COST, supply, 0, 0, 100, 100, MIN_CAP, MAX_CAP]
    params_str = " ".join(map(str, params))
    path_to_netgen = os.path.join(PATH_TO_PROJECT, "netgen")
    command = f'"{path_to_netgen}{os.path.sep}netgen"'
    return command, params_str


if __name__ == "__main__":
    import subprocess
    from call_algorithm import call_algorithm

    command, input = ('../../gridgen/gridgen', '1 1392345719982 2291 28 307 951 2 113703 1 1 10000 1 1 1000')
    print(command, input)
    instance_data = subprocess.run(command, capture_output=True, text=True, input=input).stdout
    print("Finished generating instance")
    for algo in [0, 1, 2, 3]:
        print(f"Calling {algo}")
        import time

        start = time.time()
        timed_out, result = call_algorithm(algo, instance_data)
        print(timed_out, result)
        print(time.time() - start)
