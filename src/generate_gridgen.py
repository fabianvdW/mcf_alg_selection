"""
generates GRIDGEN Instances of the following families:
    8, SR, DEG

In the studies, n is 2^k with k bewteen 8 and 16 (except at DEG)
capacities: uniformly between 1 and 1000
costs: uniformly between 1 and 10000
supply nodes = n^1/2
demand nodes = n^1/2
total supply = 1000*(n^1/2)
width of grid = n^1/2

GRIDGEN-8: m = 8n
GRIDGEN-SR: m = n*(n^1/2)
GRIDGEN-DEG: n 4096 and m = 2^k with k between 13 and 24
"""

import random

random.seed(42)
import subprocess

import sys

path_to_project = '../'
path_to_gridgen = 'cd ' + path_to_project + '/gridgen;echo "'


def generate(params, amount, family, path_to_gridgen, path_to_data, param_list):
    for i in range(amount):
        params[1] = 2 * 10 ** 7 + random.randint(10 ** 6, 10 ** 7 - 1)
        params_str = ' '.join(map(str, params))
        if family == "DEG":
            params[6] = 2 ** (i + 2)
            params_str = ' '.join(map(str, params))
        filename = str(i) + "_GRIDGEN-" + str(params[1]) + ".min"
        command = path_to_gridgen + params_str + '" |./gridgen.exe >' + path_to_data + filename + '; exit;'
        cmd = ["bash", "-c", command]
        subprocess.call(cmd)
        param_list.append(params_str + "\n")


def main(family, k, amount):
    """generates "amount" GRIDGEN Instances of family "family" with 2^k nodes"""
    seed = 0
    n = 2 ** k
    width = int(n ** 0.5)
    supply_nodes = demand_nodes = int(n ** (0.5))
    deg = 0
    total_supply = int(1000 * (n ** 0.5))
    min_cost, max_cost = 1, 10000
    min_cap, max_cap = 1, 1000
    params = [1, seed, n, width, supply_nodes, demand_nodes, deg, total_supply, 1, min_cost, max_cost, 1, min_cap,
              max_cap]

    path_to_data = path_to_project + '/data/lemon_data/gridgen/generated/'

    if family == "8":
        params[6] = 8
        generate(params, amount, family, path_to_gridgen, path_to_data)

    elif family == "SR":
        params[6] = int(n ** 0.5)
        generate(params, amount, family, path_to_gridgen, path_to_data)

    elif family == "DEG":
        n = 4096
        params[2] = n
        params[3] = int(n ** 0.5)
        params[4] = params[5] = int(n ** (0.5))
        params[7] = int(1000 * (n ** 0.5))
        amount = 11
        generate(params, amount, family, path_to_gridgen, path_to_data)


def create(nodes, costs, supply, cap, density, sdnodes, width, twa, param_list, path_to_data):
    params = [twa, 0, nodes, width, sdnodes, sdnodes, density, supply, 1, 1, costs, 1, 1, cap]
    print(params)
    generate(params, 1, "None", path_to_gridgen, path_to_data, param_list)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
