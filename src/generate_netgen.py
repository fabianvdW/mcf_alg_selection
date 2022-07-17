"""
generates NETGEN Instances of the following families:
    8, SR, LO-8, LO-SR, DEG

INPUT: "Family", log2(n), amount

In the studies, n is 2^k with k bewteen 8 and 16 (except at DEG)
capacities: uniformly between 1 and 1000
costs: uniformly between 1 and 10000
supply nodes = n^1/2
demand nodes = n^1/2
total supply = 1000*(n^1/2)

NETGEN-8: m = 8n
NETGEN-SR: m = n*(n^1/2)
NETGEN-LO-8: total supply = 10*(n^1/2) and m = 8n
NETGEN-LO-SR: total supply = 10*(n^1/2) and m = n*(n^1/2)
NETGEN-DEG: n 4096 and m = 2^k with k between 13 and 24
#define PROBLEM_PARMS 13		/* aliases for generation parameters */
#define NODES	    parms[0]		/* number of nodes */
#define SOURCES     parms[1]		/* number of sources (including transshipment) */
#define SINKS	    parms[2]		/* number of sinks (including transshipment) */
#define DENSITY     parms[3]		/* number of (requested) arcs */
#define MINCOST     parms[4]		/* minimum cost of arcs */
#define MAXCOST     parms[5]		/* maximum cost of arcs */
#define SUPPLY	    parms[6]		/* total supply */
#define TSOURCES    parms[7]		/* transshipment sources */
#define TSINKS	    parms[8]		/* transshipment sinks */
#define HICOST	    parms[9]		/* percent of skeleton arcs given maximum cost */
#define CAPACITATED parms[10]		/* percent of arcs to be capacitated */
#define MINCAP	    parms[11]		/* minimum capacity for capacitated arcs */
#define MAXCAP	    parms[12]		/* maximum capacity for capacitated arcs */
"""

import random

random.seed(42)
import subprocess
import sys

# if "cygwin64\bin" not in str(os.getcwd()):
#     os.chdir(r"..\..\..\..\bin")
#     print(str(os.getcwd()))
path_to_project = '../'
path_to_netgen = 'cd ' + path_to_project + '/netgen;echo "'


def generate(params, amount, family, path_to_netgen, path_to_data, param_list=[]):
    for i in range(amount):
        params[0] = 10 ** 7 + random.randint(10 ** 6, 10 ** 7 - 1)
        params_str = ' '.join(map(str, params))
        if family == "DEG":
            params[5] = 2 ** (i + 13)
            params_str = ' '.join(map(str, params))
        filename = str(i) + "_NETGEN-" + str(params[0]) + ".min"
        command = path_to_netgen + params_str + '" |./netgen.exe >' + path_to_data + filename + '; exit;'
        param_list.append(params_str + "\n")

        cmd = ["bash", "-c", command]
        subprocess.call(cmd)


def main(family, k, amount):
    """generates "amount" NETGEN Instances of family "family" with 2^k nodes"""
    problem_num = k
    n = 2 ** k
    supply_nodes = demand_nodes = int(n ** (0.5))
    m = 0
    min_cost, max_cost = 1, 10000
    total_supply = int(1000 * (n ** 0.5))
    min_cap, max_cap = 1, 1000
    params = [0, problem_num, n, supply_nodes, demand_nodes, m, min_cost, max_cost, total_supply, 0, 0, 100, 100,
              min_cap, max_cap]
    path_to_data = path_to_project + '/data/lemon_data/netgen/generated/'

    if family == "8":
        params[5] = 8 * n
        generate(params, amount, family, path_to_netgen, path_to_data)

    elif family == "SR":
        params[5] = int(n * (n ** 0.5))
        generate(params, amount, family, path_to_netgen, path_to_data)

    elif family == "LO-8":
        params[8] = int(10 * (n ** 0.5))
        params[5] = 8 * n
        generate(params, amount, family, path_to_netgen, path_to_data)

    elif family == "LO-SR":
        params[8] = int(10 * (n ** 0.5))
        params[5] = int(n * (n ** 0.5))
        generate(params, amount, family, path_to_netgen, path_to_data)

    elif family == "DEG":
        n = 4096
        params[2] = n
        params[3] = params[4] = int(n ** (0.5))
        params[8] = int(1000 * (n ** 0.5))
        amount = 11

        generate(params, amount, family, path_to_netgen, path_to_data)


def create(amount, nodes, costs, supply, cap, density, sdnodes, param_list, path_to_data):
    params = [0, 8, nodes, sdnodes, sdnodes, density, 1, costs, supply, 0, 0, 100, 100, 1, cap]

    # print(params)
    generate(params, amount, "None", path_to_netgen, path_to_data, param_list)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
