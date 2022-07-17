"""
generates GO-TO Instances of the following families:
    8, SR

In the studies, n is 2^k with k bewteen 8 and 16
capacities: uniformly between 1 and 1000
costs: uniformly between 1 and 10000

GO-TO-8: m = 8n
GO-TO-SR: m = n*(n^1/2)
"""

import random

random.seed(42)

import subprocess
import sys

path_to_project = '../'
path_to_goto = 'cd ' + path_to_project + '/goto;echo "'


def generate(params, amount, path_to_goto, path_to_data, param_list):
    for i in range(amount):
        params[4] = 4 * 10 ** 7 + random.randint(10 ** 6, 10 ** 7 - 1)
        params_str = ' '.join(map(str, params))
        filename = str(i) + "_GOTO-" + str(params[4]) + ".min"
        command = path_to_goto + params_str + '" |./goto.exe >' + path_to_data + filename + '; exit;'
        cmd = ["bash", "-c", command]
        subprocess.call(cmd)
        param_list.append(params_str + "\n")


def main(family, k, amount):
    """generates "amount" GO-TO Instances of family "family" with 2^k nodes"""
    n = 2 ** k
    m = 0
    cost = 10000
    cap = 1000
    seed = 0
    params = [n, m, cost, cap, seed]

    path_to_data = path_to_project + '/data/lemon_data/goto/generated/'

    if family == "8":
        params[1] = 8 * params[0]
        generate(params, amount, family, path_to_goto, path_to_data)

    elif family == "SR":
        params[1] = int(params[0] * (params[0] ** 0.5))
        generate(params, amount, family, path_to_goto, path_to_data)


def create(amount, nodes, costs, cap, density, param_list, path_to_data):
    params = [nodes, density, costs, cap, 0]
    generate(params, amount, path_to_goto, path_to_data, param_list)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
