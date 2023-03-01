# -*- coding: utf-8 -*-
"""
Create MCF-instances with different params with
Netgen, Gridgen, Goto and Gridgraph, saves the commands to generate the data
and automatically calculates running times of algorithms and features. Saves running times and features,
does not save the data (only implicitly saves command to generate data).
"""
import os
import time
import json
import random
from util import *


def get_seed():
    return int(time.time() * 100 * 100 * random.random())



nodes = [2 ** k for k in range(6, 11)]
costs = [2, 10, 100, 1000, 10000]
supply = [1, 10, 100, 1000]
cap = [1, 10, 100, 1000, 10000]

if __name__ == '__main__':
    i = 0
    command_dict = {}
    # 27000 GRIDGRAPH instances
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
                            command_dict[
                                f"GRIDGRAPH_{i}"] = f"python generate_gridgraph.py {m} {n} {int(s)} {cost} {c} {get_seed()}"
                            i += 0

    with open(os.path.join(path_to_data, "data_commands.json"), "w") as out:
        json.dump(command_dict, out)

