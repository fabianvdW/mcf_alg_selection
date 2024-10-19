import os
import itertools

PATH_TO_PROJECT = os.path.join("../..")
PATH_TO_DATA = os.path.join(PATH_TO_PROJECT, "data", "generated_data", "assumptions")
import numpy as np
import os.path as osp
import ast


def load_data():
    runtime_list = []
    with open(osp.join(PATH_TO_DATA, "runtimes.csv"), "r") as in_runtimes:
        for line in in_runtimes:
            id, rest = line.split(" ", 1)
            if "ERROR" not in rest:
                runtime_list.append([runtimes_algo for runtimes_algo in ast.literal_eval(rest)])

    runtime_list = np.array(runtime_list)
    runtime_list = runtime_list[:, :, :]
    return runtime_list


def load_independence_data():
    runtime_list = []
    for i in range(30):
        with open(osp.join(PATH_TO_DATA, f"assumptionind{i}", "runtimes.csv"), "r") as in_runtimes:
            for line in in_runtimes:
                _, rest = line.split("id>", 1)
                runtime_list.append([runtimes_algo for runtimes_algo in ast.literal_eval(rest)])
    runtime_list = np.array(runtime_list)
    return runtime_list
