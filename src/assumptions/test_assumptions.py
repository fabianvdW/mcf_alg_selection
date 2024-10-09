import os

PATH_TO_PROJECT = os.path.join("../..")
PATH_TO_DATA = os.path.join(PATH_TO_PROJECT, "data", "generated_data")

import os.path as osp
import ast

if __name__ == '__main__':
    runtime_list = []
    with open(osp.join(PATH_TO_DATA, "assumptions", "runtimes.csv"), "r") as in_runtimes:
        for line in in_runtimes:
            id, rest = line.split(" ", 1)
            if not "ERROR" in rest:
                runtime_list.append([runtimes_algo for runtimes_algo in ast.literal_eval(rest)])
