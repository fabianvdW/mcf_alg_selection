import subprocess
import queue
import os
from multiprocessing.dummy import Pool as ThreadPool
import json
import sys
import numpy as np
from util import *
from call_algorithm import call_algorithm
from stochastics import is_significant_baseline


# TODO:
# -Change the main loop such that it incorporates the data generation:
# For each of the data sources, generate random commands until a fixed amount (Say N=25,000) of feasible! instances
# were evaluated
# In a first iteration, just assume that different data sources are given via a black box function source() which returns a single
# data generation command.

def source():
    return "python generate_gridgraph.py 5 5 5 2 1 5920924449881"


def run_task(task):
    id, data_command, run_features, run_runtimes = task
    res = [id, None, None]
    instance_data = subprocess.run(data_command, capture_output=True).stdout
    if run_runtimes:
        costs = []
        invalid_or_error = None
        N = [1 for _ in range(4)]
        while invalid_or_error is None and N is not None:
            runtimes = [[] for _ in range(4)]
            # Setup the task list in terms of indices
            task_list = [j for j in range(4) for _ in range(N[j])]
            task_list = np.random.permutation(task_list)
            for algo in task_list:
                timed_out, result = call_algorithm(algo, instance_data)
                try:
                    if timed_out:
                        time = TIME_LIMIT
                    else:
                        time, cost = result.split(" ")
                        time, cost = int(time), int(cost)
                        costs.append(cost)
                    runtimes[algo].append(time)
                except:
                    invalid_or_error = "ERROR: " + " ".join(result.strip().split(" ")[0:])
                    print(f"Task with id {id} has {invalid_or_error}")
                    break
            if not invalid_or_error:
                N = is_significant_baseline(runtimes)
            if N is None and invalid_or_error is None:
                print(f"Task with id {id}: Finished as runtimes {runtimes} proved significant.")
                means = np.array([np.array(x).mean() for x in runtimes])
                res[2] = f"{means[0]} {means[1]} {means[2]} {means[3]}"
            elif invalid_or_error is None:
                if sum(N) >= 10000:
                    invalid_or_error = f"ERROR: Too many runs (N={N}) were requested, dropping instance"
                    print(f"Task with id {id}: {invalid_or_error}")
                else:
                    print(f"Task with id {id}: Retrying with N={N} as runtimes {runtimes} proved insignificant.")
        if invalid_or_error:
            res[1] = "Features not determined as instance proved to be not feasible"
            run_features = False
            res[2] = invalid_or_error
        elif any(c != costs[0] for c in costs):
            res[1] = "Features not determined as algorithms do not agree on one cost"
            run_features = False
            res[2] = "The algorithms do not agree on one cost."

    if run_features:
        features_proc = subprocess.run("python generate_features.py", capture_output=True, input=instance_data)
        res[1] = features_proc.stdout.decode("utf-8").rstrip()
    return res


if __name__ == "__main__":
    # Read existing results for features
    existing_features, existing_runtimes = [], []
    features_file, runtimes_file = os.path.join(path_to_data, "features.csv"), os.path.join(path_to_data,
                                                                                            "runtimes.csv")

    with open(features_file, "r") as in_features, open(runtimes_file, "r") as in_runtimes:
        for line in in_features:
            existing_features.append(line.split(" ")[0])
        for line in in_runtimes:
            existing_runtimes.append(line.split(" ")[0])
    with open(os.path.join(path_to_data, "data_commands.json"), "r") as infile:
        data_commands = json.load(infile)

    tasks = []
    for key in data_commands.keys():
        in_features = key in existing_features
        in_runtimes = key in existing_runtimes
        if not in_features or not in_runtimes:
            tasks.append((key, data_commands[key], not in_features, not in_runtimes))
    items_to_evaluate = len(tasks)
    result_queue = queue.Queue()


    def run_task_async(task):
        global result_queue
        result_queue.put(run_task(task))


    pool = ThreadPool(int(sys.argv[1]))
    pool.map_async(run_task_async, tasks)

    while items_to_evaluate > 0:
        with open(features_file, "a") as out_features, open(runtimes_file, "a") as out_runtimes:
            id, res_features, res_runtimes = result_queue.get()
            if res_features:
                out_features.write(f"{id} {res_features}\n")
            if res_runtimes:
                out_runtimes.write(f"{id} {res_runtimes}\n")
            print(f"Finished task with id {id}")
            items_to_evaluate -= 1
