import subprocess
import queue
import os
from multiprocessing.dummy import Pool as ThreadPool
import json
import sys
from util import *
from call_algorithm import call_algorithm


# TODO:
# -Change the runtime determination such that it can be repeated n-times until a stochastic test yields that
# it has not to be repeated anymore (A first "stochastic" test could just say: Repeat 2 times)
# -Change the runtime determination such that it runs before the features are generated and if an instance
# proves to be infeasible do not determine the features anymore
# -Change the main loop such that it incorporates the data generation:
# For each of the data sources, generate random commands until a fixed amount (Say N=25,000) of feasible! instances
# were evaluated
# In a first iteration, just assume that different data sources are given via a black box function source() which returns a single
# data generation command.

def source():
    return "python generate_gridgraph.py 5 5 5 2 1 5920924449881"


def stochastic_test(runtimes):
    return len(runtimes) >= 2


def run_task(task):
    id, data_command, run_features, run_runtimes = task
    res = [id, None, None]
    instance_data = subprocess.run(data_command, capture_output=True, shell=True).stdout
    if run_features:
        features_proc = subprocess.run("python generate_features.py", capture_output=True, input=instance_data,
                                       shell=True)
        res[1] = features_proc.stdout.decode("utf-8").rstrip()
    if run_runtimes:
        times = []
        costs = []
        invalid_or_error = None
        for algo in range(7):
            timed_out, result = call_algorithm(algo, instance_data)
            try:
                if timed_out:
                    time = TIME_LIMIT
                    cost = -1
                else:
                    time, cost = result.split(" ")
                    time, cost = int(time), int(cost)
                times.append(time)
                costs.append(cost)
            except:
                print("invalid")
                invalid_or_error = "ERROR: " + " ".join(result.strip().split(" ")[0:])
                break
        if not invalid_or_error and any(c != costs[0] and time != TIME_LIMIT for (c, time) in zip(costs, times)):
            invalid_or_error = "The algorithms do not agree on one cost."

        if invalid_or_error:
            res[2] = invalid_or_error
        else:
            res[2] = f"{times[0]} {times[1]} {times[2]} {times[3]} {times[4]} {times[5]} {times[6]}"
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
    key = "GRIDGRAPH_2724"
    tasks = [(key, data_commands[key], False, True) for _ in range(1000)]
    """
    for key in data_commands.keys():
        in_features = key in existing_features
        in_runtimes = key in existing_runtimes
        if not in_features or not in_runtimes:
            tasks.append((key, data_commands[key], not in_features, not in_runtimes))
    """
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
