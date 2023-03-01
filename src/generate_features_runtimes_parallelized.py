import subprocess

import queue
import os
from multiprocessing.dummy import Pool as ThreadPool
import json
import sys
from util import *

TIME_LIMIT = 300 * 10 ** 6  # 5min


def run_task(task):
    id, run_features, run_runtimes = task
    res = [id, None, None]
    instance_data = ...  # TODO
    if run_features:
        subprocess.run(["bash", "-c", f"python generate_feature.py {instance_data}"],
                       capture_output=True).stdout.decode(
            "utf-8")
        # TODO ...
        res[1] = ...
    if run_runtimes:
        times = []
        costs = []
        invalid_or_error = None
        for algo in range(7):
            res = subprocess.run(["bash", "-c", f"python call_lemon.py {instance_data} {algo}"],
                                 capture_output=True).stdout.decode(
                "utf-8")
            # Write instance_data to child
            try:
                _, time, cost = res.split(" ")
                time, cost = int(time), int(cost)
                times.append(time)
                costs.append(cost)
            except:
                invalid_or_error = " ".join(res.strip().split(" ")[1:])
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
    for key in data_commands.keys():
        in_features = key in existing_features
        in_runtimes = key in existing_runtimes
        if not in_features or not in_runtimes:
            tasks.append((key, data_commands[key], in_features, in_runtimes))

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
                out_features.write(res_features)
            if res_runtimes:
                out_runtimes.write(res_runtimes)
            items_to_evaluate -= 1
