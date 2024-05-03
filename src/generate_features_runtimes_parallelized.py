import ast
import subprocess
import queue
import sys
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
import numpy as np
from util import *
from call_algorithm import call_algorithm
from stochastics import is_significant
from generate_data_commands import generate_netgen, generate_gridgraph, generate_goto, generate_gridgen


def run_task(task):
    id, data_command, run_features, run_runtimes = task
    res = [id, data_command, None, None]
    instance_data = subprocess.run(data_command[0].replace("python", sys.executable), capture_output=True, text=True, shell=True, input=data_command[1]).stdout

    if run_runtimes:
        costs = []
        N = [1 for _ in range(NUM_ALGORITHMS)]
        while N is not None:
            runtimes = [[] for _ in range(NUM_ALGORITHMS)]
            # Setup the task list in terms of indices
            task_list = [j for j in range(NUM_ALGORITHMS) for _ in range(N[j])]
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
                    invalid_or_error = "ERROR: Parsing not possible " + " ".join(result.strip().split(" ")[0:])
                    res[2] = f"Features not determined due to {invalid_or_error}"
                    res[3] = invalid_or_error
                    print(f"Task with id {id} has {invalid_or_error}")
                    return res
            if any(c != costs[0] for c in costs):
                res[2] = "Features not determined as algorithms do not agree on one cost"
                res[3] = "ERROR: The algorithms do not agree on one cost."
                print(f"Task with id {id} has {res[3]}")
                return res
            if all(r == 0 for algo in runtimes for r in algo):
                invalid_or_error = "ERROR: Skipping instance, runtimes are 0 across the board"
                res[2] = f"Features not determined due to {invalid_or_error}"
                res[3] = invalid_or_error
                print(f"Task with id {id} has {invalid_or_error}")
                return res
            if all(r == TIME_LIMIT for algo in runtimes for r in algo):
                invalid_or_error = "ERROR: Skipping instance, algorithms timed out across the board"
                res[2] = f"Features not determined due to {invalid_or_error}"
                res[3] = invalid_or_error
                print(f"Task with id {id} has {invalid_or_error}")
                return res
            N = is_significant(runtimes)
            if N is None:
                print(f"Task with id {id}: Finished as runtimes {runtimes} proved significant.")
                res[3] = f"{runtimes}"
            elif sum(N) >= MAX_SAMPLES:
                invalid_or_error = f"ERROR: Too many runs (N={N}) were requested, dropping instance"
                res[2] = f"Features not determined due to {invalid_or_error}"
                res[3] = invalid_or_error
                print(f"Task with id {id}: {invalid_or_error}")
                return res
            else:
                print(f"Task with id {id}: Retrying with N={N} as runtimes {runtimes} proved insignificant.")

    if run_features:
        features_proc = subprocess.run("python generate_features.py", text=True, shell=True, capture_output=True, input=instance_data)
        res[2] = features_proc.stdout.rstrip()
    return res


if __name__ == "__main__":
    # Read existing results for features
    existing_features, runtimes_evaluated, runtimes_error = [], [], []
    features_f = os.path.join(PATH_TO_DATA, "features.csv")
    runtimes_f = os.path.join(PATH_TO_DATA, "runtimes.csv")
    commands_f = os.path.join(PATH_TO_DATA, "data_commands.csv")
    mutex = Lock()
    actual_instances = [0, 0, 0, 0]
    finished_instances = [0, 0, 0, 0]


    def add_instance_by_id(id):
        global actual_instances, mutex
        for i in range(NUM_GENERATORS):
            if GENERATOR_NAMES[i] in id:
                with mutex:
                    actual_instances[i] += 1

    def finish_instance_by_id(id):
        global mutex, finished_instances
        for i in range(NUM_GENERATORS):
            if GENERATOR_NAMES[i] in id:
                with mutex:
                    finished_instances[i] += 1

    data_commands = {}
    with open(features_f, "r") as in_features, open(runtimes_f, "r") as in_runtimes, open(commands_f,
                                                                                          "r") as in_commands:
        for line in in_features:
            existing_features.append(line.split(" ")[0])
        for line in in_runtimes:
            id = line.split(" ")[0]
            runtimes_evaluated.append(id)
            add_instance_by_id(id)
            if "ERROR" in line:
                runtimes_error.append(id)
            else:
                finish_instance_by_id(id)
        for line in in_commands:
            id, command = line.split(";")
            data_commands[id] = ast.literal_eval(command)
    # Prepare remaining unfinished tasks
    tasks = queue.Queue()
    for key in data_commands.keys():
        in_features = key in existing_features
        in_runtimes = key in runtimes_evaluated
        in_runtimes_error = key in runtimes_error
        if not in_features and not in_runtimes_error or not in_runtimes:
            tasks.put((key, data_commands[key], not in_features and not in_runtimes_error, not in_runtimes))
    result_queue = queue.Queue()


    def is_finished():
        global tasks, finished_instances, mutex
        with mutex:
            return tasks.empty() and any(
                map(lambda i: finished_instances[i] >= TARGET_INSTANCES[i], range(NUM_GENERATORS)))

    def get_task():
        global tasks, actual_instances, mutex, finished_instances
        if not tasks.empty():
            return tasks.get()
        else:
            with mutex:
                unfinished_generators = np.arange(NUM_GENERATORS)[
                    list(map(lambda i: finished_instances[i] < TARGET_INSTANCES[i], range(NUM_GENERATORS)))]
                generator = np.random.choice(unfinished_generators)
                id = f"{GENERATOR_NAMES[generator]}_{actual_instances[generator]}"
                actual_instances[generator] += 1
            if generator == GRIDGRAPH:
                command = generate_gridgraph()
            elif generator == NETGEN:
                command = generate_netgen()
            elif generator == GOTO:
                command = generate_goto()
            elif generator == GRIDGEN:
                command = generate_gridgen()

            return (id, command, True, True)

    thread_count = 0
    def run_task_async(_):
        global result_queue, thread_count, mutex

        with mutex:
            thread_num = thread_count
            thread_count += 1
        print(f"Thread {thread_num} starting!")
        while not is_finished():
            try:
                print(f"Thread {thread_num} getting task!")
                task = get_task()
                print(f"Thread {thread_num} running task!")
                res = run_task(task)
                print(f"Thread {thread_num} finished task!")
                if not (task[3] and isinstance(res[3], str)):
                    finish_instance_by_id(task[0])
                result_queue.put(res)
            except Exception as e:
                print(f"Thread {thread_num} has error")
                print(e)
                assert False

    threads = int(sys.argv[1])
    pool = ThreadPool(threads)
    pool.map_async(run_task_async, range(threads))
    while not is_finished():
        with open(features_f, "a") as o_features, open(runtimes_f, "a") as o_runtimes, open(commands_f,
                                                                                            "a") as o_commands:
            id, command, res_features, res_runtimes = result_queue.get()
            if id not in data_commands:
                o_commands.write(f"{id};{command}\n")
            if res_features:
                o_features.write(f"{id} {res_features}\n")
            if res_runtimes:
                o_runtimes.write(f"{id} {res_runtimes}\n")
            print(f"Finished task with id {id}")
