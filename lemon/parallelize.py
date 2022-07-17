import subprocess

import queue
from os import listdir
from os.path import join
from multiprocessing.dummy import Pool as ThreadPool

TIME_LIMIT = 300 * 10 ** 6  # 5min


def run_file(file):
    times = []
    costs = []
    invalid_or_error = None
    for algo in range(7):
        res = subprocess.run(["bash", "-c", f"python call_lemon.py {file} {algo}"], capture_output=True).stdout.decode(
            "utf-8")
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
        return f"{file} {invalid_or_error}\n"
    else:
        return f"{file} {times[0]} {times[1]} {times[2]} {times[3]} {times[4]} {times[5]} {times[6]}\n"


if __name__ == "__main__":
    # Read existing results
    existing_results = []
    with open("existingresults.txt", "r") as infile:
        for line in infile:
            existing_results.append(line.split(" ")[0].split("/")[-1])
    tasks = []
    data_path = "./../data/generated_data"
    for f in listdir(data_path):
        if str(f).endswith(".min") and not f in existing_results:
            tasks.append(join(data_path, f))

    items_to_evaluate = len(tasks)
    result_queue = queue.Queue()


    def run_file_async(file):
        global result_queue
        result_queue.put(run_file(file))


    pool = ThreadPool(8)
    pool.map_async(run_file_async, tasks)

    while items_to_evaluate > 0:
        with open("existingresults.txt", "a") as outfile:
            res = result_queue.get()
            outfile.write(res)
            items_to_evaluate -= 1
