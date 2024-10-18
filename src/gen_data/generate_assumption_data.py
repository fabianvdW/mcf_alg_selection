import subprocess
from call_algorithm import call_algorithm
import time
from util import *
import numpy as np
import argparse


def setup_parser():
    out = argparse.ArgumentParser()
    out.add_argument('-n', default=1, type=int,
                     help='The number of workers used for training and evaluation.')
    out.add_argument("-dsroot", default=PATH_TO_DATA, type=str, help="Root folder of ds")
    out.add_argument("-cs2path", default=os.path.join(PATH_TO_PROJECT, "cs2"), type=str, help="Root folder of cs2")
    out.add_argument("-lemonpath", default=os.path.join(PATH_TO_PROJECT, "lemon"), type=str,
                     help="Root folder of lemon")
    return out


if __name__ == "__main__":
    parser = setup_parser()
    _args = parser.parse_args()
    command, input = ('"../../gridgen/gridgen"', '1 2294618482800 3542 15 1146 1770 17 95749 1 1 10000 1 1 1000')
    print(command, input)
    instance_data = subprocess.run(command, capture_output=True, text=True, shell=True, input=input).stdout
    instance_data = instance_data.replace("\n", "\r\n")
    print("Finished generating instance")
    N = [100 for _ in range(NUM_ALGORITHMS)]
    runtimes = [[] for _ in range(NUM_ALGORITHMS)]
    task_list = [j for j in range(NUM_ALGORITHMS) for _ in range(N[j])]
    task_list = np.random.permutation(task_list)
    for algo in task_list:
        timed_out, result = call_algorithm(algo, instance_data, cs2_path=_args.cs2path, lemon_path=_args.lemonpath)
        if timed_out:
            time = TIME_LIMIT
        else:
            time, _ = result.split(" ")
            time = int(time)
        runtimes[algo].append(time)
    runtimes_f = os.path.join(_args.dsroot, "runtimes.csv")
    with open(runtimes_f, "a") as o_runtimes:
        o_runtimes.write(f"{id} {runtimes}\n")
