import os
import argparse
import pickle

from constants import *

def setup_parser():
    out = argparse.ArgumentParser()
    out.add_argument("-dsroot", default=DATA_PATH, type=str, help="Root folder of ds")
    return out


def main(args):
    with open(os.path.join(args.dsroot, 'result', 'skopt_result.pkl'), "rb") as f:
        skopt_result = pickle.load(f)
    with open(os.path.join(args.dsroot, 'result', 'log_info_result.pkl'), "rb") as f:
        log_info_result = pickle.load(f)

    print(skopt_result)


if __name__ == "__main__":
    os.chdir("..")
    parser = setup_parser()
    _args = parser.parse_args()
    main(_args)
