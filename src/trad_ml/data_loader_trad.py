import ast
import os.path as osp
import pandas as pd
import numpy as np
from io import StringIO
from util import FEATURE_NAMES, ALGO_NAMES

DATA_PATH = osp.dirname(osp.realpath(__file__))
DATA_PATH = osp.join(DATA_PATH, "../..", "data", "generated_data", "large_ds_parts", "gen_data", "merged")


def get_data():
    # Read runtimes to get an overview of all valid id's and runtimes
    id_to_runtime = {}
    with open(osp.join(DATA_PATH, "runtimes.csv"), "r") as in_runtimes:
        for line in in_runtimes:
            id, rest = line.split(" ", 1)
            if not "ERROR" in rest:
                id_to_runtime[id] = [np.mean(runtimes_algo) for runtimes_algo in ast.literal_eval(rest)]
    # Filter out all lines containing Features, the rest can be parsed as csv
    with open(osp.join(DATA_PATH, "features.csv"), "r") as in_features:
        features_str = "\n".join(line for line in in_features if not "Features" in line)

    features_data = pd.read_csv(StringIO(features_str), sep=" ", header=None, index_col=0, names=FEATURE_NAMES)
    runtimes_data = pd.DataFrame.from_dict(id_to_runtime, orient="index", columns=ALGO_NAMES)

    # combine data according to ID, and clean it up
    combined_data = features_data.join(runtimes_data)
    combined_data["Minimum"] = combined_data.loc[:, ALGO_NAMES].idxmin(axis=1)

    return combined_data


if __name__ == "__main__":
    data = get_data()
