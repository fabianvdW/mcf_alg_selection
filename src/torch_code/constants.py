import os.path as osp
import torch

NUM_CLASSES = 4
NUM_WORKERS = 1

DATA_PATH = osp.dirname(osp.realpath(__file__))
DATA_PATH = osp.join(DATA_PATH, "../..", "data", "generated_data")


def transform_fn(data):
    data.edge_index = data.edge_index.type(torch.int64)
    return data
