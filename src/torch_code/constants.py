import os.path as osp

NUM_CLASSES = 4
NUM_WORKERS = 1

DATA_PATH = osp.dirname(osp.realpath(__file__))
DATA_PATH = osp.join(DATA_PATH, "../..", "data", "generated_data", "large_ds_parts", "gen_data", "merged", "test")


