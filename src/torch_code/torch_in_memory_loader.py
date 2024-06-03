import ast
import os.path as osp
import torch

import numpy as np

from torch_geometric.data import InMemoryDataset


class MCFDataset(InMemoryDataset):
    def __init__(self, root):
        self.id_to_runtime = {}
        self.id_to_cmd = {}
        self.idx_to_id = []
        with open(osp.join(root, "runtimes.csv"), "r") as in_runtimes:
            for line in in_runtimes:
                id, rest = line.split(" ", 1)
                if not "ERROR" in rest:
                    self.id_to_runtime[id] = [np.mean(runtimes_algo) for runtimes_algo in ast.literal_eval(rest)]
        with open(osp.join(root, "data_commands.csv"), "r") as in_commands:
            for line in in_commands:
                id, command = line.split(";")
                if id in self.id_to_runtime:
                    self.idx_to_id.append(id)
                    self.id_to_cmd[id] = ast.literal_eval(command)
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def process(self) -> None:
        data_list = []
        for graph_id in self.idx_to_id:
            data_list.append(torch.load(osp.join(self.processed_dir, graph_id + ".pt")))
        self.save(data_list, self.processed_paths[0])
