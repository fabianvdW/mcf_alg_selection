import ast
import os.path as osp
import subprocess
from collections import defaultdict
from typing import Any, Dict
import sys
import torch

import numpy as np

from torch_geometric.data import Dataset, Data


class MCFDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self.id_to_runtime = {}
        self.idx_to_id = []
        with open(osp.join(self.root, "runtimes.csv"), "r") as in_runtimes:
            for line in in_runtimes:
                id, rest = line.split(" ", 1)
                if not "ERROR" in rest:
                    self.id_to_runtime[id] = [np.mean(runtimes_algo) for runtimes_algo in ast.literal_eval(rest)]
        with open(osp.join(self.root, "data_commands.csv"), "r") as in_commands:
            for line in in_commands:
                id, command = line.split(";")
                if id in self.id_to_runtime:
                    self.idx_to_id.append((id, ast.literal_eval(command)))
        self._indices = list(range(len(self.idx_to_id)))

    def len(self):
        return len(self._indices)

    def get(self, idx):
        print(idx, self._indices)
        idx = self._indices[idx]
        # Code copied mainly from https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx
        # and our readdimacs.py
        graph_id, command = self.idx_to_id[idx]
        instance_data = subprocess.run(
            command[0].replace("python", sys.executable), capture_output=True, text=True, shell=True, input=command[1]
        ).stdout
        lines = instance_data.split("\n")
        while lines[0].startswith("c"):
            del lines[0]
        assert lines[0].startswith("p ")
        _, mode, n, m = lines[0].split()
        num_nodes, num_edges = int(n), int(m)
        assert mode == "min"
        mapping = dict(zip(range(1, num_nodes + 1), range(num_nodes)))
        edge_index = torch.empty((2, num_edges), dtype=torch.long)
        data_dict: Dict[str, Any] = defaultdict(list)
        data_dict["edge_index"] = edge_index
        data_dict["demand"] = [0 for _ in range(num_nodes)]
        edge_num = 0
        for line in lines[1:]:
            if line.startswith("n "):
                _, node_id, demand = line.split()
                data_dict["demand"][mapping[int(node_id)]] = int(demand)
            elif line.startswith("a "):
                _, src, dst, low, cap, cost = line.split()
                assert int(low) == 0
                data_dict["edge_index"][0, edge_num] = mapping[int(src)]
                data_dict["edge_index"][1, edge_num] = mapping[int(dst)]
                data_dict["capacity"].append(int(cap))
                data_dict["weight"].append(int(cost))
                edge_num += 1
        data_dict["label"] = self.id_to_runtime[graph_id]
        data_dict["y"] = np.zeros(4)
        data_dict["y"][np.argmin(data_dict["label"])] = 1
        for key, value in data_dict.items():
            try:
                data_dict[key] = torch.as_tensor(value)
            except Exception as e:
                print(e)
        data = Data.from_dict(data_dict)
        data.num_nodes = num_nodes
        # Prepare node attributes
        data.x = data["demand"].view(-1, 1)
        del data["demand"]
        # Prepare edge attributes
        data.edge_attr = torch.cat([data["capacity"].view(-1, 1), data["weight"].view(-1, 1)], dim=-1)
        del data["capacity"]
        del data["weight"]
        return data
