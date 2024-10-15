import subprocess
import ast
import os.path as osp
import torch
import numpy as np
from collections import defaultdict
from typing import Any
import sys
from torch_geometric.data import InMemoryDataset, Dataset, Data


def transform_fn(data):
    data.edge_index = data.edge_index.type(torch.int64)
    return data


class MCFDatasetInMemory(InMemoryDataset):
    def __init__(self, root, transform=transform_fn):
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
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def process(self) -> None:
        data_list = []
        for graph_id in self.idx_to_id:
            data_list.append(torch.load(osp.join(self.processed_dir, graph_id + ".pt")))
        self.save(data_list, self.processed_paths[0])


class MCFDataset(Dataset):
    def __init__(self, root, transform=transform_fn):
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
        super().__init__(root, transform)
        self._indices = list(range(len(self.idx_to_id)))

    @property
    def processed_file_names(self) -> list[str]:
        return [graph_id + ".pt" for graph_id in self.idx_to_id]

    def process(self) -> None:
        for graph_id in self.idx_to_id:
            # Code copied mainly from https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx
            # and our readdimacs.py
            command = self.id_to_cmd[graph_id]
            instance_data = subprocess.run(
                command[0].replace("python", sys.executable).replace("generate_gridgraph", "../gen_data/generate_gridgraph"), capture_output=True, text=True, shell=True,
                input=command[1]
            ).stdout
            lines = instance_data.split("\n")
            while lines[0].startswith("c"):
                del lines[0]
            assert lines[0].startswith("p ")
            _, mode, n, m = lines[0].split()
            num_nodes, num_edges = int(n), int(m)
            assert mode == "min"
            mapping = dict(zip(range(1, num_nodes + 1), range(num_nodes)))
            edge_index = torch.empty((2, num_edges), dtype=torch.short)
            data_dict: dict[str, Any] = defaultdict(list)
            data_dict["edge_index"] = edge_index
            data_dict["demand"] = [0.0 for _ in range(num_nodes)]
            edge_num = 0
            for line in lines[1:]:
                if line.startswith("n "):
                    _, node_id, demand = line.split()
                    data_dict["demand"][mapping[int(node_id)]] = float(demand)
                elif line.startswith("a "):
                    _, src, dst, low, cap, cost = line.split()
                    assert int(low) == 0
                    data_dict["edge_index"][0, edge_num] = mapping[int(src)]
                    data_dict["edge_index"][1, edge_num] = mapping[int(dst)]
                    data_dict["capacity"].append(float(cap))
                    data_dict["weight"].append(float(cost))
                    edge_num += 1
            data_dict["label"] = self.id_to_runtime[graph_id]
            data_dict["y"] = np.argmin(data_dict["label"])
            for key, value in data_dict.items():
                try:
                    data_dict[key] = torch.as_tensor(value)
                except Exception as e:
                    print(e)
            data = Data.from_dict(data_dict)
            max_weight = torch.max(data["weight"])
            max_supply = torch.max(torch.abs(data["demand"]))
            data.num_nodes = num_nodes
            # Prepare node attributes
            data.label = data["label"].view(1, -1)
            data.x = data["demand"].view(-1, 1) / max_supply
            del data["demand"]
            # Prepare edge attributes
            data.edge_attr = torch.cat(
                [data["capacity"].view(-1, 1) / max_supply, data["weight"].view(-1, 1) / max_weight], dim=-1)
            del data["capacity"]
            del data["weight"]
            torch.save(data, osp.join(self.processed_dir, graph_id + ".pt"))

    def len(self):
        return len(self._indices)

    def get(self, idx):
        assert idx in self._indices
        graph_id = self.idx_to_id[idx]
        return torch.load(osp.join(self.processed_dir, graph_id + ".pt"))
