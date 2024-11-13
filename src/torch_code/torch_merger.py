import torch_in_memory_loader as torch_loader
import os
import sys

if __name__ == "__main__":
    loader = torch_loader.MCFDataset(os.path.join(sys.argv[1], "train"))
    loader = torch_loader.MCFDatasetInMemory(os.path.join(sys.argv[1], "train"))
    loader2 = torch_loader.MCFDataset(os.path.join(sys.argv[1], "test"))
    loader2 = torch_loader.MCFDatasetInMemory(os.path.join(sys.argv[1], "test"))
