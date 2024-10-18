import torch_in_memory_loader as torch_loader
import os
import sys

if __name__ == "__main__":
    loader2 = torch_loader.MCFDatasetInMemory(os.path.join(sys.argv[1], "test"))
