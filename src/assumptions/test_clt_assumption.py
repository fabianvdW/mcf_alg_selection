import numpy as np
from scipy.stats import shapiro
import load_data
from matplotlib import pyplot as plt

if __name__ == '__main__':
    runtime_list = load_data.load_independence_data()

    # Shape: Realization, Algo, Index
    print(runtime_list.shape)
    # hoeffding independence test for the following pairwise combinations
    N_CLT = [30]
    results = {}
    for j in range(4):
        # Compute mean and std for all 30 * 100 instances
        mean = runtime_list[:, j, :].mean()
        std = runtime_list[:, j, :].std()
        samples = []
        for i in range(len(runtime_list)):
            normalized = np.sqrt(len(runtime_list[i, j, :])) * (runtime_list[i, j, :].mean() - mean) / std
            samples.append(normalized)
        print(samples)
        print(j, shapiro(samples)[1])
