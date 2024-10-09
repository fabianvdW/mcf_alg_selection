import numpy as np
from scipy.stats import shapiro
import load_data
from matplotlib import pyplot as plt

if __name__ == '__main__':
    runtime_list = load_data.load_data()

    # Shape: Realization, Algo, Index
    print(runtime_list.shape)
    # hoeffding independence test for the following pairwise combinations
    N_CLT = [30]
    results = {}
    for n_clt in N_CLT:
        for instance in range(len(runtime_list)):
            for j in range(4):
                if runtime_list[instance, j, :].std() == 0.:
                    continue
                samples = []
                for i in range(100-n_clt):
                    data_points = runtime_list[instance, j, i:i+n_clt]
                    normalized = np.sqrt(n_clt) * (data_points.mean() - runtime_list[instance, j, :].mean()) / runtime_list[
                                                                                                               instance, j,
                                                                                                               :].std()
                    samples.append(normalized)
                print(n_clt, shapiro(samples)[1])
