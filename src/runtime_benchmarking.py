import numpy as np
import scipy.stats


def is_significant(runtimes, sigma_upper=200000, significance_level=0.95):
    runtimes = [np.array(x) for x in runtimes]
    algorithms, N = len(runtimes), [len(x) for x in runtimes]
    means = [x.mean() for x in runtimes]
    print(means)
    min_algo = np.argmin(means)
    min_mean = means[min_algo]
    # First approach
    first_test = True
    q = scipy.stats.norm.ppf(significance_level ** (1 / algorithms))
    upper_min_algo = min_mean + sigma_upper * q / np.sqrt(N[min_algo])
    for i in np.arange(algorithms):
        if i == min_algo:
            continue
        lower_i = means[i] - sigma_upper * q / np.sqrt(N[i])
        if lower_i < upper_min_algo:
            first_test = False
            break
    if first_test:
        return None
    elif all([x >= 2 for x in N]):
        # Second approach
        s = [np.sqrt(np.var(x, axis=0, ddof=1)) for x in runtimes]
        print(s)
        q_hat = scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N - 1)
        upper_min_algo = min_mean + s[min_algo] * q_hat / np.sqrt(N)
        print(upper_min_algo)
        for i in np.arange(algorithms):
            if i == min_algo:
                continue
            lower_i = means[i] - s[i] * q_hat / np.sqrt(N)
            print(lower_i)
            if lower_i < upper_min_algo:
                return False
        return None
    return [2 for _ in range(algorithms)]


import pandas as pd
import os
from util import *

if __name__ == "__main__":
    runtimes_file = os.path.join(path_to_data, "runtimes.csv")
    runtimes = pd.read_csv(runtimes_file, header=None, sep=" ")
    runtimes.drop(columns=runtimes.columns[0], axis=1, inplace=True)
    runtimes = runtimes.to_numpy().transpose()
    print(is_significant(runtimes[:], sigma_upper=2000000000000000000000000))
