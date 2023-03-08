import numpy as np
import scipy.stats


def is_significant(runtimes, sigma_upper=25, significance_level=0.95):
    arr = np.array(runtimes)
    algorithms, N = arr.shape
    means = arr.mean(axis=1)
    print(means)
    min_algo = np.argmin(means)
    min_mean = means[min_algo]
    # First approach
    first_test = True
    q = scipy.stats.norm.ppf(significance_level ** (1 / algorithms))
    upper_min_algo = min_mean + sigma_upper * q / np.sqrt(N)
    for i in np.arange(algorithms):
        if i == min_algo:
            continue
        lower_i = means[i] - sigma_upper * q / np.sqrt(N)
        if lower_i < upper_min_algo:
            first_test = False
            break
    if first_test:
        return True
    elif N >= 2:
        # Second approach
        s = np.sqrt(np.var(arr, axis=1, ddof=1))
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
        return True
    return False

import pandas as pd
import os
from util import *
if __name__ == "__main__":
    runtimes_file = os.path.join(path_to_data,"runtimes.csv")
    runtimes = pd.read_csv(runtimes_file, header=None, sep=" ")
    runtimes.drop(columns=runtimes.columns[0], axis=1, inplace=True)
    runtimes = runtimes.to_numpy().transpose()
    print(is_significant(runtimes[:4], sigma_upper=10000))
    print(np.sqrt(np.var(runtimes[3][runtimes[3] > 50], axis=0, ddof=1)))
    print(runtimes[3][runtimes[3] > 50])