import numpy as np
import scipy.stats


def is_significant(runtimes, sigma_upper=200000, significance_level=0.95):
    runtimes = [np.array(x) for x in runtimes]
    algorithms, N = len(runtimes), [len(x) for x in runtimes]
    means = np.array([x.mean() for x in runtimes])
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
        s = [np.sqrt(np.var(x, ddof=1)) for x in runtimes]
        q_hat_upper = scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[min_algo] - 1)
        upper_min_algo = min_mean + s[min_algo] * q_hat_upper / np.sqrt(N[min_algo])
        N_new = N.copy()
        N_new[min_algo] *= 2
        second_test = True
        for i in np.arange(algorithms):
            if i == min_algo:
                continue
            q_hat_lower = scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[i] - 1)
            lower_i = means[i] - s[i] * q_hat_lower / np.sqrt(N[i])
            if lower_i < upper_min_algo:
                second_test = False
                N_new[i] *= 2
        if second_test:
            return None
        else:
            return N_new
    return [2 for _ in range(algorithms)]


import pandas as pd
import os
from util import *

if __name__ == "__main__":
    runtimes_file = os.path.join(path_to_data, "runtimes.csv")
    runtimes = pd.read_csv(runtimes_file, header=None, sep=" ")
    runtimes.drop(columns=runtimes.columns[0], axis=1, inplace=True)
    runtimes = runtimes.to_numpy().transpose()
    print(is_significant(runtimes[:, :], sigma_upper=2000000000000000000000000))
