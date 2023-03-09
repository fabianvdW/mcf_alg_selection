import numpy as np
import scipy.stats


def debug_print(s, debug):
    if debug:
        print(s)


def is_significant_baseline(runtimes, sigma_upper=200000, significance_level=0.95, debug=False):
    runtimes = [np.array(x) for x in runtimes]
    algorithms, N = len(runtimes), np.array([len(x) for x in runtimes])
    debug_print(N, debug)
    means = np.array([x.mean() for x in runtimes])
    debug_print(means, debug)
    j_star = np.argmin(means)
    min_mean = means[j_star]
    # First approach -> Baseline feasible solution as given in PDF
    sol_val = significance_level ** (1 / algorithms)
    a = min_mean - means
    b = sigma_upper / np.sqrt(N)
    q_j_star = scipy.stats.norm.ppf(significance_level ** (1 / algorithms))
    for j in range(algorithms):
        if j == j_star:
            continue
        q_j = - (a[j] + b[j_star] * q_j_star) / b[j]
        sol_val *= scipy.stats.norm.cdf(q_j)
    debug_print(f"Significance level reached in first test: {sol_val}", debug)
    if sol_val >= significance_level:
        return None
    elif all([x >= 2 for x in N]):
        # Second approach -> Baseline feasible solution as given in PDF
        s = np.array([np.sqrt(np.var(x, ddof=1)) for x in runtimes])
        debug_print(s, debug)
        a = min_mean - means
        b = s / np.sqrt(N)
        q = [0 for _ in range(algorithms)]
        q[j_star] = scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[j_star] - 1) if b[
                                                                                                       j_star] > 0. else np.inf
        sol_val = scipy.stats.t.cdf(q[j_star], df=N[j_star] - 1)
        for j in range(algorithms):
            if j == j_star:
                continue
            if q[j_star] == np.inf:
                assert b[j_star] == 0.
                q[j] = - a[j] / b[j] if b[j] > 0. else np.inf
            else:
                q[j] = - (a[j] + b[j_star] * q[j_star]) / b[j] if b[j] > 0. else np.inf
            sol_val *= scipy.stats.t.cdf(q[j], df=N[j] - 1)
        debug_print(q, debug)
        debug_print(f"Significance level reached in second test: {sol_val}", debug)
        if sol_val >= significance_level:
            return None
        else:
            # Do estimation for new N according to baseline feasible solution given in PDF
            means[j_star] = np.inf
            j_prime = np.argmin(means)
            means[j_star] = min_mean
            b = s  # Definition of b changes, see pdf
            q = [scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[j] - 1) for j in range(algorithms)]
            if np.isclose(a[j_prime], 0):
                return 2 * N
            N_initial = ((q[j_star] * b[j_star] + q[j_prime] * b[j_prime]) / a[j_prime]) ** 2
            N_new = np.zeros(algorithms)
            N_new[j_star] = N_initial
            for j in range(algorithms):
                if j == j_star:
                    continue
                else:
                    N_new[j] =  ((q[j] * b[j]) / (a[j] + b[j_star] * q[j_star] / np.sqrt(N_initial))) ** 2
            N_new = np.clip(N_new, N, 8 * N)
            N_new = [int(x) for x in np.ceil(N_new)]
            debug_print(f"N target value: {means @ np.array(N_new)}", debug)
            return N_new

    return [2 for _ in range(algorithms)]


def is_significant_naive(runtimes, sigma_upper=200000, significance_level=0.95, debug=False):
    runtimes = [np.array(x) for x in runtimes]
    algorithms, N = len(runtimes), [len(x) for x in runtimes]
    debug_print(N, debug)
    means = np.array([x.mean() for x in runtimes])
    debug_print(means, debug)
    min_algo = np.argmin(means)
    min_mean = means[min_algo]
    # First approach
    first_test = True
    q = scipy.stats.norm.ppf(significance_level ** (1 / algorithms))
    upper_min_algo = min_mean + sigma_upper * q / np.sqrt(N[min_algo])
    for i in range(algorithms):
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
        debug_print(s, debug)
        q_hat_upper = scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[min_algo] - 1)
        upper_min_algo = min_mean + s[min_algo] * q_hat_upper / np.sqrt(N[min_algo])
        debug_print(upper_min_algo, debug)
        N_new = N.copy()
        N_new[min_algo] *= 2
        second_test = True
        for i in range(algorithms):
            if i == min_algo:
                continue
            q_hat_lower = scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[i] - 1)
            lower_i = means[i] - s[i] * q_hat_lower / np.sqrt(N[i])
            debug_print(lower_i, debug)
            if lower_i < upper_min_algo:
                second_test = False
                N_new[i] *= 2
        if second_test:
            return None
        else:
            return N_new
    return [2 for _ in range(algorithms)]
