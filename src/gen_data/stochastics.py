import numpy as np
import scipy.stats
import mystic
from mystic.penalty import linear_inequality

ALGORITHMS = 4


def debug_print(s, debug):
    if debug:
        print(s)


def obj_n(x, means, process_creation_time=200000):
    return (means + process_creation_time) @ x[: len(means)]


def F_j(lambda_j, n_j, sigma_j):
    if n_j >= 30:
        if sigma_j == 0. and lambda_j >= 0.:
            return 1.
        if sigma_j == 0. and lambda_j < 0.:
            return 0.
        return scipy.stats.norm.cdf(lambda_j * np.sqrt(n_j) / sigma_j)
    if lambda_j <= 0.:
        return 0.
    return (float(lambda_j) ** 2) / (float(sigma_j) ** 2 + float(lambda_j) ** 2)


def F(lambda_j_star, means, s, N, j_star):
    lambdas = means - (means[j_star] + lambda_j_star)
    lambdas[j_star] = lambda_j_star
    # print(lambdas)
    # print([F_j(lambdas[j], N[j], s[j]) for j in range(ALGORITHMS)])
    # assert False
    return np.prod([F_j(lambdas[j], N[j], s[j]) for j in range(ALGORITHMS)])


def hypothesis_test_new(s, N, j_star, means, significance_level, debug):
    f_onedim = lambda lambda_j_star: -F(lambda_j_star, means, s, N, j_star)
    if N[j_star] >= 30:
        lambda_j_star_0 = scipy.stats.norm.ppf(significance_level ** (1 / ALGORITHMS)) * s[j_star] / np.sqrt(N[j_star])
    else:
        alpha = significance_level ** (1 / ALGORITHMS)
        lambda_j_star_0 = np.sqrt(alpha * float(s[j_star]) ** 2 / (N[j_star] * (1 - alpha)))
    debug_print(f"lambda_j_star_0={lambda_j_star_0}", debug)
    debug_print(f"F(lambda_j_star_0)={f_onedim(lambda_j_star_0)}", debug)
    lambda_j_star_opt = mystic.solvers.fmin(f_onedim, [lambda_j_star_0], disp=debug)
    debug_print(f"lambda_j_star_opt={lambda_j_star_opt}, level={-f_onedim(lambda_j_star_opt)}", debug)
    return -f_onedim(lambda_j_star_opt) >= significance_level


def estimate_new_n_new(s, N, j_star, means, significance_level, debug):
    def return_handler(N_new):
        N_new = np.clip(N_new, N, 8 * N)
        N_new = [min(N_new[i], 30) if N[i] < 30. else N_new[i] for i in range(ALGORITHMS)]
        return [int(x) for x in np.ceil(N_new)]

    bounds = [(2, 100000) for _ in range(ALGORITHMS)] + [(0, 100000)]

    def constraint_f(x):
        lambda_j_star = x[-1]
        lambdas = means - (means[j_star] + lambda_j_star)
        lambdas[j_star] = lambda_j_star
        Ns = x[:-1]
        # print(lambdas, Ns, s, [F_j(lambdas[j], Ns[j], s[j]) for j in range(ALGORITHMS)])
        return -(np.prod([F_j(lambdas[j], Ns[j], s[j]) for j in range(ALGORITHMS)]) - significance_level)

    @linear_inequality(constraint_f, k=1e100)
    def penalty(x):
        return 0.0

    index = np.arange(ALGORITHMS)
    j_prime = np.argmin(means[index != j_star])
    if j_prime >= j_star:
        j_prime += 1
    lambda_j_star_0 = (means[j_prime] - means[j_star]) / 2.
    lambdas_0 = means - (means[j_star] + lambda_j_star_0)
    lambdas_0[j_star] = lambda_j_star_0
    # Calculate x0 that would make the test pass for this lambda_j_star_0
    target_alpha = significance_level ** (1 / ALGORITHMS)
    x0 = 2 * N
    for j in range(ALGORITHMS):
        if F_j(lambdas_0[j], 29, s[j]) >= target_alpha:
            # Search for <= 29.:
            assert lambdas_0[j] > 0.
            x0[j] = float(s[j]) ** 2 * target_alpha / ((1 - target_alpha) * float(lambdas_0[j]) ** 2)
        else:
            # Search for >= 30.
            q_0 = scipy.stats.norm.ppf(target_alpha)
            if lambdas_0[j] == 0. and s[j] > 0.:
                return 2 * N
            if lambdas_0[j] == 0. and s[j] == 0.:
                x0[j] = max(N[j], 30)
                continue
            x0[j] = max(q_0 ** 2 * float(s[j]) ** 2 / (float(lambdas_0[j]) ** 2), 30)
        x0[j] = max(N[j], x0[j])
    debug_print(f"x0:{x0}, lambda_j_star_0: {lambda_j_star_0}", debug)
    debug_print(f"Penalty function for x0: {constraint_f(np.concatenate((x0, [lambda_j_star_0])))}", debug)
    solution = mystic.solvers.diffev(lambda x: obj_n(x, means), np.concatenate((x0, [lambda_j_star_0])),
                                     penalty=penalty, bounds=bounds, maxfun=1000, npop=40, disp=debug)

    debug_print(solution, debug)
    debug_print(f"Penalty function for x_opt: {constraint_f(solution)}", debug)
    return return_handler(np.array(solution)[:ALGORITHMS])


def is_significant_new(runtimes, sigma_upper=300000, significance_level=0.95, debug=False):
    runtimes = [np.array(x) for x in runtimes]
    N = np.array([len(x) for x in runtimes])
    debug_print(f"N: {N}", debug)
    means = np.array([x.mean() for x in runtimes])
    debug_print(f"Means: {means}", debug)
    j_star = np.argmin(means)
    s = np.array([np.sqrt(np.var(x, ddof=1)) if len(x) >= 30. else sigma_upper for x in runtimes])
    debug_print(f"Est. |S|: {s}", debug)
    if hypothesis_test_new(s, N, j_star, means, significance_level, debug):
        return None
    # Do estimation for new N
    return estimate_new_n_new(s, N, j_star, means, significance_level, debug)
