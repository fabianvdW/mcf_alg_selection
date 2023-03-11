import numpy as np
import scipy.stats
import mystic
from mystic.penalty import linear_inequality

ALGORITHMS = 4


def debug_print(s, debug):
    if debug:
        print(s)


def g_norm(q):
    return np.prod(scipy.stats.norm.cdf(q))


def g_t_1(q, N):
    return np.prod(scipy.stats.t.cdf(q, df=np.array(N) - 1))


def solve_qs_1(j_star, a, b, q_j_star):
    N = len(a)

    def solve_single(j):
        if j == j_star:
            return q_j_star
        elif b[j] == 0.:
            return np.inf
        elif b[j_star] == 0.:
            return -a[j] / b[j]
        else:
            return -(a[j] + b[j_star] * q_j_star) / b[j]

    return [solve_single(j) for j in range(N)]


def hypothesis_test(N, j_star, a, b, significance_level, debug, normal_dist=True):
    if normal_dist:
        q0 = solve_qs_1(j_star, a, b, scipy.stats.norm.ppf(significance_level ** (1 / ALGORITHMS)))
        g_onedim = lambda q_j_star: -g_norm(solve_qs_1(j_star, a, b, q_j_star[0]))
    else:
        q0_j_star = scipy.stats.t.ppf(significance_level ** (1 / ALGORITHMS), df=N[j_star] - 1) if b[
                                                                                                       j_star] > 0. else np.inf
        q0 = solve_qs_1(j_star, a, b, q0_j_star)
        g_onedim = lambda q_j_star: -g_t_1(solve_qs_1(j_star, a, b, q_j_star[0]), N)
    if b[j_star] > 0.:
        q_j_star = mystic.solvers.fmin(g_onedim, [q0[j_star]], disp=debug)
        q = solve_qs_1(j_star, a, b, q_j_star[0])
    else:
        q = q0
    debug_print(f"q's in {'first' if normal_dist else 'second'} test: {q}", debug)
    debug_print(
        f"Significance level reached in {'first' if normal_dist else 'second'} test: {g_norm(q) if normal_dist else g_t_1(q, N)}",
        debug)
    return normal_dist and g_norm(q) >= significance_level or not normal_dist and g_t_1(q, N) >= significance_level


def obj_n(x, means, process_creation_time=200000):
    return (means + process_creation_time) @ x[:len(means)]


# Where b[j_star] was 0. earlier
def solve_qs_2_case1(N, a, b):
    algorithms = len(a)
    # Equation comes from (8) solved for q_j in the equality case
    return np.array([- a[j] / b[j] * np.sqrt(N[j]) if N[j] >= 0. else -np.inf for j in range(algorithms)])


def solve_ns_case1(q, a, b):
    # Equation comes from (8) solved for N_j in the equality case
    return ((b * q) / a) ** 2


def g_t_2_case1(N, a, b, significance_level):
    if any(np.array(N) < 2.):
        return significance_level
    qs = solve_qs_2_case1(N, a, b)
    # Write side constraint as basis for penalty function
    return -(g_t_1(qs, N) - significance_level)


# b[j_star] == 0.
def estimate_new_n_case1(N, means, a, b, significance_level, debug):
    algorithms = len(a)

    # Calculate almost-feasible solution x0
    q0 = np.array([scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[j] - 1) for j in range(algorithms)])
    x0 = np.maximum(N, solve_ns_case1(q0, a, b))
    debug_print(f"Case 1: x0={x0}", debug)
    debug_print(f"Case 1: side constraint g_t_1 of x0 ={g_t_1(q0, x0)}", debug)

    # Prepare mystic
    g_t_2_case1_adj = lambda x: g_t_2_case1(x, a, b, significance_level)

    @linear_inequality(g_t_2_case1_adj, k=1e100)
    def penalty(x):
        return 0.0

    bounds = [(2, 100000) for _ in range(algorithms)]
    solution = mystic.solvers.diffev(lambda x: obj_n(x, means), x0, penalty=penalty, bounds=bounds,
                                     maxfun=1000, npop=40, disp=debug)
    # Have a look at the solution
    debug_print(f"Case 1: estimation received N={solution}", debug)
    q = solve_qs_2_case1(solution, a, b)
    debug_print(f"Case 1: solved q={q}", debug)
    debug_print(f"Case 1: side constraint g_t_1={g_t_1(q, solution)}", debug)
    debug_print(
        f"Case 1: Objective solution={obj_n(solution, means)} vs Objective x0: {obj_n(x0, means)} (ratio = {obj_n(solution, means) / obj_n(x0, means)})",
        debug)
    return solution


def solve_qs_2_case2(x, a, b, j_star):
    # x is of the form [N, q_j_star]
    algorithms = len(a)
    q_j_star = x[algorithms]

    # Equation obtained by NLP (12)
    def solve_q(j):
        if j == j_star:
            return q_j_star
        elif x[j_star] <= 1e-5 or x[j] <= 1e-5:
            return -np.inf
        else:
            return -(a[j] + b[j_star] * q_j_star / np.sqrt(x[j_star])) / b[j] * np.sqrt(x[j])

    return np.array([solve_q(j) for j in range(algorithms)])


def solve_ns_case2(q, a, b, j_star, n_j_star):
    assert n_j_star >= 1e-5
    algorithms = len(a)

    # Equation obtained by NLP (12)
    def solve_n(j):
        if j == j_star:
            return n_j_star
        else:
            c = a[j] + b[j_star] * q[j_star] / np.sqrt(n_j_star)
            assert c <= -1e-10
            return (b[j] * q[j] / c) ** 2

    return np.array([solve_n(j) for j in range(algorithms)])


def solve_n_jstar(q, a, b, j_star, j_prime):
    # Taken from pdf, page 8
    return ((b[j_star] * q[j_star] + b[j_prime] * q[j_prime]) / a[j_prime]) ** 2


def g_t_2_case2(x, a, b, j_star, significance_level):
    # x is of the form [N, q_j_star]
    N = np.array(x)[:len(a)]
    if any(N < 2.):
        return significance_level
    # Write side constraint as basis for penalty function
    qs = solve_qs_2_case2(x, a, b, j_star)
    return -(g_t_1(qs, N) - significance_level)


# b[j_star] > 0.
def estimate_new_n_case2(N, means, a, b, significance_level, debug):
    algorithms = len(a)
    j_star = np.argmin(means)
    index = np.arange(algorithms)
    j_prime = np.argmin(means[index != j_star])
    if j_prime >= j_star:
        j_prime += 1

    # Calculate almost-feasible solution x0
    q0 = np.array([scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[j] - 1) for j in index])
    n0_j_star = solve_n_jstar(q0, a, b, j_star, j_prime)
    x0 = np.maximum(N, solve_ns_case2(q0, a, b, j_star, n0_j_star))
    debug_print(f"Case 2: x0={x0}", debug)
    debug_print(f"Case 2: side constraint g_t_1 of x0 ={g_t_1(q0, x0)}", debug)

    # Prepare mystic
    g_t_2_case2_adj = lambda x: g_t_2_case2(x, a, b, j_star, significance_level)

    @linear_inequality(g_t_2_case2_adj, k=1e100)
    def penalty(x):
        return 0.0

    bounds = [(2, 100000) for _ in index] + [(-100000, 100000)]
    solution = mystic.solvers.diffev(lambda x: obj_n(x, means), np.concatenate((x0, [q0[j_star]])), penalty=penalty,
                                     bounds=bounds,
                                     maxfun=1000, npop=40, disp=debug)
    # Have a look at the solution
    q = solve_qs_2_case2(solution, a, b, j_star)
    solution = np.array(solution)[:algorithms]
    debug_print(f"Case 2: estimation received N={solution}", debug)
    debug_print(f"Case 2: solved q={q}", debug)
    debug_print(f"Case 2: side constraint g_t_1={g_t_1(q, solution)}", debug)
    debug_print(
        f"Case 2: Objective solution={obj_n(solution, means)} vs Objective x0: {obj_n(x0, means)} (ratio = {obj_n(solution, means) / obj_n(x0, means)})",
        debug)
    return solution


def estimate_new_n(N, j_star, means, a, b, significance_level, debug):
    def return_handler(N_new):
        N_new = np.clip(N_new, 0.75 * N, 16 * N)
        return [int(x) for x in np.ceil(N_new)]

    index = np.arange(ALGORITHMS)

    # In the case where a[j_prime] == 0. there is no feasible solution to (7)
    j_prime = np.argmin(means[index != j_star])
    if j_prime >= j_star:
        j_prime += 1
    if np.isclose(a[j_prime], 0):
        return return_handler(4 * N)

    non_zeros = index[b > 0.]
    if b[j_star] == 0:
        N_est = estimate_new_n_case1(N[non_zeros], means[non_zeros], a[non_zeros], b[non_zeros], significance_level,
                                     debug)
    else:
        N_est = estimate_new_n_case2(N[non_zeros], means[non_zeros], a[non_zeros], b[non_zeros], significance_level,
                                     debug)
    N_new = np.ones(ALGORITHMS) * 2
    N_new[non_zeros] = N_est
    N_new = return_handler(N_new)
    debug_print(f"Obj. value of N-NLP: {means @ np.array(N_new)}", debug)
    return N_new


def is_significant(runtimes, sigma_upper=200000, significance_level=0.95, debug=False):
    runtimes = [np.array(x) for x in runtimes]
    N = np.array([len(x) for x in runtimes])
    debug_print(f"N: {N}", debug)
    means = np.array([x.mean() for x in runtimes])
    debug_print(f"Means: {means}", debug)
    j_star = np.argmin(means)
    min_mean = means[j_star]
    # First approach -> Baseline feasible solution as given in PDF used as start value, optimize NLP
    a = min_mean - means
    b = sigma_upper / np.sqrt(N)
    if hypothesis_test(N, j_star, a, b, significance_level, debug, normal_dist=True):
        return None
    elif all([x >= 2 for x in N]):
        # Second approach -> Baseline feasible solution as given in PDF used as start value, optimize NLP
        s = np.array([np.sqrt(np.var(x, ddof=1)) for x in runtimes])
        debug_print(f"Est. |S|: {s}", debug)
        b = s / np.sqrt(N)
        if hypothesis_test(N, j_star, a, b, significance_level, debug, normal_dist=False):
            return None
        else:
            # Do estimation for new N according to baseline feasible solution given in PDF used as input for NLP!
            b = s  # Definition of b changes, see pdf
            return estimate_new_n(N, j_star, means, a, b, significance_level, debug)
    else:
        return [2 for _ in range(ALGORITHMS)]


def is_significant_baseline(runtimes, sigma_upper=200000, significance_level=0.95, debug=False):
    runtimes = [np.array(x) for x in runtimes]
    N = np.array([len(x) for x in runtimes])
    debug_print(f"N: {N}", debug)
    means = np.array([x.mean() for x in runtimes])
    debug_print(f"Means: {means}", debug)
    j_star = np.argmin(means)
    min_mean = means[j_star]
    # First approach -> Baseline feasible solution as given in PDF
    a = min_mean - means
    b = sigma_upper / np.sqrt(N)
    q_j_star = scipy.stats.norm.ppf(significance_level ** (1 / ALGORITHMS))
    q = solve_qs_1(j_star, a, b, q_j_star)
    debug_print(f"q's in first test: {q}", debug)
    debug_print(f"Significance level reached in first test: {g_norm(q)}", debug)
    if g_norm(q) >= significance_level:
        return None
    elif all([x >= 2 for x in N]):
        # Second approach -> Baseline feasible solution as given in PDF
        s = np.array([np.sqrt(np.var(x, ddof=1)) for x in runtimes])
        debug_print(f"Est. |S|: {s}", debug)
        b = s / np.sqrt(N)
        q_j_star = scipy.stats.t.ppf(significance_level ** (1 / ALGORITHMS), df=N[j_star] - 1) if b[
                                                                                                      j_star] > 0. else np.inf
        q = solve_qs_1(j_star, a, b, q_j_star)
        debug_print(f"q's in second test: {q}", debug)
        debug_print(f"Significance level reached in second test: {g_t_1(q, N)}", debug)
        if g_t_1(q, N) >= significance_level:
            return None
        else:
            N_orig = np.copy(N)
            # Do estimation for new N according to baseline feasible solution given in PDF
            b = s  # Definition of b changes, see pdf
            index = np.arange(ALGORITHMS)
            j_prime = np.argmin(means[index != j_star])
            if j_prime >= j_star:
                j_prime += 1
            if np.isclose(a[j_prime], 0):
                return 4 * N
            non_zeros = index[b > 0.]
            if b[j_star] == 0.:
                a, b, N, means = a[non_zeros], b[non_zeros], N[non_zeros], means[non_zeros]
                algorithms = len(a)

                # Calculate almost-feasible solution x0
                q0 = np.array(
                    [scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[j] - 1) for j in range(algorithms)])
                x0 = np.maximum(N, solve_ns_case1(q0, a, b))
            else:
                a, b, N, means = a[non_zeros], b[non_zeros], N[non_zeros], means[non_zeros]
                algorithms = len(a)
                j_star = np.argmin(means)
                index = np.arange(algorithms)
                j_prime = np.argmin(means[index != j_star])
                if j_prime >= j_star:
                    j_prime += 1

                # Calculate almost-feasible solution x0
                q0 = np.array([scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[j] - 1) for j in index])
                n0_j_star = solve_n_jstar(q0, a, b, j_star, j_prime)
                x0 = np.maximum(N, solve_ns_case2(q0, a, b, j_star, n0_j_star))
            N_new = np.ones(ALGORITHMS) * 2
            N_new[non_zeros] = x0
            N_new = np.clip(N_new, 0.75 * N_orig, 8 * N_orig)
            N_new = [int(x) for x in np.ceil(N_new)]
            debug_print(f"Obj. value of N-NLP: {obj_n(N_new, means)}", debug)
            return N_new

    return [2 for _ in range(ALGORITHMS)]


def is_significant_naive(runtimes, sigma_upper=200000, significance_level=0.95, debug=False):
    runtimes = [np.array(x) for x in runtimes]
    algorithms, N = len(runtimes), [len(x) for x in runtimes]
    debug_print(f"N:{N}", debug)
    means = np.array([x.mean() for x in runtimes])
    debug_print(f"Means: {means}", debug)
    j_star = np.argmin(means)
    min_mean = means[j_star]
    # First approach
    first_test = True
    q = scipy.stats.norm.ppf(significance_level ** (1 / algorithms))
    upper_min_algo = min_mean + sigma_upper * q / np.sqrt(N[j_star])
    for j in range(algorithms):
        if j == j_star:
            continue
        lower_i = means[j] - sigma_upper * q / np.sqrt(N[j])
        if lower_i < upper_min_algo:
            first_test = False
            break
    if first_test:
        return None
    elif all([x >= 2 for x in N]):
        # Second approach
        s = [np.sqrt(np.var(x, ddof=1)) for x in runtimes]
        debug_print(f"Est. |S|: {s}", debug)
        q_hat_upper = scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[j_star] - 1)
        upper_min_algo = min_mean + s[j_star] * q_hat_upper / np.sqrt(N[j_star])
        debug_print(f"Upper bound of Interval I_{j_star}: {upper_min_algo}", debug)
        N_new = N.copy()
        N_new[j_star] *= 2
        second_test = True
        for j in range(algorithms):
            if j == j_star:
                continue
            q_hat_lower = scipy.stats.t.ppf(significance_level ** (1 / algorithms), df=N[j] - 1)
            lower_i = means[j] - s[j] * q_hat_lower / np.sqrt(N[j])
            debug_print(f"Upper bound of Interval I_{j}: {lower_i}", debug)
            if lower_i < upper_min_algo:
                second_test = False
                N_new[j] *= 2
        if second_test:
            return None
        else:
            return N_new
    return [2 for _ in range(algorithms)]
