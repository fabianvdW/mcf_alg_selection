import numpy as np
import scipy.stats
import mystic
from mystic.penalty import quadratic_inequality, linear_inequality

if __name__ == "__main__":

    #runtimes = [[24, 24, 24, 24, 24], [110, 58], [16, 15, 15, 16, 15, 28, 16, 16, 16, 16, 15, 16, 15, 16, 16, 15, 15, 15, 16, 16, 15, 16, 16, 15, 16, 16, 19, 15, 20, 15, 15, 16, 15, 15, 16, 16, 16, 25, 16, 15, 15, 15, 15, 19, 16, 15, 16, 15, 16, 15, 15, 16, 17, 16, 19, 15, 15, 16, 16, 15, 15, 15, 15, 15, 16, 19, 18, 16, 16, 15, 15, 16, 16, 16, 23, 41, 15, 15, 15, 15, 15, 16, 16, 15, 15, 16, 29, 16, 16, 15, 16, 15, 15, 17, 15, 16, 15, 15, 27, 15, 17, 15, 18, 15, 15, 16, 15, 15, 15, 16, 15, 16, 16, 16, 16, 16, 16, 19, 15, 16, 15, 16, 15, 15, 15, 15, 15, 16, 16, 15, 16, 16, 16, 16, 17, 17, 21, 21, 16, 16, 23, 16, 16, 15, 15, 16, 16, 16, 21, 21, 16, 16, 15, 16, 18, 16, 19, 16, 16, 15, 16, 21, 18, 15, 16, 16, 16, 18, 23, 15, 16, 15, 21, 16, 15, 16, 16, 15, 15, 16, 15, 15, 16, 23, 15, 15, 15, 16, 16, 16, 15, 15, 15, 17, 15, 16, 16, 15, 16, 15, 16, 15, 16, 15, 18, 16, 16, 16, 18, 16, 16, 16, 16, 22, 22, 18, 18, 16, 15, 15, 15, 15, 15, 15, 15, 17, 15, 15, 16, 16, 15, 15, 16, 16, 15, 16, 15, 15, 15, 15, 18, 16, 16, 16, 15, 16, 15, 15, 16, 16, 15, 15, 16, 16, 15, 23, 17, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 16, 18, 15, 19, 16, 15, 15, 15, 15, 15, 15, 15, 16, 15, 16, 15, 15, 15, 15, 16, 16, 16, 15, 15, 15, 17, 16, 19, 18, 15, 16, 15, 15, 15, 16, 16, 23, 16, 16, 15, 16, 18, 15, 16, 16, 16, 15, 21, 17, 16, 16, 15, 16, 15, 16, 16, 17, 31, 16, 16, 16, 15, 15, 15, 15, 29, 16, 15, 15, 16, 15, 15, 16, 15, 15, 16, 15, 16, 20, 20, 20, 16, 15, 16, 16, 15, 16, 15, 15, 16, 15, 16, 15, 16, 16, 19, 15, 16, 44, 15, 16, 15, 16, 16, 16, 22, 21, 16, 16, 15, 15, 16, 15, 16, 16, 30, 29, 16, 15, 43, 15, 16, 16, 29, 16, 22, 15, 16, 16, 15, 16, 16, 30, 15, 15, 16, 16, 15, 16, 16, 15, 15, 15, 16, 20, 15, 23, 16, 16, 15, 15, 16, 20, 16, 15, 16, 16, 15, 15, 16, 15, 15, 15, 15, 16, 16, 16, 16, 15, 20, 20, 15, 16, 15, 15, 15, 15, 16, 16, 15, 16, 15, 16, 16, 25, 16, 16, 15, 15, 16, 16, 16, 16, 15, 15, 15, 28, 16, 16, 16, 15, 15, 15, 16, 16, 15, 15, 21, 15, 15, 15, 16, 15, 20, 19, 16, 16, 15, 15, 16, 16, 16, 15, 15, 15, 16, 16], [16, 16, 16, 15, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 15, 15, 16, 16, 16, 16, 16, 16, 16, 15, 16, 16, 15, 15, 16, 16, 16, 16, 19, 15, 16, 18, 16, 16, 15, 16, 15, 15, 16, 18, 16, 16, 16, 16, 16, 16, 15, 15, 15, 16, 16, 17, 16, 16, 22, 16, 16, 16, 15, 16, 17, 15, 15, 16, 16, 16, 16, 16, 16, 16, 15, 16, 16, 16, 17, 16, 16, 16, 15, 16, 16, 20, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 17, 16, 16, 20, 16, 20, 15, 15, 18, 28, 15, 16, 15, 18, 16, 16, 16, 16, 15, 15, 28, 15, 16, 30, 17, 16, 16, 16, 15, 16, 15, 15, 16, 15, 16, 17, 21, 21, 22, 16, 42, 16, 16, 17, 15, 24, 24, 23, 16, 15, 16, 16, 16, 15, 18, 15, 16, 16, 15, 15, 16, 18, 16, 16, 16, 16, 16, 16, 17, 16, 16, 16, 16, 15, 15, 20, 19, 19, 19, 16, 18, 15, 42, 15, 16, 18, 19, 19, 16, 15, 16, 16, 15, 16, 20, 21, 20, 19, 16, 16, 16, 17, 15, 16, 23, 23, 16, 30, 16, 16, 16, 15, 16, 16, 16, 16, 16, 15, 16, 19, 16, 15, 16, 16, 16, 16, 16, 15, 16, 16, 16, 16, 21, 15, 16, 16, 16, 16, 16, 15, 16, 16, 16, 16, 19, 16, 16, 16, 15, 16, 16, 16, 17, 18, 16, 16, 16, 16, 16, 16, 15, 15, 15, 16, 16, 16, 15, 15, 16, 16, 16, 24, 24, 16, 16, 15, 16, 23, 16, 16, 16, 16, 16, 16, 16, 15, 16, 15, 16, 16, 16, 33, 16, 16, 16, 15, 15, 15, 16, 18, 16, 16, 16, 15, 16, 16, 15, 16, 16, 16, 16, 15, 15, 15, 16, 18, 16, 15, 23, 16, 16, 16, 16, 15, 29, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 16, 31, 16, 16, 17, 16, 16, 16, 15, 16, 16, 15, 16, 16, 16, 16, 29, 16, 16, 16, 16, 20, 16, 16, 16, 16, 16, 33, 16, 16, 15, 16, 16, 15, 16, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16, 44, 17, 16, 15, 16, 16, 15, 15, 16, 17, 16, 15, 16, 16, 16, 28, 22, 16, 15, 16, 15, 16, 16, 16, 16, 17, 16, 15, 16, 16, 15, 15, 16, 15, 16, 16, 16, 15, 16, 36, 16, 16, 16, 16, 34, 16, 18, 16, 15, 16, 16, 16, 16, 16, 16, 29, 16, 20, 16, 16, 16, 16, 15, 15, 16, 16, 16, 16, 15, 15, 16, 16, 16, 16, 15, 15, 16, 16, 16, 16, 29, 21, 15, 16, 16, 16, 24, 24, 16, 15, 15, 16, 16, 16, 16, 16, 15, 16, 16, 19, 22, 16, 15, 16, 23, 15, 16, 15, 15]]
    runtimes = [[10414, 10532], [36866, 36865], [3390, 3337, 3919, 3441, 3938, 4064, 3746, 3268, 3729, 3706, 3798, 3423, 3267, 4183, 3261, 3291, 3329, 3386, 3833, 3283, 3463, 3939, 4108, 4379, 3620, 3361, 3774, 3584, 3908, 3294, 4239, 3282, 3300, 4519, 3475, 3294, 3436, 3609, 3869, 4222, 3371, 3909, 3877, 3404, 3833, 3271, 4060, 3355, 3302, 3333, 3270, 3327, 3268, 3273, 3255, 3856, 3894, 3328, 3312, 3720, 3286, 3781, 3348, 3298, 3974, 3617, 3385, 3410, 3291, 3293, 4203, 3698, 3404, 3273, 4140, 4035, 3337, 3337, 3368, 3323, 3530, 3343, 3971, 3814, 3807, 3360, 3351, 3601, 3631, 3343, 3347, 3965, 3452, 3266, 3267, 3320, 3801, 3265, 3770, 3374, 4129, 3265, 3424, 3270, 3337, 4273, 3762, 3258, 3638, 4059, 3851, 3319, 3337, 3740, 3263, 3761, 4069, 4182, 3334, 3661, 3883], [3713, 3759, 3420, 4350, 3388, 3309, 3626, 3483, 4178, 3352, 3515, 3831, 3541, 3475, 3309, 3869, 3985, 4353, 3423, 3373, 4214, 3352, 3324, 3391, 3839, 4042, 4515, 3317, 3943, 3324, 3649, 3988, 3329, 3317, 3844, 3850, 3873, 3475, 3822, 3383, 3326, 3328, 3458, 4019, 4297, 3333, 3811, 3345, 3314, 3546, 4030, 3968, 3823, 3488, 3991, 3612, 3312, 4652, 3818, 3326, 3828, 4112, 3348, 3384, 3836, 3336, 3961, 3356, 3430, 3327, 3493, 3382, 3371, 3794, 3375, 3340, 3332, 3318, 3322, 3465, 3326, 3896, 4182, 3833, 3328, 3312, 4224, 3939, 4273, 3919, 3316, 3437, 3328, 3404, 3762, 4290, 3406, 3311, 4245, 3343, 3347, 3320, 3714, 3345, 3333, 3519, 3907, 3972, 3347, 3324, 3687, 3680, 3605, 4098, 4281, 3373, 3855, 3409, 3415, 3720, 3869]]
    runtimes = [np.array(x) for x in runtimes]
    algorithms, N = len(runtimes), np.array([len(x) for x in runtimes])
    print(N)
    means = np.array([x.mean() for x in runtimes])
    print(means)
    s = np.array([np.sqrt(np.var(x, ddof=1)) for x in runtimes])
    print(s)
    j_star = np.argmin(means)
    min_mean = means[j_star]
    a = min_mean - means
    b = s

    # Find initial solution
    q = [scipy.stats.t.ppf(0.95 ** (1 / algorithms), df=N[j] - 1) for j in range(algorithms)]
    means[j_star] = np.inf
    j_prime = np.argmin(means)
    means[j_star] = min_mean

    N_new = np.array([0 for j in range(algorithms)])
    N_initial = ((q[j_star] * b[j_star] + q[j_prime] * b[j_prime]) / a[j_prime]) ** 2
    N_new[j_star] = N_initial
    for j in range(algorithms):
        if j == j_star:
            continue
        else:
            N_new[j] = ((q[j] * b[j]) / (a[j] + b[j_star] * q[j_star] / np.sqrt(N_initial))) ** 2
    N_new = np.clip(N_new, N, 1600*N)
    N_new = [int(x) for x in np.ceil(N_new)]
    print(f"Initial N_new: {N_new}")
    print(f"Initial opt value: {means @ N_new}")
    x0 = np.concatenate((N_new, q))

    def obj(x):
        return means @ x[:algorithms]
    print

    def g(x):
        res = np.prod(scipy.stats.t.cdf(x[algorithms:], df=x[:algorithms] - 1)) - 0.95
        return -res

    def interval_constraints(j,x):
        if any(x < 0.):
            return 1
        if j==j_star:
            return 0
        else:
            return a[j] + b[j_star] * x[algorithms + j_star] / np.sqrt(x[j_star]) + b[j] * x[algorithms +j] / np.sqrt(x[j])

    @quadratic_inequality(lambda x: interval_constraints(0, x), k=1e20)
    @quadratic_inequality(lambda x: interval_constraints(1, x), k=1e20)
    @quadratic_inequality(lambda x: interval_constraints(2, x), k=1e20)
    @quadratic_inequality(lambda x: interval_constraints(3, x), k=1e20)
    @linear_inequality(g,k=1e100)
    def penalty(x):
        return 0.0

    import time
    curr_time = time.time()
    solution = mystic.solvers.fmin(obj, x0, penalty=penalty)
    print(f"Time {time.time() - curr_time}s")
    print(solution)
    print(f"Obj: {obj(solution)}")
    def g_orig(x):
        res = np.prod(scipy.stats.t.cdf(x[algorithms:], df=x[:algorithms] - 1))
        return res
    print(g_orig(solution))
    for j in range(algorithms):
        if j==j_star:
            continue
        print(interval_constraints(j, solution))