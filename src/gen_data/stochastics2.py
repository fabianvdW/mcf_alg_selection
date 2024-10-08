import numpy as np
from typing import Optional


def boundary_g_c(c, t):
    return np.sqrt((c + np.log(t + 1)) * (t + 1))


def get_sample_variance(runtimes_a, runtimes_b, mean_a, mean_b):
    assert len(runtimes_a) == len(runtimes_b)
    n = len(runtimes_a)
    diff = runtimes_a - runtimes_b - (mean_a - mean_b)
    return 1 / (n - 1) * np.sum(diff ** 2)


class SequentialProcedure:
    def __init__(self, initial_runtimes, significance_level=0.95, debug=False):
        assert all(len(x) >= 2 for x in initial_runtimes)

        self.debug = debug
        self.runtimes = None
        self.alpha = 1 - significance_level
        self.k = len(initial_runtimes)
        self.c = -2. * np.log(2. * self.alpha / (self.k - 1))
        self.I = [i for i in range(self.k)]
        self.means = None
        self.sample_variances = None
        self.n = None
        self.set_runtimes(initial_runtimes)

    def set_runtimes(self, runtimes):
        self.runtimes = [-np.array(x) for x in runtimes]
        assert len(set(len(self.runtimes[i]) for i in self.I)) == 1
        self.update()

    def update(self):
        self.n = len(self.runtimes[self.I[0]])
        self.means = np.array([x.mean() for x in self.runtimes])
        self.sample_variances = {}
        for i in self.I:
            self.sample_variances[i] = {}
            for j in self.I:
                if i == j:
                    continue
                if j < i:
                    self.sample_variances[i][j] = self.sample_variances[j][i]
                    continue
                self.sample_variances[i][j] = get_sample_variance(self.runtimes[i], self.runtimes[j], self.means[i],
                                                                  self.means[j])

    def passes_check(self, i: int, j: int):
        if self.sample_variances[i][j] == 0.:
            return False
        tau_i_j = self.n / self.sample_variances[i][j]
        z_i_j = tau_i_j * (self.means[i] - self.means[j])
        print(i, j, z_i_j, -boundary_g_c(self.c, tau_i_j))
        return z_i_j <= - boundary_g_c(self.c, tau_i_j)

    def screen(self):
        I_old = self.I.copy()
        self.I = []
        for i in I_old:
            if any(i != j and self.passes_check(i, j) for j in I_old):
                continue
            self.I.append(i)

    def is_finished(self) -> Optional[int]:
        if len(self.I) == 1:
            return self.I[0]
        return None
