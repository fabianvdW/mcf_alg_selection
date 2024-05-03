import numpy as np
import scipy
from src.stochastics import is_significant

if __name__ == "__main__":
    X_1 = np.random.normal(loc=20, scale=10, size=60)
    X_2 = np.random.normal(loc=25, scale=9, size=80)
    print(scipy.stats.ttest_ind(X_1, X_2, alternative="less", equal_var=False))
    print(is_significant([X_1, X_2], sigma_upper=10, debug=True))
