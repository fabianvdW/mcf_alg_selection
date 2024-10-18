import itertools

import numpy as np
from scipy.io import savemat
from XtendedCorrel import hoeffding
from hoeffd2 import hoeffd2
import load_data
from scipy.io import loadmat
p_value_matrix = loadmat('p_value_matrix.mat')['p_value_matrix']

if __name__ == '__main__':
    runtime_list = load_data.load_data()

    # Shape: Realization, Algo, Index
    print(runtime_list.shape)
    print(runtime_list[1,1, :])
    savemat('runtime_data.mat', {'runtime_list': runtime_list})
    assert False

    p_value_matrix = loadmat('p_value_matrix.mat')['p_value_matrix']
    # Print resuts to typst table
    def gen_table(columns):
        for i_1 in combinations_indices:
            row_name = f'$X_{i_1[1] + 1}^{i_1[0] + 1}$'
            row_values = [f"{(results[(i_1, column)]):.4f}" if (i_1, column) in results else '"-"' for column
                          in columns]
            print(f'({row_name},' + ",".join(row_values) + '),')


    print("Table 1:")
    gen_table(combinations_indices[:8])
    print("Table 2: ")
    gen_table(combinations_indices[8:])
    print(f"Out of {pairwise_comparisons}, {pairwise_comparisons_lower_than_p} were lower than p=0.05")
