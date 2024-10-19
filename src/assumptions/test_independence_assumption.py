import itertools

import numpy as np
from scipy.io import savemat
from XtendedCorrel import hoeffding
from hoeffd2 import hoeffd2
import load_data
from scipy.io import loadmat

if __name__ == '__main__':
    runtime_list = load_data.load_independence_data()

    # Shape: Realization, Algo, Index
    print(runtime_list.shape)
    print(runtime_list[1, 1, :])
    savemat('runtime_data.mat', {'runtime_list': runtime_list})

    p_value_matrix = loadmat('p_value_matrix.mat')['p_value_matrix']
    combinations_indices = []
    for i in range(4):
        combinations_indices.extend([(i, 0), (i, 1), (i, 49), (i, 99)])


    # Print resuts to typst table
    def gen_table(columns):
        for (index_1, i_1) in enumerate(combinations_indices):
            row_name = f'$X_{i_1[1] + 1}^{i_1[0] + 1}$'
            row_values = [f"{(p_value_matrix[index_1, column]):.4f}" if index_1 != column else '"-"' for column
                          in columns]
            print(f'({row_name},' + ",".join(row_values) + '),')


    print("Table 1:")
    gen_table(list(range(8)))
    print("Table 2: ")
    gen_table(list(range(8, 16)))
