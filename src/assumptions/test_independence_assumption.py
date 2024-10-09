import itertools

import numpy as np
from XtendedCorrel import hoeffding
from hoeffd2 import hoeffd2
import load_data

if __name__ == '__main__':
    runtime_list = load_data.load_data()

    means = runtime_list.mean(axis=2)
    # Shape: Realization, Algo, Index
    print(runtime_list.shape)
    # hoeffding independence test for the following pairwise combinations
    combinations_indices = []
    for i in range(4):
        combinations_indices.extend([(i, 0), (i, 1), (i, 49), (i, 99)])

    results = {}
    pairwise_comparisons = 0
    pairwise_comparisons_lower_than_p = 0
    for c_1, c_2 in itertools.combinations(combinations_indices, 2):
        r_1 = runtime_list[:, c_1[0], c_1[1]] - means[:, c_1[0]]
        r_2 = runtime_list[:, c_2[0], c_2[1]] - means[:, c_2[0]]
        results[(c_1, c_2)] = hoeffding(r_1, r_2)
        results[(c_2, c_1)] = results[(c_1, c_2)]
        pairwise_comparisons += 1
        if results[(c_1, c_2)] <= 0.05:
            pairwise_comparisons_lower_than_p += 1
        print(c_1, c_2, hoeffding(r_1, r_2), hoeffd2(r_1, r_2))


    # Print resuts to typst table
    def gen_table(columns):
        for i_1 in combinations_indices:
            row_name = f'$X_{i_1[1] + 1}^{i_1[0] + 1}$'
            transform = lambda x: 0 if x < 0 else x
            row_values = [f"{transform(results[(i_1, column)]):.4f}" if (i_1, column) in results else '"-"' for column
                          in columns]
            print(f'({row_name},' + ",".join(row_values) + '),')


    print("Table 1:")
    gen_table(combinations_indices[:8])
    print("Table 2: ")
    gen_table(combinations_indices[8:])
    print(f"Out of {pairwise_comparisons}, {pairwise_comparisons_lower_than_p} were lower than p=0.05")
