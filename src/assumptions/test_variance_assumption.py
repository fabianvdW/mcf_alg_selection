import numpy as np
import load_data

if __name__ == '__main__':
    runtime_list = load_data.load_data()

    # Shape: Realization, Algo, Index
    print(runtime_list.shape)
    # hoeffding independence test for the following pairwise combinations
    stds = []
    counter = 0
    for instance in range(len(runtime_list)):
        for j in range(4):
            std = runtime_list[instance, j, :].std()
            if std >= 200000.:
                counter +=1
                print(instance, j, runtime_list[instance, j, :])
            stds.append(std)

    print(stds)
    print(np.quantile(stds, 0.95))
    print(np.max(stds))
    print(counter)