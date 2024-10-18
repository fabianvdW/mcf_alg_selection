import numpy as np
from scipy.stats import shapiro
import load_data
from matplotlib import pyplot as plt

if __name__ == '__main__':
    runtime_list = load_data.load_data()

    print(runtime_list.shape)
    print(runtime_list[13, 3, :])

    results = {}
    tested_instances = 0
    passes_test = 0
    all_runtime_limit = 0
    for instance in range(len(runtime_list)):
        for j in range(4):
            results[(instance, j)] = shapiro(runtime_list[instance, j,:])[1]
            tested_instances += 1
            if results[(instance, j)] > 0.05:
                passes_test += 1
            if all(runtime_list[instance, j, :] == 10000000.):
                all_runtime_limit += 1
    print(tested_instances, passes_test, all_runtime_limit)
    data = runtime_list[14, 0, :]

    plt.figure(figsize=(10, 6))  # Set the figure size

    # Create the histogram
    n, bins, patches = plt.hist(data, bins=30, edgecolor='black')

    # Add labels and title
    plt.xlabel('Runtime (in microseconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Runtime', fontsize=14, fontweight='bold')

    # Add grid for better readability
    plt.grid(axis='y', alpha=0.75, linestyle='--')

    # Improve tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add a text box with statistics
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    stats_text = f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std_dev:.2f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('runtime_histogram.png', dpi=300, bbox_inches='tight')

    plt.show()
