# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.gen_data import util as g_util
from util import ALGO_NAMES, load_dict
from data_loader_trad import get_data, DATA_PATH
import matplotlib as mpl
import seaborn as sns
import os

plt.style.use("seaborn-v0_8")
sns.set_palette("pastel")

nice_fonts = {
    # Use LaTeX to write all text
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
COLORS = dict(zip(ALGO_NAMES, ["red", "green", "blue", "yellow", "black", "orange", "purple"]))

mpl.rcParams.update(nice_fonts)
mpl.rcParams["figure.dpi"] = 300


def get_algorithm_from_seed(seed):
    seed = seed.name
    if "NETGEN" in seed:
        return "NETGEN"
    elif "GRIDGEN" in seed:
        return "GRIDGEN"
    elif "GRIDGRAPH" in seed:
        return "GRIDGRAPH"
    elif "GOTO" in seed:
        return "GO_TO"


"""
def plot_features(gen):
    data = combined_data[combined_data["Generator"] == gen]
    for param in names_features[1:]:
        plt.figure(dpi=1200)
        for alg in ALGO_NAMES[0:4]:
            unique_values = sorted(data[param].unique())
            y = [data[alg][data[param] == u].mean() for u in unique_values]
            p = np.polyfit(unique_values, y, 10)
            y = np.polyval(p, unique_values)
            plt.plot(unique_values, y, c=COLORS[alg], label=alg)
        plt.yscale("log")
        plt.legend()
        plt.title("param: " + param + ", generator: " + gen)
"""


def plot_results():
    results = load_dict(os.path.join(DATA_PATH, "result_trad", "1", "production_training_runs.json"))
    classifiers = ["KNN", "SVC", "DTree", "RForest", "MLP", "ADA", "GNN", "NS", "Base"]
    accuracies = [x["test_accuracy"] for x in results]
    plt.figure(dpi=300)
    plt.bar(
        classifiers,
        accuracies,
        color=[
            (0.4, 0.7607843137254902, 0.6470588235294118) if clf != "NS" and clf != "Base" else (
                0.9882352941176471, 0.5529411764705883, 0.3843137254901961)
            for clf in classifiers
        ],
    )
    plt.savefig("results1.png")

    classifiers.append("GT")
    minruntime = results[0]['test_minruntime_sum']
    results.append({'name': 'GT', 'test_runtime_sum': minruntime})

    time = [x["test_runtime_sum"] / 10 ** 6 for x in results]

    plt.figure(dpi=300)
    plt.bar(
        classifiers,
        time,
        color=[
            (
                (0.4, 0.7607843137254902, 0.6470588235294118)
                if clf != "NS" and clf != "Base" and clf != "GT"
                else (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)
            )
            for clf in classifiers
        ],
    )
    plt.yticks([200, 400, 600, 800, 1000, 1200, 1400, 1600], [200, 400, 600, 800, 1000, 1200, 1400, 1600])
    plt.ylabel("time in seconds")
    plt.savefig("results2.png")

    #Dont show Network Simplex for thrid graphic, show ratios
    idx_ns = classifiers.index("NS")
    del classifiers[idx_ns]
    del time[idx_ns]
    ratios = [x / (minruntime / 10 ** 6) for x in time]
    print(ratios)
    plt.figure(dpi=300)
    plt.bar(
        classifiers,
        ratios,
        color=[
            (
                (0.4, 0.7607843137254902, 0.6470588235294118)
                if clf != "NS" and clf != "Base" and clf != "GT"
                else (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)
            )
            for clf in classifiers
        ],
    )
    plt.ylim(0.99, 1.06)
    plt.yticks([1., 1.01, 1.02, 1.03, 1.04, 1.05], [1., 1.01, 1.02, 1.03, 1.04, 1.05])
    plt.ylabel("Runtime ratio metric")
    plt.savefig("results3.png")



def plot_alg_generators():
    datanet = combined_data[combined_data["Generator"] == "NETGEN"]
    dataggen = combined_data[combined_data["Generator"] == "GRIDGEN"]
    dataggraph = combined_data[combined_data["Generator"] == "GRIDGRAPH"]
    datagoto = combined_data[combined_data["Generator"] == "GO_TO"]

    # Get Set2 colors
    set2_colors = sns.color_palette("Set2", 7)
    color_dict = {
        'NS': set2_colors[0],  # Green
        'CS2': set2_colors[1],  # Coral
        'SSP': set2_colors[2],  # Purple
        'CAS': set2_colors[3]  # Blue
    }

    # Function to reorder data and colors to maintain consistent positioning
    def prepare_pie_data(data_counts, algo_names):
        full_data = []
        colors = []
        labels = []
        for algo in algo_names:
            if algo in color_dict:  # Only include algorithms we want to show
                count = data_counts[algo_names.index(algo)]
                if count > 0:
                    full_data.append(count)
                    colors.append(color_dict[algo])
                    labels.append(algo)
        return full_data, colors, labels

    number_of_minimum = np.array([len(combined_data[combined_data["Minimum"] == alg]) for alg in ALGO_NAMES])
    data_all, colors_all, labels = prepare_pie_data(
        number_of_minimum,
        ALGO_NAMES
    )
    plt.pie(data_all, colors=colors_all, startangle=90, labels=labels)
    plt.savefig("pieplot_algorithms_all.png")

    fig, axs = plt.subplots(2, 3, figsize=(10, 7))

    # Netgen
    number_of_minimum_net = np.array([len(datanet[datanet["Minimum"] == alg]) for alg in ALGO_NAMES])
    data_net, colors_net, _ = prepare_pie_data(
        number_of_minimum_net,
        ALGO_NAMES
    )
    axs[0, 0].pie(data_net, colors=colors_net, startangle=90)
    axs[0, 0].set_title("Netgen")

    # Gridgen
    number_of_minimum_ggen = np.array([len(dataggen[dataggen["Minimum"] == alg]) for alg in ALGO_NAMES])
    data_ggen, colors_ggen, _ = prepare_pie_data(
        number_of_minimum_ggen,
        ALGO_NAMES
    )
    axs[0, 1].pie(data_ggen, colors=colors_ggen, startangle=90)
    axs[0, 1].set_title("Gridgen")

    # Gridgraph
    number_of_minimum_ggraph = np.array([len(dataggraph[dataggraph["Minimum"] == alg]) for alg in ALGO_NAMES])
    data_ggraph, colors_ggraph, _ = prepare_pie_data(
        number_of_minimum_ggraph,
        ALGO_NAMES
    )
    axs[1, 0].pie(data_ggraph, colors=colors_ggraph, startangle=90)
    axs[1, 0].set_title("Gridgraph")

    # Goto
    number_of_minimum_goto = np.array([len(datagoto[datagoto["Minimum"] == alg]) for alg in ALGO_NAMES])
    data_goto, colors_goto, _ = prepare_pie_data(
        number_of_minimum_goto,
        ALGO_NAMES
    )
    axs[1, 1].pie(data_goto, colors=colors_goto, startangle=90)
    axs[1, 1].set_title("Goto")

    # Remove empty subplot
    axs[0, 2].axis("off")

    # All data
    number_of_minimum = np.array([len(combined_data[combined_data["Minimum"] == alg]) for alg in ALGO_NAMES])
    data_all, colors_all, _ = prepare_pie_data(
        number_of_minimum,
        ALGO_NAMES
    )
    axs[1, 2].pie(data_all, colors=colors_all, startangle=90)
    axs[1, 2].set_title("All")

    # Create custom legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_dict[algo])
                       for algo in ['NS', 'CS2', 'SSP', 'CAS']]
    fig.legend(legend_elements, ['NS', 'CS2', 'SSP', 'CAS'],
               loc='center right', bbox_to_anchor=(0.98, 0.6))

    plt.tight_layout()
    plt.savefig("test.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    #combined_data_train = get_data(True)
    #combined_data_test = get_data(False)
    #combined_data = pd.concat([combined_data_train, combined_data_test])
    #combined_data["Generator"] = combined_data.apply(get_algorithm_from_seed, axis=1)

    #instances = len(combined_data)
    #plot_alg_generators()
    plot_results()

"""





# plot_results(True)


def plot_features():
    for param in names_features[1:]:
        plt.figure()
        for alg in names_output:
            unique_values = sorted(combined_data[param].unique())
            y = [combined_data[alg][combined_data[param] == u].mean() for u in unique_values]
            p = np.polyfit(unique_values, y, 4)
            plt.plot(unique_values, np.polyval(p, unique_values), c=colour[alg], label=alg)
        plt.yscale("log")
        plt.legend()
        plt.title(param)

"""

# plot_features("GRIDGRAPH")


# plot_features()
# plot_alg()

# for param in params:
#     plt.figure()    
#     for alg in names_output[:3]:
#         alg_list = []
#         alg_list_std_high = []
#         alg_list_std_low = []
#         for p in sorted(dict_params[param]):
#             rows_with_param = combined_data[combined_data[param] == p]
#             alg_list.append(rows_with_param[alg].mean())     
#             alg_list_std_high.append(rows_with_param[alg].quantile(q=0.8))
#             alg_list_std_low.append(rows_with_param[alg].quantile(q=0.2))

#         plt.plot(sorted(dict_params[param]), alg_list_std_high,c=colour[alg], linestyle="dotted")
#         plt.plot(sorted(dict_params[param]), alg_list_std_low,c=colour[alg], linestyle="dotted")
#         plt.plot(sorted(dict_params[param]), alg_list,c=colour[alg],label=alg)

#     plt.yscale("log")
#     plt.legend()
#     plt.title(param)
