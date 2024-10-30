# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.gen_data import util as g_util
from util import ALGO_NAMES
from data_loader_trad import get_data
import matplotlib as mpl
import seaborn as sns

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
mpl.rcParams["figure.dpi"] = 1200


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


def plot_fastest_algo_allgen():
    number_of_minimum = np.array([len(combined_data[combined_data["Minimum"] == alg]) for alg in ALGO_NAMES])
    patches, _ = plt.pie(
        number_of_minimum, colors=sns.color_palette("Set2", 7),
        labels=ALGO_NAMES
    )
    plt.savefig("pieplot_algorithms_all.png")

def plot_alg_generators():
    datanet = combined_data[combined_data["Generator"] == "NETGEN"]
    dataggen = combined_data[combined_data["Generator"] == "GRIDGEN"]
    dataggraph = combined_data[combined_data["Generator"] == "GRIDGRAPH"]
    datagoto = combined_data[combined_data["Generator"] == "GO_TO"]

    # Netgen
    number_of_minimum_net = np.array([len(datanet[datanet["Minimum"] == alg]) for alg in ALGO_NAMES])
    indices_where_not_0 = [i for i in range(len(number_of_minimum_net)) if number_of_minimum_net[i] != 0]

    fig, axs = plt.subplots(2, 3, figsize=(7, 5))
    patches, _ = axs[0, 0].pie(number_of_minimum_net[indices_where_not_0], colors=sns.color_palette("Set2", 7))
    axs[0, 0].set_title("Netgen")

    # Gridgen
    number_of_minimum_ggen = np.array([len(dataggen[dataggen["Minimum"] == alg]) for alg in ALGO_NAMES])
    indices_where_not_0 = [i for i in range(len(number_of_minimum_ggen)) if number_of_minimum_ggen[i] != 0]

    patches, _ = axs[0, 1].pie(number_of_minimum_ggen[indices_where_not_0], colors=sns.color_palette("Set2", 7))
    axs[0, 1].set_title("Gridgen")

    # Gridgraph
    number_of_minimum_ggraph = np.array([len(dataggraph[dataggraph["Minimum"] == alg]) for alg in ALGO_NAMES])
    indices_where_not_0 = [i for i in range(len(number_of_minimum_ggraph)) if number_of_minimum_ggraph[i] != 0]

    patches, _ = axs[1, 0].pie(number_of_minimum_ggraph[indices_where_not_0], colors=sns.color_palette("Set2", 7))
    fig.legend(patches[:-3], ["NS", "CS2", "SSP", "CAS"], loc="upper right")

    axs[1, 0].set_title("Gridgraph")

    # Goto
    number_of_minimum_goto = np.array([len(datagoto[datagoto["Minimum"] == alg]) for alg in ALGO_NAMES])
    indices_where_not_0 = [i for i in range(len(number_of_minimum_ggraph)) if number_of_minimum_goto[i] != 0]

    patches, _ = axs[1, 1].pie(number_of_minimum_goto[indices_where_not_0], colors=sns.color_palette("Set2", 7))
    axs[1, 1].set_title("Goto")

    axs[0, 2].axis("off")

    # Alle
    number_of_minimum = np.array([len(combined_data[combined_data["Minimum"] == alg]) for alg in ALGO_NAMES])
    indices_where_not_0 = [i for i in range(len(number_of_minimum)) if number_of_minimum[i] != 0]

    patches, _ = axs[1, 2].pie(number_of_minimum[indices_where_not_0], colors=sns.color_palette("Set2", 7))

    axs[1, 2].set_title("All")
    #TODO: Legend, make order of colors the same if possible.

    plt.savefig("test.png")

if __name__ == "__main__":
    combined_data_train = get_data(True)
    combined_data_test = get_data(False)
    combined_data = pd.concat([combined_data_train, combined_data_test])
    combined_data["Generator"] = combined_data.apply(get_algorithm_from_seed, axis=1)

    instances = len(combined_data)
    #plot_fastest_algo_allgen()
    plot_alg_generators()

"""








# plot_alg_generators()


# generator = "GRIDGEN"
# combined_data = combined_data[combined_data["Generator"] == generator]


def plot_results(alle=True):
    if alle:
        results = util.load_dict("../../experiments/basic_sklearn_classifiers/6/production_training_runs.json")
        classifiers = ["KNN", "SVC", "DTree", "RForest", "NN", "Ada", "NS"]
        size_ds = str(73130 // 5)
    else:
        results = util.load_dict("../../experiments/basic_sklearn_classifiers/2/production_training_runs.json")
        classifiers = ["KNN", "SVC", "GaussP", "Tree", "Forest", "NN", "Ada", "Gauss", "QDA", "NS"]
        size_ds = str(3119 // 5)

    accuracies = [x["test_accuracy"] for x in results[:-1] if x["name"] != "GaussianNB"]
    plt.figure(dpi=1200)
    plt.bar(
        classifiers,
        accuracies,
        color=[
            (0.4, 0.7607843137254902, 0.6470588235294118) if clf != "NS" else (
            0.9882352941176471, 0.5529411764705883, 0.3843137254901961)
            for clf in classifiers
        ],
    )
    # if alle:
    #     plt.title("Accuracy of all classifiers on " + size_ds + " test instances")
    # else:
    #     plt.title("Accuracy of all classifiers on " + size_ds + " Gridgraph test instances")

    classifiers.append("GT")

    if alle:
        # classifiers.remove("Gauss")
        time = [x["test_time"] / 10 ** 6 for x in results if x["name"] != "GaussianNB"]
    else:
        classifiers.remove("GaussP")
        time = [x["test_time"] / 10 ** 6 for x in results if x["name"] != "GaussianProcessClassifier"]

    plt.figure(dpi=1200)
    plt.bar(
        classifiers,
        time,
        color=[
            (
                (0.4, 0.7607843137254902, 0.6470588235294118)
                if clf != "NS" and clf != "GT"
                else (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)
            )
            for clf in classifiers
        ],
    )
    if alle:
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80], [10, 20, 30, 40, 50, 60, 70, 80])
        plt.ylabel("time in seconds")
        # plt.title("Time needed with algorithms predicted by classifier on " + size_ds + " test instances")
    else:
        plt.yticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        plt.ylabel("time in seconds")
        # plt.title("Time needed with algorithms predicted by classifier on " + size_ds + " Gridgraph test instances ")

    classifiers = ["DTree", "RForest", "NetworkSimplex", "GroundTruth"]
    time = [
        x["test_time"] / 10 ** 6
        for x in results
        if x["name"] in ["DecisionTreeClassifier", "RandomForestClassifier", "baseline", "GroundTruth"]
    ]
    plt.figure(dpi=1200)
    barplot = plt.bar(
        classifiers,
        time,
        color=[
            (
                (0.4, 0.7607843137254902, 0.6470588235294118)
                if clf != "NetworkSimplex" and clf != "GroundTruth"
                else (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)
            )
            for clf in classifiers
        ],
    )
    if alle:
        plt.ylim(53.5, 60)
        # plt.title("Time needed with algorithms predicted by classifier on " + size_ds + " test instances")
        bar_label = ["+1,04\%", "+0,74\%", "+8,5\%", ""]
        plt.yticks([54, 55, 56, 57, 58, 59], [54, 55, 56, 57, 58, 59])
        plt.ylabel("time in seconds")
    else:
        plt.ylim(1, 1.125654)
        # plt.title("Time needed with algorithms predicted by classifier on " + size_ds + " Gridgraph test instances")
        bar_label = ["+1,01\%", "+0,93\%", "+11,43\%", ""]
        plt.yticks([1, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12], [1, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12])
        plt.ylabel("time in seconds")

    def autolabel(rects):
        for idx, rect in enumerate(barplot):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, bar_label[idx], ha="center", va="bottom",
                     rotation=0)

    autolabel(barplot)


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
