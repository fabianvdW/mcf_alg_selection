# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:29:14 2021

@author: Philipp
"""
# features_of_graph = [seed, number of nodes, number of arcs, min cost, max cost, sum cost, mean cost, std cost, min cap, max cap,
#                 sum cap, mean cap, std cap, max of min of shortest path from supply to demand, supply, num of supply nodes]
# features of mst = [sum cost, mean cost,std cost, sum cap, mean cap, std cap]

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Path to features.txt
path_to_data = os.path.join("data", "generated_data")
path_to_time = os.path.join("lemon","existingresults.txt")
path_to_parent_dir = os.path.dirname(os.getcwd())

# read features.txt
names_features = ["seed","g_number_of_nodes", "g_number_of_arcs", "g_min_cost", "g_max_cost", "g_sum_cost",
                  "g_mean_cost", "g_std_cost", "g_min_cap",
                  "g_max_cap", "g_sum_cap", "g_mean_cap", "g_std_cap", "g_supply",
                  "g_number_of_supply_nodes", "g_arcs_source_sink", "mst_sum_cost",
                  "mst_mean_cost", "mst_std_cost", "mst_sum_cap", "mst_mean_cap", "mst_std_cap"]

names_output = ["Network Simplex", "CS2", "SSP", "CAS", "CC Simple", "CC Minimum Mean Cycle", "CC Cancel and Tighten"]


def get_data():
    features_data = pd.read_csv(os.path.join(path_to_parent_dir,path_to_data, "features.txt"), sep=" ", header=None, index_col=0,
                                names=names_features)

    # read output.txt
    output_data = pd.read_csv(os.path.join(path_to_parent_dir,path_to_time), sep=" ", header=None, index_col=0,
                              names=names_output, converters={0: lambda x: str(x.split("/")[-1])})
    # combine data according to seeds, and clean it up
    combined_data = features_data.join(output_data)
    combined_data = combined_data.dropna()
    combined_data = combined_data[combined_data["Network Simplex"] != "There"]
    for n in names_output:
        combined_data[n] = pd.to_numeric(combined_data[n])
    combined_data["Minimum"] = combined_data.loc[:, names_output].idxmin(axis=1)

    return combined_data


#combined_data = get_data()

    
"""
combined_data_infeasible = combined_data[combined_data["Network Simplex"] == "There"]
combined_data_feasible = combined_data[combined_data["Network Simplex"] != "There"]

fig = plt.figure()
ax = plt.axes()

ax.scatter(combined_data_infeasible["g_supply"], combined_data_infeasible["g_mean_cap"])
ax.set_xlabel("g_supply")
ax.set_ylabel("g_mean_cap")
ax.set_title("infeasible")

fig = plt.figure()
ax = plt.axes()

ax.scatter(combined_data_feasible["g_supply"], combined_data_feasible["g_sum_cap"])
ax.set_xlabel("g_supply")
ax.set_ylabel("g_sum_cap")
ax.set_title("feasible")

length_before = len(combined_data_infeasible)

to_delete = []
for data, i in zip(combined_data_infeasible.iloc(), range(length_before)):
    max_cap = data["g_max_cap"]
    supply = data["g_supply"]
    length = (data["g_number_of_nodes"] - 2) / (data["g_arcs_source_sink"] - 1)
    mean_cap = data["g_mean_cap"]
    index = int(combined_data_infeasible.index[i])
    if mean_cap >= 0.95:
        to_delete.append(index)
    if supply > length * max_cap:
        to_delete.append(index)
    if 900 <= max_cap and max_cap <= 1000 and supply > 23000:
        to_delete.append(index)

    if 90 <= max_cap and max_cap <= 100 and supply > 2500:
        to_delete.append(index)

    if 9 <= max_cap and max_cap <= 10 and supply > 350:
        to_delete.append(index)

    if max_cap <= 5 and supply > 15:
        to_delete.append(index)

x = combined_data_infeasible.drop(to_delete, axis="index")
length_after = len(x)

print(length_before, length_after)

fig = plt.figure()
ax = plt.axes()
ax.scatter(x["g_supply"], x["g_sum_cap"])
ax.set_xlabel("g_supply")
ax.set_ylabel("g_sum_cap")
ax.set_title("infeasible")

# def get_algorithm_from_seed(seed):
#     seed = seed.name
#     if 10000000 <= seed < 20000000:
#         return "NETGEN"
#     elif 20000000 <= seed < 30000000:
#         return "GRIDGEN"
#     elif 30000000 <= seed < 40000000:
#         return "GRIDGRAPH"
#     elif 40000000 <= seed < 50000000:
#         return "GO_TO"


# combined_data["Generator"] = combined_data.apply(get_algorithm_from_seed, axis=1)

# calculate minimum of each instance


colour = dict(zip(names_output, ["red", "green", "blue", "yellow", "black", "orange", "purple"]))
generators = ["GRIDGEN", "NETGEN", "GO_TO", "GRIDGRAPH"]


def plot_features(gen):
    data = combined_data[combined_data["Generator"] == gen]
    for param in names_features[1:]:
        plt.figure()
        for alg in names_output[0:4]:
            unique_values = sorted(data[param].unique())
            y = [data[alg][data[param] == u].mean() for u in unique_values]
            p = np.polyfit(unique_values, y, 10)
            y = np.polyval(p, unique_values)
            plt.plot(unique_values, y, c=colour[alg], label=alg)
        plt.yscale("log")
        plt.legend()
        plt.title("param: " + param + ", generator: " + gen)


def plot_alg(gen=""):
    if gen not in generators:
        data = combined_data
        gen = "All"
    else:
        data = combined_data[combined_data["Generator"] == gen]
    plt.figure()
    number_of_minimum = np.array([len(data[data["Minimum"] == alg]) for alg in names_output])


colour = dict(zip(names_output, ["red", "green", "blue", "yellow", "black", "orange", "purple"]))


# generator = "GRIDGEN"
# combined_data = combined_data[combined_data["Generator"] == generator]


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


def plot_alg():
    number_of_minimum = np.array([len(combined_data[combined_data["Minimum"] == alg]) for alg in names_output])
    indices_where_not_0 = [i for i in range(len(number_of_minimum)) if number_of_minimum[i] != 0]

    names_output_modified = [names_output[i] for i in range(len(names_output)) if i in indices_where_not_0]

    plt.pie(number_of_minimum[indices_where_not_0],
            colors=["red", "green", "blue", "yellow", "black", "orange", "purple"], labels=names_output_modified)

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
"""