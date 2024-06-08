import os
import json

FEATURE_NAMES = [
    "id",
    "g_number_of_nodes",
    "g_number_of_arcs",
    "g_min_cost",
    "g_max_cost",
    "g_sum_cost",
    "g_mean_cost",
    "g_std_cost",
    "g_min_cap",
    "g_max_cap",
    "g_sum_cap",
    "g_mean_cap",
    "g_std_cap",
    "g_supply",
    "g_number_of_supply_nodes",
    "g_arcs_source_sink",
    "mst_sum_cost",
    "mst_mean_cost",
    "mst_std_cost",
    "mst_sum_cap",
    "mst_mean_cap",
    "mst_std_cap",
]
ALGO_NAMES = ["NS", "CS2", "SSP", "CAS"]


def algorithm_selection_metric(algorithm_list, df_section):
    # Maybe one can vectorize this(cf. https://stackoverflow.com/questions/24833130/how-can-i-select-a-specific-column-from-each-row-in-a-pandas-dataframe)
    return int(sum(df_section.iloc[i][algo] for i, algo in enumerate(algorithm_list)))


def simple_grid_search(options, curr_selection):
    if not options:
        yield curr_selection.copy()
    else:
        option_name, option_values = options[0]
        for value in option_values:
            curr_selection[option_name] = value
            yield from simple_grid_search(options[1:], curr_selection=curr_selection)


def save_obj_to_exp(exp, path, saving_method):
    saving_method(path)
    if os.path.exists(path):
        exp.add_artifact(path)
        os.remove(path)


def save_str_to_exp(exp, content, filename):
    def saving_method(path):
        with open(path, "w") as out:
            out.write(content)

    save_obj_to_exp(exp, filename, saving_method)


def load_dict(filename):
    with open(filename, "r") as in_file:
        return json.loads(in_file.read())
