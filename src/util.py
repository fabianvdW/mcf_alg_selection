import os, json

TIME_LIMIT = 4 * 10 ** 6  # 4s

path_to_project = os.path.join("..")
path_to_data = os.path.join(path_to_project, "data", "generated_data")


def algorithm_selection_metric(algorithm_list, df_section):
    # Maybe one can vectorize this(cf. https://stackoverflow.com/questions/24833130/how-can-i-select-a-specific-column-from-each-row-in-a-pandas-dataframe)
    return int(sum(df_section.iloc[i][algo] for i, algo in enumerate(algorithm_list)))


def algorithm_selection_scorer(estimator, X, times):
    return algorithm_selection_metric(estimator.predict(X), times)


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
