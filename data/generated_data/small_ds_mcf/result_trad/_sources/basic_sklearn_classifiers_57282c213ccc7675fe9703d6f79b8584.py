from util import *
from data_loader_trad import get_data, DATA_PATH
import numpy as np
import os
import json

import sklearn as sl
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sacred import Experiment
from sacred.observers import file_storage

experiment_name = "result_trad"
exp, EXP_FOLDER = Experiment(experiment_name), os.path.join(DATA_PATH)


@exp.main
def main():
    combined_data = get_data()

    save_obj_to_exp(exp, "raw_data.csv", lambda path: combined_data.to_csv(path))

    # train_data, test_data, train_label, test_label, train_times, test_times = train_test_split(combined_data.iloc[:, :-8], combined_data["Minimum"], combined_data.iloc[:, -8:-1], test_size=0.2, random_state=1)
    train_data, train_label, train_times = combined_data.iloc[:, :-8], combined_data["Minimum"], combined_data.iloc[:,
                                                                                                 -8:-1]

    algo_and_options = [
        (KNeighborsClassifier, [("n_neighbors", [8, 10, 20, 50, 70, 90]), ("weights", ["uniform", "distance"])]),
        (SVC, [("C", [0.5, 0.75, 1.0, 1.25, 1.5]), ("kernel", ["poly", "rbf", "sigmoid"])]),
        # (GaussianProcessClassifier, [("n_restarts_optimizer", [0, 1])]),
        (
            DecisionTreeClassifier,
            [
                ("criterion", ["gini", "entropy"]),
                ("splitter", ["best", "random"]),
                ("max_depth", [None, 3, 5, 8]),
                ("class_weight", [None, "balanced"]),
            ],
        ),
        (
            RandomForestClassifier,
            [
                ("criterion", ["gini", "entropy"]),
                ("max_depth", [None, 3, 5, 8]),
                ("n_estimators", [10, 50, 100, 200]),
                ("class_weight", [None, "balanced"]),
            ],
        ),
        #(
        #    MLPClassifier,
        #    [
        #        ("hidden_layer_sizes", [(100), (50, 50), (50, 100, 100, 50)]),
        #        ("activation", ["tanh", "relu"]),
        #        ("solver", ["sgd", "adam"]),
        #        ("learning_rate_init", [0.001, 0.01, 0.1]),
        #    ],
        #),
        (AdaBoostClassifier, [("n_estimators", [5, 7, 9, 11, 13, 50]), ("learning_rate", [0.8, 0.85, 1.0, 1.15, 1.3]), ("algorithm", ["SAMME"])]),
        (GaussianNB, []),
        ("Baseline", []),
    ]

    training_cv_runs = []
    production_training_runs = []
    np.random.seed(1)
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    gen = kf.split(list(range(len(train_data))))
    train_eval_indices = [(train_indices, eval_indices) for train_indices, eval_indices in gen]
    for algo, hyperparameter_options in algo_and_options:
        algo_name = algo.__name__ if algo != "Baseline" else "Baseline"
        # Do a simple grid search with cross validation to find usable hyperparameters
        best_args, best_args_score = None, None
        for args in simple_grid_search(hyperparameter_options, curr_selection={}):
            eval_acc_scores = []
            eval_runtime_sums = []
            eval_minruntime_sums = []
            for train_indices, eval_indices in train_eval_indices:
                if algo == "Baseline":
                    y_pred = ["NS" for _ in range(len(eval_indices))]

                else:
                    clf = algo(**args)
                    clf.fit(train_data.iloc[train_indices], train_label.iloc[train_indices])
                    y_pred = clf.predict(train_data.iloc[eval_indices])

                eval_acc_scores.append(sl.metrics.accuracy_score(train_label.iloc[eval_indices], y_pred))
                eval_runtime_sums.append(algorithm_selection_metric(y_pred, train_times.iloc[eval_indices]))
                eval_minruntime_sums.append(
                    algorithm_selection_metric(train_label.iloc[eval_indices], train_times.iloc[eval_indices]))

            eval_ratios = [x / y for x, y in zip(eval_runtime_sums, eval_minruntime_sums)]

            mean_score = np.mean(eval_ratios) - np.mean(eval_acc_scores)
            print("Finished", algo_name,args, " with acc scores", eval_acc_scores, " ratios", eval_ratios, " final Obj. ",
                  mean_score)

            training_cv_runs.append(
                (algo_name, args, eval_acc_scores, eval_runtime_sums, eval_minruntime_sums, eval_ratios, mean_score))
            if best_args_score is None or mean_score < best_args_score:
                best_args_score = mean_score
                best_args = args
        """
        # After we determined usable hyperparameters, now do a training run on those
        # with all of the available training data.
        if algo == "Baseline":
            y_pred_test = ["NS" for _ in range(len(test_label))]
        else:
            clf = algo(**best_args)
            clf.fit(train_data, train_label)

            # Determine the score of this classifier in terms of test accuracy and
            # test algorithm selection time difference
            y_pred_test = clf.predict(test_data)
        test_accuracy = sl.metrics.accuracy_score(test_label, y_pred_test)
        test_runtime_sum = algorithm_selection_metric(y_pred_test, test_times)
        test_minruntime_sum = algorithm_selection_metric(test_label, test_times)
        test_ratio = test_runtime_sum / test_minruntime_sum
        production_training_runs.append(
            {"name":algo_name, "test_accuracy": test_accuracy, "test_runtime_sum": test_runtime_sum, "test_minruntime_sum": test_minruntime_sum, "test_ratio": test_ratio,
             "hyperparameters": best_args}
        )
        print("Algorithm ", algo_name, " has achieved test_acc:", test_accuracy, "and test_time:", test_time)
        print("The optimal hyperparameters were determined as", best_args)
        """

    # Save results to experiment
    save_str_to_exp(exp, json.dumps(training_cv_runs, indent=1), "training_cv_runs.json")
    save_str_to_exp(exp, json.dumps(production_training_runs, indent=1), "production_training_runs.json")


if __name__ == "__main__":
    log_location = os.path.join(EXP_FOLDER, experiment_name)
    exp.observers.append(file_storage.FileStorageObserver(log_location))
    exp.run()
