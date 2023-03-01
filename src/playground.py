import compare_generated_instances as data
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.model_selection import train_test_split, cross_val_score
from util import *

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

combined_data = data.get_data()

train_data, test_data, train_label, test_label, train_times, test_times = train_test_split(combined_data.iloc[:, :21],
                                                                                           combined_data["Minimum"],
                                                                                           combined_data.iloc[:, 21:-1],
                                                                                           test_size=0.2,
                                                                                           random_state=1)



algo_and_options = [
    (KNeighborsClassifier, [("n_neighbors", [8, 10, 20, 50, 70, 90]), ("weights", ["uniform", "distance"])]),
    (SVC, [("C", [0.5, 0.75, 1.0, 1.25, 1.5]), ("kernel", ["poly", "rbf", "sigmoid"])]),
    #(GaussianProcessClassifier, [("n_restarts_optimizer", [0, 1])]),
    (DecisionTreeClassifier, [("criterion", ["gini", "entropy"]), ("splitter", ["best", "random"]),
                              ("max_depth", [None, 3, 5, 8])]),
    (RandomForestClassifier, [("criterion", ["gini", "entropy"]), ("max_depth", [None, 3, 5, 8]),
                              ("n_estimators", [10, 50, 100, 200])]),
    (MLPClassifier,
     [("hidden_layer_sizes", [(100), (50, 50), (50, 100, 100, 50)]), ("activation", ["tanh", "relu"]),
      ("solver", ["sgd", "adam"]), ("learning_rate_init", [0.001, 0.01, 0.1])]),
    (AdaBoostClassifier,
     [("n_estimators", [5, 7, 9, 11, 13, 50]), ("learning_rate", [0.8, 0.85, 1., 1.15, 1.3])]),
    (GaussianNB, []),
    (QuadraticDiscriminantAnalysis, [("reg_param", [-1, -0.5, 0, 0.5, 1])])]

training_cv_runs = []
production_training_runs = []
for (algo, hyperparameter_options) in algo_and_options:
    # Do a simple grid search with cross validation to find usable hyperparameters
    best_args, best_args_score = None, None
    for args in simple_grid_search(hyperparameter_options, curr_selection={}):
        clf = algo(**args)
        cv_scores = cross_val_score(clf, train_data, y=train_label, n_jobs=-1)
        mean_score = np.mean(cv_scores)
        print("Finished", args, " with cv scores", cv_scores, " mean:", mean_score)
        training_cv_runs.append((algo.__name__, args, cv_scores, mean_score))
        if best_args_score is None or mean_score > best_args_score:
            best_args_score = mean_score
            best_args = args

    # After we determined usable hyperparameters, now do a training run on those
    # with all of the available training data.
    clf = algo(**args)
    clf.fit(train_data, train_label)

    # Determine the score of this classifier in terms of test accuracy and
    # test algorithm selection time difference
    y_pred_test = clf.predict(test_data)
    test_accuracy = sl.metrics.accuracy_score(test_label, y_pred_test)
    test_time = algorithm_selection_metric(y_pred_test, test_times)
    production_training_runs.append(
        {"name": algo.__name__, "test_accuracy": test_accuracy, "test_time": test_time, "hyperparameters": best_args})
    print("Algorithm ", algo.__name__, " has achieved test_acc:", test_accuracy, "and test_time:", test_time)
    print("The optimal hyperparameters were determined as", best_args)

baseline_test_acc = sl.metrics.accuracy_score(test_label, ["Network Simplex" for _ in range(len(test_data))])
baseline_test_time = algorithm_selection_metric(["Network Simplex" for _ in range(len(test_data))], test_times)
production_training_runs.append(
    {"name": "baseline", "test_accuracy": baseline_test_acc, "test_time": baseline_test_time})
print("Baseline(always predict NetworkSimplex) test_acc", baseline_test_acc, "and test_time:", baseline_test_time)

perfect_test_time = algorithm_selection_metric(test_label, test_times)
production_training_runs.append({"name": "GroundTruth", "test_accuracy": 1., "test_time": perfect_test_time})
print("Optimal time:", perfect_test_time)
