import os
import argparse
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from constants import *

nice_fonts = {
    # Use LaTeX to write all text
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 18,
    "font.size": 18,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    'figure.autolayout': True,
    'figure.dpi': 180
}


def setup_parser():
    out = argparse.ArgumentParser()
    out.add_argument("-dsroot", default=DATA_PATH, type=str, help="Root folder of ds")
    return out


result_folder = os.path.join("result", "skip_f_loss_mix_post_hpo")


def post_hpo(args):
    info_results = []
    for i in range(1, 49):
        if i == 38:
            continue
        with open(os.path.join(args.dsroot, "result", "skip_f_loss_mix_post_hpo" + (f"_{i}" if i < 48 else ""),
                               f"log_info_post{'_' if i==48 else ''}hpo" + ("_result.pkl" if i < 48 else ".pkl")), "rb") as f:
            info_results.append(pickle.load(f))
    accuracies = []
    ratios = []
    for _, epochs_info, _, _ in info_results:
        accuracies.append(epochs_info[-1]['eval_total_acc'])
        ratios.append(epochs_info[-1]['eval_runtime_sum'] / epochs_info[-1]['eval_minruntime_sum'])

    plt.style.use('seaborn-v0_8-paper')
    matplotlib.rcParams.update(nice_fonts)

    # First plot: Accuracies
    fig1 = plt.figure(figsize=(10, 10))
    plt.scatter(range(1, len(accuracies)+1), accuracies, marker='o', s=100)
    plt.xlabel('Number of training data points (in 1,000)')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)
    plt.grid(True)

    fig1.savefig(os.path.join(args.dsroot, "result", "skip_f_loss_mix_post_hpo",
                              f"sensitivity_acc.png"))
    plt.close(fig1)

    # Second plot: Ratios
    fig2 = plt.figure(figsize=(10, 10))
    plt.scatter(range(1, len(ratios)+1), ratios, marker='o', color='orange', s=100)
    plt.xlabel('Number of training data points (in 1,000)')
    plt.ylabel('Runtime Ratio')
    plt.ylim(1.0, 1.04)
    plt.grid(True)

    fig2.savefig(os.path.join(args.dsroot, "result", "skip_f_loss_mix_post_hpo",
                              f"sensitivity_ratio.png"))
    plt.close(fig2)

    print(accuracies)
    print(ratios)

def main(args):
    if os.path.exists(os.path.join(args.dsroot, result_folder, "log_info_post_hpo.pkl")):
        with open(os.path.join(args.dsroot, result_folder, "log_info_post_hpo.pkl"), "rb") as f:
            log_info_result = pickle.load(f)
        a, b, c, d = log_info_result
        print(a)
        for x in b:
            print(x)
        print(c)
        print(d)
        assert False
    with open(os.path.join(args.dsroot, result_folder, 'skopt_result.pkl'), "rb") as f:
        skopt_result = pickle.load(f)
    with open(os.path.join(args.dsroot, result_folder, 'log_info_result.pkl'), "rb") as f:
        log_info_result = pickle.load(f)
    optimal_index = np.argmax([configuration[0] == skopt_result for configuration in log_info_result])
    print("Best parameters are given by the following configuration, achieved in bayes sample ", optimal_index)
    print("Achieved in bayes sample ", -np.mean(log_info_result[optimal_index][2]))
    print(skopt_result)
    try:
        os.mkdir(os.path.join(args.dsroot, result_folder, "plots"))
    except FileExistsError:
        pass
    plt.style.use('seaborn-v0_8-paper')
    matplotlib.rcParams.update(nice_fonts)


    def key_formatter(key):
        if key == "weight_decay":
            return "decay"
        elif key == "hidden_channels":
            return "channels"
        elif key == "num_gin_layers":
            return "gin_layers"
        elif key == "num_mlp_layers":
            return "mlp_layers"
        elif key == "num_mlp_readout_layers":
            return "readout_layers"
        elif key == "skip_connections":
            return "ResNet"
        return key

    def value_formatter(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return value

    for i, configuration in enumerate(log_info_result):
        parameters, cv_epoch_info, obj_values, runtime = configuration

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 10))
        final_obj = -np.mean(obj_values)
        subtitle = ", ".join(
            [f"{key_formatter(key)}={value_formatter(value)}" for key, value in parameters.items()] + [
                f"Final obj.={final_obj:.4f}",
                f"Runtime={runtime:.1f}s"])
        # fig.suptitle(f'Visualized metrics of bayes sample {i}\n', fontsize=14)
        # fig.text(s=subtitle, x=0.5, y=0.95, fontsize=6, ha='center', va='center')
        ax1.set_xlabel("Epochs")
        ax1.set_ylim([0.7, 1.0])
        ax2.set_xlabel("Epochs")
        ax2.set_ylim([1.0, 1.3])
        ax3.set_xlabel("Epochs")
        ax3.set_ylim([0.0, 1.0])

        cv_runs = len(cv_epoch_info)
        assert cv_runs == 5
        ###
        # Plot Accuracy/Runtimes ratios of train and evaluation runs on ax1
        # Plot Losses of train and evaluation runs on ax2.
        # We use the same color for all training lines and all evaluation lines, but iterate over the markers
        marker_types = ['o', 'v', '^', '<', '>']
        color_cycle = iter(plt.rcParams['axes.prop_cycle'])
        train_color = next(color_cycle)['color']
        evaluation_color = next(color_cycle)['color']
        train_color2 = next(color_cycle)['color']
        evaluation_color2 = next(color_cycle)['color']
        train_color3 = next(color_cycle)['color']
        evaluation_color3 = next(color_cycle)['color']

        for cv_run in range(cv_runs):
            epochs_info = cv_epoch_info[cv_run]
            train_accs, eval_accs = [info['train_total_acc'] for info in epochs_info], [info['eval_total_acc'] for info
                                                                                        in epochs_info]
            train_ratios = [info['train_runtime_sum'] / info['train_minruntime_sum'] for info in epochs_info]
            eval_ratios = [info['eval_runtime_sum'] / info['eval_minruntime_sum'] for info in epochs_info]
            train_losses, eval_losses = [info['train_total_loss'] for info in epochs_info], [info['eval_total_loss'] for
                                                                                             info in epochs_info]

            ax1.plot(train_accs, linestyle='--', label='Train accuracy' if cv_run == 0 else None, color=train_color,
                     alpha=0.5)
            ax1.plot(train_accs, linestyle='', markeredgecolor='none', marker=marker_types[cv_run], color=train_color,
                     alpha=0.5)

            ax1.plot(eval_accs, linestyle='--', label='Evaluation accuracy' if cv_run == 0 else None,
                     color=evaluation_color, alpha=0.5)
            ax1.plot(eval_accs, linestyle='', markeredgecolor='none', marker=marker_types[cv_run],
                     color=evaluation_color, alpha=0.5)

            ax2.plot(train_ratios, linestyle='--', label='Train runtime ratios' if cv_run == 0 else None,
                     color=train_color2, alpha=0.5)
            ax2.plot(train_ratios, linestyle='', markeredgecolor='none', marker=marker_types[cv_run],
                     color=train_color2, alpha=0.5)

            ax2.plot(eval_ratios, linestyle='--', label='Evaluation runtime ratios' if cv_run == 0 else None,
                     color=evaluation_color2,
                     alpha=0.5)
            ax2.plot(eval_ratios, linestyle='', markeredgecolor='none', marker=marker_types[cv_run],
                     color=evaluation_color2, alpha=0.5)

            ax3.plot(train_losses, linestyle='--', label='Train loss' if cv_run == 0 else None, color=train_color3,
                     alpha=0.5)
            ax3.plot(train_losses, linestyle='', markeredgecolor='none', marker=marker_types[cv_run],
                     color=train_color3, alpha=0.5)

            ax3.plot(eval_losses, linestyle='--', label='Evaluation loss' if cv_run == 0 else None,
                     color=evaluation_color3, alpha=0.5)
            ax3.plot(eval_losses, linestyle='', markeredgecolor='none', marker=marker_types[cv_run],
                     color=evaluation_color3, alpha=0.5)

            ax1.legend()
            ax2.legend()
            ax3.legend()
        fig.savefig(os.path.join(args.dsroot, result_folder, "plots",
                                 f"bayes_{i},bs={parameters['batch_size']},hc={parameters['hidden_channels']}.png"))
        plt.close(fig)


if __name__ == "__main__":
    os.chdir("..")
    parser = setup_parser()
    _args = parser.parse_args()
    #main(_args)
    post_hpo(_args)
