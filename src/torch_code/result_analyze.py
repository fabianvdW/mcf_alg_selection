import os
import argparse
import pickle
import matplotlib.pyplot as plt

from constants import *

def setup_parser():
    out = argparse.ArgumentParser()
    out.add_argument("-dsroot", default=DATA_PATH, type=str, help="Root folder of ds")
    return out

result_folder="result_skip=False_loss=ce"
def main(args):
    with open(os.path.join(args.dsroot, result_folder, 'skopt_result.pkl'), "rb") as f:
        skopt_result = pickle.load(f)
    with open(os.path.join(args.dsroot,  result_folder, 'log_info_result.pkl'), "rb") as f:
        log_info_result = pickle.load(f)
    print(skopt_result)
    try:
        os.mkdir(os.path.join(args.dsroot, result_folder, "plots"))
    except FileExistsError:
        pass
    for i, configuration in enumerate(log_info_result):
        parameters, cv_epoch_info, obj_values = configuration

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        cv_runs = len(cv_epoch_info)
        assert cv_runs == 5
        for cv_run in range(cv_runs):
            epochs_info = cv_epoch_info[cv_run]
            train_losses, eval_losses = [info['train_total_loss'] for info in epochs_info], [info['eval_total_loss'] for info in epochs_info]
        fig.savefig(os.path.join(args.dsroot, result_folder, "plots", f"bayes_{i},bs={parameters['batch_size']},hc={parameters['hidden_channels']}.png"))

        assert False


if __name__ == "__main__":
    os.chdir("..")
    parser = setup_parser()
    _args = parser.parse_args()
    main(_args)
