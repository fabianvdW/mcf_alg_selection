import os
import time
import argparse
import torch
import torch_geometric
from optimization import optimize
import pickle
from pathlib import Path
from gin import GIN, GINRes
from torch_geometric.loader import DataLoader
from optimization import Objective
from constants import *
from torch_in_memory_loader import MCFDatasetInMemory


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_parser():
    out = argparse.ArgumentParser()
    out.add_argument('-num_workers', default=0, type=int,
                     help='The number of workers used for training and evaluation.')
    out.add_argument("-dsroottrain", default=DATA_PATH, type=str, help="Root folder of train ds")
    out.add_argument("-dsroottest", default=DATA_PATH, type=str, help="Root folder of test ds")
    out.add_argument("-experiment_name", default="test", type=str, help="Name of the experiment")
    out.add_argument("-ntrain", default=48000, type=int, help="Number of training datapoints used")
    out.add_argument('-cuda', default=0, type=int, help='The cuda device used.')
    out.add_argument("-compile_model", default=False, type=str2bool)
    return out


def main(args):
    torch.set_float32_matmul_precision("high")
    if args.ntrain < 1000:
        args.ntrain = args.ntrain * 1000
    train_dataset = MCFDatasetInMemory(args.dsroottrain).shuffle()[:args.ntrain]
    test_dataset = MCFDatasetInMemory(args.dsroottest).shuffle()
    # Scale epochs according to ntrain
    epochs = int(25 * 48000 / args.ntrain)
    hyperparameters = {'batch_size': 24, 'epochs': epochs, 'lr': -2.9871179346013337, 'weight_decay': -10.0,
                       'step_size': 0.5809405735679614,
                       'hidden_channels': 73, 'num_gin_layers': 6, 'num_mlp_layers': 2, 'num_mlp_readout_layers': 3,
                       'skip_connections': False, 'loss': 'mix_expected_runtime', 'loss_weight': 0.6831435486104581}

    def save_checkpoint(log_info):
        with open(os.path.join(args.dsroottrain, 'result', args.experiment_name, 'log_info_posthpo_result.pkl'),
                  'wb') as f:
            pickle.dump(log_info, f)

    start = time.time()
    Path(os.path.join(args.dsroottrain, 'result', args.experiment_name)).mkdir(parents=True, exist_ok=True)
    objective = Objective(None, None, device, args.num_workers, args.compile_model, None, None)
    objective.model = GIN(
        device=device,
        in_channels=1,
        hidden_channels=hyperparameters['hidden_channels'],
        out_channels=NUM_CLASSES,
        num_gin_layers=hyperparameters['num_gin_layers'],
        num_mlp_layers=hyperparameters['num_mlp_layers'],
        num_mlp_readout_layers=hyperparameters['num_mlp_readout_layers'],
    ).to(device)
    if args.compile_model:
        objective.model = torch.compile(objective.model, dynamic=True, fullgraph=True)
    train_loader = DataLoader(train_dataset, batch_size=int(hyperparameters['batch_size']), shuffle=True,
                              drop_last=True, num_workers=int(args.num_workers))
    eval_loader = DataLoader(test_dataset, batch_size=int(hyperparameters['batch_size']),
                             num_workers=int(args.num_workers))
    epochs_info = objective.train_eval(train_loader, eval_loader, hyperparameters)
    objective_value = epochs_info[-1]['eval_obj']
    end = time.time()
    log_info = (hyperparameters, epochs_info, objective_value, end - start)
    save_checkpoint(log_info)
    print(end - start, 's')


if __name__ == "__main__":
    os.chdir("..")
    parser = setup_parser()

    _args = parser.parse_args()
    if _args.cuda > -1 and torch.cuda.is_available():
        device = _args.cuda
    else:
        device = "cpu"
    main(_args)
