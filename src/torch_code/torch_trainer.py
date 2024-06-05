import os
import time
import argparse
import torch
import torch_geometric
from optimization import optimize

from skopt.space import Real, Integer, Categorical
from constants import *
from torch_in_memory_loader import MCFDataset

def setup_parser():
    out = argparse.ArgumentParser()
    out.add_argument('-num_workers', default=1, type=int,
                     help='The number of workers used for training and evaluation.')
    out.add_argument('-num_bayes_samples', default=100, type=int,
                     help='The number of samples used to estimate the optimal hyperparameters.')
    out.add_argument('-seeds', default=[42], type=int, nargs='+',
                     help='The seeds used, one after the other, to determine the splits as well as the baysian'
                          'sampling.')
    out.add_argument('-cuda', default=0, type=int, help='The cuda device used.')
    out.add_argument('-batch_size', default=[8, 64], type=int, nargs=2, help='The minimum and maximum batch size.')
    out.add_argument('-epochs', default=[10, 100], type=int, nargs=2,
                     help='The minimum and maximum amount of epochs.')
    out.add_argument('-lr', default=[-5.0, -2.0], type=float, nargs=2,
                     help='The minimum and maximum logarithmic learning '
                          'rate, i.e. 10**lr is used for training.')
    out.add_argument('-weight_decay', default=[-10.0, -0.3], type=float, nargs=2,
                     help='The minimum and maximum logarithmic weight decay.')
    out.add_argument('-hidden_channels', default=[16, 100], type=int, nargs=2,
                     help='The minimum and maximum number of hidden_channels used in the GNN.')
    out.add_argument('-num_gin_layers', default=[2, 6], type=int, nargs=2,
                     help='The minimum and maximum number of layers used in the GNN.')
    out.add_argument('-num_mlp_layers', default=[0, 3], type=int, nargs=2,
                     help='The minimum and maximum number of layers used for the MLP used in the GNN.')
    out.add_argument('-num_mlp_readout_layers', default=[1, 3], type=int, nargs=2,
                     help='The minimum and maximum number of layers used for the MLP used in the GNN.')
    out.add_argument('-step_size', default=[0.01, 1.0], type=float, nargs=2,
                     help='The minimum and maximum step_size used for the learning rate to drop relative to the number '
                          'of epochs.')
    return out


def get_space(name, tuple):
    assert len(tuple) == 2
    if isinstance(tuple[0], int):
        return Integer(name=name, low=tuple[0], high=tuple[1])
    elif isinstance(tuple[0], float):
        return Real(name=name, low=tuple[0], high=tuple[1])
    assert False

# TODO: Elwetritsch
def main(args, seed):
    torch_geometric.seed_everything(seed)
    dataset = MCFDataset(DATA_PATH).to(device).shuffle()
    search_space = [get_space(name='batch_size', tuple=args.batch_size),
                    get_space(name='epochs', tuple=args.epochs),
                    get_space(name='lr', tuple=args.lr),
                    get_space(name='weight_decay', tuple=args.weight_decay),
                    get_space(name='step_size', tuple=args.step_size),
                    get_space(name="hidden_channels", tuple=args.hidden_channels),
                    get_space(name="num_gin_layers", tuple=args.num_gin_layers),
                    get_space(name="num_mlp_layers", tuple=args.num_gin_layers),
                    get_space(name="num_mlp_readout_layers", tuple=args.num_gin_layers),
                    Categorical([True, False], name="skip_connections"), #Part of evaluation, i.e. make this constant,
                    Categorical([
                        "cross_entropy",
                        #"expected_runtime"
                    ], name="loss") # Part of evaluation, i.e. make this constant
                    ]
    start = time.time()
    result = optimize(dataset=dataset, device=device, search_space=search_space, num_bayes_samples=args.num_bayes_samples, num_workers=args.num_workers , seed=seed)
    end = time.time()
    print(end-start, 's')
    print(result)


if __name__ == "__main__":
    os.chdir("..")
    parser = setup_parser()

    _args = parser.parse_args()
    if _args.cuda > -1 and torch.cuda.is_available():
        device = _args.cuda
    else:
        device = "cpu"
    for seed in _args.seeds:
        main(_args, seed)
