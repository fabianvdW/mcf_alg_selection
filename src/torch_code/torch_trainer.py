import time
import argparse
import torch
import copy
import numpy as np
import random
from optimization import optimize
from sklearn.model_selection import StratifiedKFold, KFold
from skopt.space import Real, Integer, Categorical
from constants import *
from torch_in_memory_loader import MCFDataset

dataset = MCFDataset(DATA_PATH).shuffle()
BATCH_SIZE = 5


def setup_parser():
    out = argparse.ArgumentParser()
    out.add_argument('-num_workers', default=1, type=int,
                     help='The number of workers used for training and evaluation.')
    out.add_argument('-num_bayes_samples', default=50, type=int,
                     help='The number of samples used to estimate the optimal hyperparameters.')
    out.add_argument('-seeds', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], type=int, nargs='+',
                     help='The seeds used, one after the other, to determine the splits as well as the baysian'
                          'sampling.')
    out.add_argument('-cuda', default=-1, type=int, help='The cuda device used.')
    out.add_argument('-batch_size', default=[8, 256], type=int, nargs=2, help='The minimum and maximum batch size.')
    out.add_argument('-epochs', default=[32, 512], type=int, nargs=2,
                     help='The minimum and maximum amount of epochs.')
    out.add_argument('-lr', default=[-6.0, -2.0], type=float, nargs=2,
                     help='The minimum and maximum logarithmic learning '
                          'rate, i.e. 10**lr is used for training.')
    out.add_argument('-weight_decay', default=[-10.0, -0.3], type=float, nargs=2,
                     help='The minimum and maximum logarithmic weight decay.')
    out.add_argument('-hidden_channels', default=[16, 128], type=int, nargs=2,
                     help='The minimum and maximum number of hidden_channels used in the GNN.')
    out.add_argument('-layers', default=[2, 10], type=int, nargs=2,
                     help='The minimum and maximum number of layers used in the GNN.')
    out.add_argument('-step_size', default=[0.01, 1.0], type=float, nargs=2,
                     help='The minimum and maximum step_size used for the learning rate to drop relative to the number '
                          'of epochs.')
    return out


def get_space(name, tuple):
    if len(tuple) == 2:
        if tuple[0] == tuple[1]:
            return Categorical(name=name, categories=[tuple[0]])
        else:
            if isinstance(tuple[0], int):
                return Integer(name=name, low=tuple[0], high=tuple[1])
            elif isinstance(tuple[0], float):
                return Real(name=name, low=tuple[0], high=tuple[1])
    return Categorical(name=name, categories=tuple)


def main(args, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    search_space = [get_space(name='batch_size', tuple=args.batch_size),
                    get_space(name='epochs', tuple=args.epochs),
                    get_space(name='lr', tuple=args.lr),
                    get_space(name='weight_decay', tuple=args.weight_decay),
                    get_space(name='step_size', tuple=args.step_size),
                    ]
    start = time.time()
        else:
            kf = KFold(n_splits=10, random_state=seed, shuffle=True)
            gen = kf.split(list(range(len(datasets.choices[args.dataset][4]()))))
            train_eval = []
            for _ in range(args.num_train_eval_samples):
                train, val = next(gen)
                train = train.astype(np.int64)
                val = val.astype(np.int64)
                train_eval += [(train, val)]
            test = np.array(range(len(datasets.choices[args.dataset][5]())), dtype=np.int64)
            trainset = datasets.choices[args.dataset][4]
            evalset = copy.deepcopy(datasets.choices[args.dataset][4])
            testset = datasets.choices[args.dataset][5]
    result = optimize(train_set=train_set, eval_set=eval_set, device=device, search_space=search_space, num_workers=args.num_workers, num_bayes_samples=args.num_bayes_samples, seed=seed)

    print(result)


if __name__ == "__main__":
    parser = setup_parser()

    _args = parser.parse_args()
    if _args.cuda > -1 and torch.cuda.is_available():
        device = _args.cuda
    else:
        device = "cpu"
    for seed in _args.seeds:
        with torch.cuda.device(device):
            main(_args, seed)
