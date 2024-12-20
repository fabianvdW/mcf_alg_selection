import os
import time
import argparse
import torch
import torch_geometric
from optimization import optimize
import pickle
from pathlib import Path

from skopt.utils import use_named_args
from skopt.space import Real, Integer, Categorical
from constants import *
from torch_in_memory_loader import MCFDatasetInMemory


# TODO: LaPlace eigenvectors, weird cause not invariant and some randomness
# TODO: Alternative random walk kernels (This is a sort of feature preprocessing)
# https://github.com/G-Taxonomy-Workgroup/GPSE
# https://github.com/G-Taxonomy-Workgroup/GPSE/blob/6cafeb05dd20b3590f4fe588d17528d44044a21c/graphgym/transform/posenc_stats.py#L138

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
    out.add_argument('-num_bayes_samples', default=50, type=int,
                     help='The number of samples used to estimate the optimal hyperparameters.')
    out.add_argument('-seeds', default=[42], type=int, nargs='+',
                     help='The seeds used, one after the other, to determine the splits as well as the baysian'
                          'sampling.')
    out.add_argument("-dsroot", default=DATA_PATH, type=str, help="Root folder of ds")
    out.add_argument("-experiment_name", type=str, help="Name of the experiment")
    out.add_argument('-cuda', default=0, type=int, help='The cuda device used.')
    out.add_argument('-batch_size', default=[24, 128], type=int, nargs=2, help='The minimum and maximum batch size.')
    out.add_argument('-epochs', default=[4, 30], type=int, nargs=2,
                     help='The minimum and maximum amount of epochs.')
    out.add_argument('-lr', default=[-4.0, -1.5], type=float, nargs=2,
                     help='The minimum and maximum logarithmic learning '
                          'rate, i.e. 10**lr is used for training.')
    out.add_argument('-weight_decay', default=[-10.0, -0.3], type=float, nargs=2,
                     help='The minimum and maximum logarithmic weight decay.')
    out.add_argument('-hidden_channels', default=[16, 80], type=int, nargs=2,
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
    out.add_argument("-compile_model", default=False, type=str2bool)
    out.add_argument("-skip", type=str2bool, help='Residual connections on/off')
    out.add_argument("-loss_ce", type=str2bool, help='Loss function is CE(true) or mix (false)')
    return out


def get_space(name, tuple):
    assert len(tuple) == 2
    if isinstance(tuple[0], int):
        return Integer(name=name, low=tuple[0], high=tuple[1])
    elif isinstance(tuple[0], float):
        return Real(name=name, low=tuple[0], high=tuple[1])
    assert False


def checkpoint_log_info_functions(args):
    def save_checkpoint(log_info):
        with open(os.path.join(args.dsroot, 'result', args.experiment_name, 'log_info_result.pkl'), 'wb') as f:
            pickle.dump(log_info, f)

    def load_checkpoint():
        try:
            with open(os.path.join(args.dsroot, 'result', args.experiment_name, 'log_info_result.pkl'), 'rb') as f:
                log_info = pickle.load(f)
                return log_info
        except:
            return []

    return save_checkpoint, load_checkpoint


def main(args, seed):
    torch_geometric.seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    dataset = MCFDatasetInMemory(args.dsroot).shuffle()
    search_space = [get_space(name='batch_size', tuple=args.batch_size),
                    get_space(name='epochs', tuple=args.epochs),
                    get_space(name='lr', tuple=args.lr),
                    get_space(name='weight_decay', tuple=args.weight_decay),
                    get_space(name='step_size', tuple=args.step_size),
                    get_space(name="hidden_channels", tuple=args.hidden_channels),
                    get_space(name="num_gin_layers", tuple=args.num_gin_layers),
                    get_space(name="num_mlp_layers", tuple=args.num_mlp_layers),
                    get_space(name="num_mlp_readout_layers", tuple=args.num_mlp_readout_layers),
                    Categorical([args.skip], name="skip_connections"),  # Part of evaluation, i.e. make this constant,
                    Categorical([
                        "cross_entropy" if args.loss_ce else "mix_expected_runtime",
                        # "expected_runtime"
                        # "mix_expected_runtime"
                    ], name="loss"),  # Part of evaluation, i.e. make this constant
                    # get_space(name="loss_weight", tuple=(0., 1.))
                    ]
    if not args.loss_ce:
        search_space.append(get_space(name="loss_weight", tuple=(0., 1.)))

    start = time.time()
    Path(os.path.join(args.dsroot, 'result', args.experiment_name)).mkdir(parents=True, exist_ok=True)
    checkpoint_file_skopt = os.path.join(args.dsroot, 'result', args.experiment_name, 'checkpoint_skopt.pkl')
    result = optimize(dataset=dataset, device=device, search_space=search_space,
                      num_bayes_samples=args.num_bayes_samples, num_workers=args.num_workers, seed=seed,
                      compile_model=args.compile_model, checkpoint_functions_log=checkpoint_log_info_functions(args),
                      checkpoint_file_skopt=checkpoint_file_skopt)
    end = time.time()
    print(end - start, 's')
    print(result)

    @use_named_args(dimensions=search_space)
    def to_kwargs(**kwargs):
        return kwargs

    with open(os.path.join(args.dsroot, 'result', args.experiment_name, 'skopt_result.pkl'), 'wb') as f:
        pickle.dump(to_kwargs(result.x), f)


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
