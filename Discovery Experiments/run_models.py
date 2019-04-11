import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import experiment_types

import argparse
parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--exp_types', nargs='+', required=True)
parser.add_argument('--structured', dest='structured', action='store_true')
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument("--bsize", type=int, default=16)
parser.add_argument("--n_iters", type=int, default=10)
parser.add_argument('--do_test', dest='do_test', action='store_true')

args, extras = parser.parse_known_args()
args.extras = extras

def modify_config(config) :
    config['training_config']['common']['bsize'] = args.bsize
    return config

args.modify_config = modify_config

data = dataloaders.dataloaders[args.dataset](args)
for exp_type in args.exp_types :
    experiment_types[exp_type](data, args)