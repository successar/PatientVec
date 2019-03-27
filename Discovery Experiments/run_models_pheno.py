import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import experiment_types

import argparse
parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument("--label_field", type=str, required=True)
parser.add_argument("--bsize", type=int, default=16)

args, extras = parser.parse_known_args()
args.extras = extras
args.structured = False

data = dataloaders.dataloaders[args.dataset](args)
experiment_types['lr'](data, args)
experiment_types['lda'](data, args)
experiment_types['vanilla'](data, args)
experiment_types['attention'](data, args)
experiment_types['basic_sru'](data, args)
experiment_types['hierarchical'](data, args)
