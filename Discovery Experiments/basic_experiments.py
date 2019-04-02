import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)

args, extras = parser.parse_known_args()
args.extras = extras
args.structured = True

data = dataloaders.dataloaders[args.dataset](args)
configs = [LSTM]
experiment_types['basic'](data, configs, args)