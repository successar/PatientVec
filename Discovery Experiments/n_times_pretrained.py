import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run TS Experiments on a dataset')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument('--n', type=int, required=True)
parser.add_argument("--output_dir", type=str)

args, extras = parser.parse_known_args()
args.extras = extras

args.structured = True
args.n_iters = 8
args.seed = 16348

data = dataloaders.dataloaders['readmission'](args)
experiment_types['pretrained'](data, args)

del data

data = dataloaders.dataloaders['mortality_30day'](args)
experiment_types['pretrained'](data, args)