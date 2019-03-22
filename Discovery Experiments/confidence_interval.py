import os
import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run confidence experiments on a dataset')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)

args, extras = parser.parse_known_args()
args.extras = extras

args.structured = True
args.output_dir = os.path.join(args.output_dir, 'confidence')
args.n_iters = 8

configs = [LSTM, LSTM_with_attention]

data = dataloaders.dataloaders['readmission'](args)
basic_experiments(data, configs, args)

del data

data = dataloaders.dataloaders['diagnosis'](args)
basic_experiments(data, configs, args)