import os
import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run confidence experiments on a dataset')
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument('--do_test', dest='do_test', action='store_true')
parser.add_argument('--do_all', dest='do_all', action='store_true')

args, extras = parser.parse_known_args()
args.extras = extras

args.output_dir = os.path.join(args.output_dir, 'confidence')
args.n_iters = 8

configs = [LSTM, LSTM_with_attention, CNN, CNN_with_attention]

data = dataloaders.dataloaders[args.dataset](args)
args.structured = data.structured_dim > 0
basic_experiments(data, configs, args)