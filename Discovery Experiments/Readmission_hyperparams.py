from PatientVec.dataloaders import readmission_dataset
from PatientVec.Experiments.hyperparam_exps import experiment_types

import argparse
parser = argparse.ArgumentParser(description='Run Readmission Hyperparams experiments')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--reg', type=float, required=True)
parser.add_argument('--exps', type=str, required=True)

from PatientVec.Experiments.configs import modify_training_params

args = parser.parse_args()
def modify_config(c) :
    c = modify_training_params(c, 'Adam', lr=args.lr, weight_decay=args.reg)
    return c

args.modify_config = modify_config

data = readmission_dataset(args)
experiment = experiment_types[args.exps]
experiment(data, args)