from PatientVec.dataloaders import readmission_dataset
from PatientVec.Experiments.training_exps import basic_experiments, hierarchical_experiments, structured_attention_experiments

import argparse
parser = argparse.ArgumentParser(description='Run Diagnosis experiments')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument("--mock", dest='mock', action='store_true')
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--reg', type=float, required=True)

from PatientVec.configs import modify_training_params

args = parser.parse_args()
def modify_config(c) :
    c = modify_training_params(c, 'Adam', lr=args.lr, weight_decay=args.reg)
    return c

args.modify_config = modify_config

data = readmission_dataset(args)
basic_experiments(data, args)
# hierarchical_experiments(data, args)
structured_attention_experiments(data, args)