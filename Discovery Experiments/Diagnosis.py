from PatientVec.dataloaders import diagnosis_dataset
from PatientVec.Experiments.training_exps import basic_experiments, hierarchical_experiments, structured_attention_experiments

import argparse
parser = argparse.ArgumentParser(description='Run Diagnosis experiments')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument("--mock", dest='mock', action='store_true')

args, extras = parser.parse_known_args()

data = diagnosis_dataset(args)
basic_experiments(data, args)
hierarchical_experiments(data, args)
structured_attention_experiments(data, args)