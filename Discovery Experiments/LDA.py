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

print(args)
data = dataloaders.dataloaders[args.dataset](args)
train_data, dev_data = get_basic_data(data, structured=True, truncate=90)
config = {'vocab' : data.vocab, 
          'stop_words' : True, 
          'exp_name' : data.name, 
          'type' : data.metrics_type, 
          'norm' : None, 
          'constant_mul' : 1.0, 
          'has_structured' : True, 
          'lda' : True, 
          'basepath' : args.output_dir}

lr = LR(config)
lr.train_lda(train_data)
lr.evaluate_lda(dev_data, save_results=True)