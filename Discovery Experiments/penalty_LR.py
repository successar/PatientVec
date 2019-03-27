import os
import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run confidence experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument('--structured', dest='structured', action='store_true')
parser.add_argument('--penalty', type=float, required=True)
parser.add_argument('--type', type=str, required=True)

args, extras = parser.parse_known_args()
args.extras = extras

args.n_iters = 8

data = dataloaders.dataloaders[args.dataset](args)
train_data, dev_data = get_basic_data(data, structured=args.structured, truncate=90)

config = {'vocab' : data.vocab, 
              'stop_words' : True, 
              'exp_name' : data.name, 
              'type' : data.metrics_type, 
              'norm' : 'l2', 
              'constant_mul' : 1.0, 
              'has_structured' : args.structured, 
              'lda' : False, 
              'methods' : [args.type],
              'only_structured' : args.structured,
              'basepath' : args.output_dir,
              'penalty' : args.penalty}

print(config)
lr = LR(config)
lr.train(train_data)
lr.evaluate(dev_data, save_results=True)

config = {'vocab' : data.vocab, 
              'stop_words' : True, 
              'exp_name' : data.name, 
              'type' : data.metrics_type, 
              'norm' : None, 
              'constant_mul' : 1.0, 
              'has_structured' : args.structured, 
              'lda' : False, 
              'methods' : [args.type],
              'only_structured' : args.structured,
              'basepath' : args.output_dir,
              'penalty' : args.penalty}

print(config)
lr = LR(config)
lr.train(train_data)
lr.evaluate(dev_data, save_results=True)