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

args, extras = parser.parse_known_args()
args.extras = extras

data = dataloaders.dataloaders[args.dataset](args)
train_data, dev_data, test_data = get_basic_data(data, structured=args.structured, truncate=90)

for norm in [None, 'l1', 'l2'] :
    config = {'vocab' : data.vocab, 
              'stop_words' : True, 
              'exp_name' : data.name, 
                'type' : data.metrics_type, 
                'norm' : norm, 
                'constant_mul' : 1.0, 
                'has_structured' : args.structured, 
                'lda' : False, 
                'methods' : ['count', 'binary', 'tfidf'],
                'only_structured' : args.structured,
                'basepath' : args.output_dir,
                'penalty' : args.penalty}

    print(config)
    lr = LR(config)
    lr.train(train_data)
    lr.evaluate(name='dev', data=dev_data, save_results=True)
    lr.evaluate(name='test', data=test_data, save_results=True)