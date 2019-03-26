import os
import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run confidence experiments on a dataset')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument('--structured', dest='display', action='store_true')

args, extras = parser.parse_known_args()
args.extras = extras

args.output_dir = os.path.join(args.output_dir, 'confidence')
args.n_iters = 8

data = dataloaders.dataloaders['hip_1yr'](args)
train_data, dev_data = get_basic_data(data, structured=args.structured, truncate=90)

for i in range(10) :
    config = {'vocab' : data.vocab, 
              'stop_words' : True, 
              'exp_name' : data.name, 
              'type' : data.metrics_type, 
              'norm' : 'l2', 
              'constant_mul' : 1.0, 
              'has_structured' : args.structured, 
              'lda' : False, 
              'methods' : ['count'],
              'only_structured' : args.structured,
              'basepath' : args.output_dir}

    lr = LR(config)
    lr.train(train_data)
    lr.evaluate(dev_data, save_results=True)

del data

for i in range(10) :
    data = dataloaders.dataloaders['knee_1yr'](args)
    train_data, dev_data = get_basic_data(data, structured=args.structured, truncate=90)
    config = {'vocab' : data.vocab, 
              'stop_words' : True, 
              'exp_name' : data.name, 
              'type' : data.metrics_type, 
              'norm' : 'l2', 
              'constant_mul' : 1.0, 
              'has_structured' : args.structured, 
              'lda' : False, 
              'methods' : ['count'],
              'only_structured' : args.structured,
              'basepath' : args.output_dir}

    lr = LR(config)
    lr.train(train_data)
    lr.evaluate(dev_data, save_results=True)