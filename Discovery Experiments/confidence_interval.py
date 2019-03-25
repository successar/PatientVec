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

args.structured = False
args.output_dir = os.path.join(args.output_dir, 'confidence')
args.n_iters = 8

configs = [LSTM, LSTM_with_attention]

data = dataloaders.dataloaders['hip_1yr'](args)
basic_experiments(data, configs, args)
train_data, dev_data = get_basic_data(data, structured=True, truncate=90)
config = {'vocab' : data.vocab, 
          'stop_words' : True, 
          'exp_name' : data.name, 
          'type' : data.metrics_type, 
          'norm' : 'l2', 
          'constant_mul' : 1.0, 
          'has_structured' : True, 
          'lda' : False, 
          'methods' : ['count'],
          'only_structured' : True,
          'basepath' : args.output_dir}

lr = LR(config)
lr.train(train_data)
lr.evaluate(dev_data, save_results=True)

del data

data = dataloaders.dataloaders['knee_1yr'](args)
basic_experiments(data, configs, args)
train_data, dev_data = get_basic_data(data, structured=True, truncate=90)
config = {'vocab' : data.vocab, 
          'stop_words' : True, 
          'exp_name' : data.name, 
          'type' : data.metrics_type, 
          'norm' : 'l2', 
          'constant_mul' : 1.0, 
          'has_structured' : True, 
          'lda' : False, 
          'methods' : ['count'],
          'only_structured' : True,
          'basepath' : args.output_dir}

lr = LR(config)
lr.train(train_data)
lr.evaluate(dev_data, save_results=True)