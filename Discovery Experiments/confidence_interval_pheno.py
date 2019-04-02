import os
import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *
import time

import argparse
parser = argparse.ArgumentParser(description='Run confidence experiments on a dataset')
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument("--label_field", type=str, required=True)
parser.add_argument('--do_test', dest='do_test', action='store_true')
parser.add_argument('--do_all', dest='do_all', action='store_true')

args, extras = parser.parse_known_args()
args.extras = extras

args.output_dir = os.path.join(args.output_dir, 'confidence')
args.n_iters = 8

configs = [LSTM, LSTM_with_attention, CNN, CNN_with_attention]

data = dataloaders.dataloaders[args.dataset](args)
args.structured = data.structured_dim > 0
for _ in range(10) :
    basic_experiments(data, configs, args)

train_data, dev_data, test_data = get_basic_data(data, structured=args.structured, truncate=90)
for i in range(10) :
    time.sleep(5)
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
    lr.evaluate(name='dev', data=dev_data, save_results=True)
    lr.evaluate(name='test', data=test_data, save_results=True)