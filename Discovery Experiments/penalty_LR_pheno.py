import os
import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run penalty experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument("--label_field", type=str, required=True)

args, extras = parser.parse_known_args()
args.extras = extras

for penalty in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] :
    data = dataloaders.dataloaders[args.dataset](args)
    args.structured = data.structured_dim > 0
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
                  'penalty' : penalty}

        print(config)
        lr = LR(config)
        lr.train(train_data)
        lr.evaluate(name='dev', data=dev_data, save_results=True)
        lr.evaluate(name='test', data=test_data, save_results=True)