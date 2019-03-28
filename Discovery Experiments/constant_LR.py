import os
import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run penalty experiments on a dataset')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)
parser.add_argument('--constant', type=float, required=True)

args, extras = parser.parse_known_args()
args.extras = extras

for dataset in ['hip_1yr', 'knee_1yr', 'readmission', 'mortality_30day', 'mortality_1yr', 'diagnosis'] :
    data = dataloaders.dataloaders[dataset](args)
    args.structured = data.structured_dim > 0
    train_data, dev_data, test_data = get_basic_data(data, structured=args.structured, truncate=90)

    for norm in [None, 'l1', 'l2'] :
        config = {'vocab' : data.vocab, 
                  'stop_words' : True, 
                  'exp_name' : data.name, 
                    'type' : data.metrics_type, 
                    'norm' : norm, 
                    'constant_mul' : args.constant, 
                    'has_structured' : args.structured, 
                    'lda' : False, 
                    'methods' : ['count', 'binary', 'tfidf'],
                    'only_structured' : args.structured,
                    'basepath' : args.output_dir,
                    'penalty' : 1.0, 
                     'vary_scaling' : True}

        print(config)
        lr = LR(config)
        lr.train(train_data)
        lr.evaluate(name='dev', data=dev_data, save_results=True)
        lr.evaluate(name='test', data=test_data, save_results=True)