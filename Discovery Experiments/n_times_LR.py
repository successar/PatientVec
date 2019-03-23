import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run confidence experiments on a dataset')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument('--n', type=int, required=True)
parser.add_argument("--output_dir", type=str)

args, extras = parser.parse_known_args()
args.extras = extras

args.structured = True
args.n_iters = 8
args.seed = 16348

data = dataloaders.dataloaders['readmission'](args)
train_data, dev_data = get_basic_data(data, structured=True, truncate=90)
np.random.seed(args.seed)
train_data = train_data.sample(n=args.n)
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
          'basepath' : args.output_dir + '/ts_experiments/n=' + str(args.n)}

lr = LR(config)
lr.train(train_data)
lr.evaluate(dev_data, save_results=True)

del data

data = dataloaders.dataloaders['mortality_30day'](args)
train_data, dev_data = get_basic_data(data, structured=True, truncate=90)
np.random.seed(args.seed)
train_data = train_data.sample(n=args.n)
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
          'basepath' : args.output_dir + '/ts_experiments/n=' + str(args.n)}

lr = LR(config)
lr.train(train_data)
lr.evaluate(dev_data, save_results=True)