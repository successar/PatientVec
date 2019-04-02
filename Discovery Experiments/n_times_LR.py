import PatientVec.dataloaders as dataloaders
from PatientVec.Experiments.hyperparam_exps import *

import argparse
parser = argparse.ArgumentParser(description='Run confidence experiments on a dataset')
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument('--n', type=int, required=True)
parser.add_argument("--output_dir", type=str)

args, extras = parser.parse_known_args()
args.extras = extras

args.n_iters = 8
args.seed = 16348

data = dataloaders.dataloaders[args.dataset](args)
args.structured = data.structured_dim > 0

train_data, dev_data, test_data = get_basic_data(data, structured=args.structured, truncate=90)
np.random.seed(args.seed)

if len(train_data.X) < args.n :
    print("Less train than n", args.n, len(train_data.X))
else :    
    train_data = train_data.sample(n=args.n)
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
              'basepath' : args.output_dir + '/ts_experiments/n=' + str(args.n)}

    lr = LR(config)
    lr.train(train_data)
    lr.evaluate(name='dev', data=dev_data, save_results=True)
    lr.evaluate(name='test', data=test_data, save_results=True)