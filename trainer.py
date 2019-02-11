from tqdm import tqdm_notebook
import numpy as np
from metrics import *
import json
import pandas as pd
from collections import defaultdict

from IPython.display import display

metrics_map = {
    'classifier' : calc_metrics_classification,
    'regression' : calc_metrics_regression,
    'multilabel' : calc_metrics_multilabel
}

def print_metrics(metrics) :
    tabular = {k:v for k, v in metrics.items() if '/' in k}
    non_tabular = {k:v for k, v in metrics.items() if '/' not in k}
    print(non_tabular)

    d = defaultdict(dict)
    for k, v in tabular.items() :
        if '/1/' in k :
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v

    df = pd.DataFrame(d)
    with pd.option_context('display.max_columns', 30):
        display(df.round(3))

class Trainer() :
    def __init__(self, Model, config, _type='classifier', display_metrics=True) :
        self.model = Model(config)
        self.metrics = metrics_map[_type]
        self.display_metrics = display_metrics
    
    def train(self, train_data, test_data, n_iters=20, save_on_metric='roc_auc') :
        best_metric = 0.0
        for i in tqdm_notebook(range(n_iters)) :
            self.model.train(train_data)

            outputs = self.model.evaluate(test_data)
            predictions = np.array(outputs["predictions"])
            test_metrics = self.metrics(test_data.y, predictions)
            if self.display_metrics :
                print_metrics(test_metrics)
            self.model.tensorboard_writer.add_metrics(self.model.epoch, {'test' : test_metrics})

            metric = test_metrics[save_on_metric]
            if metric > best_metric and i > 0 :
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)
            else :
                save_model = False
                print("Model not saved on ", save_on_metric, metric)
            
            dirname = self.model.save_values(save_model=save_model)
            f = open(dirname + '/epoch.txt', 'a')
            f.write(str(test_metrics) + '\n')
            f.close()


class Evaluator() :
    def __init__(self, Model, dirname, _type='classifier', display_metrics=True) :
        self.model = Model.init_from_config(dirname, evaluate=True)
        self.model.dirname = dirname
        self.metrics = metrics_map[_type]
        self.display_metrics = display_metrics

    def evaluate(self, test_data, save_results=False) :
        outputs = self.model.evaluate(test_data)
        outputs['predictions'] = np.array(outputs["predictions"])

        test_metrics = self.metrics(test_data.y, outputs['predictions'])
        if self.display_metrics :
            print_metrics(test_metrics)

        if save_results :
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        return outputs