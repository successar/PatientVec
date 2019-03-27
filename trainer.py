from tqdm import tqdm_notebook
import numpy as np
from PatientVec.metrics import *
import json
from collections import deque

class Trainer() :
    def __init__(self, Model, config, _type='classifier', display_metrics=True) :
        self.model = Model(config)
        self.metrics = metrics_map[_type]
        self.display_metrics = display_metrics

    def load_pretrained_model(self, dirname) :
        self.model.load_filtered_model(dirname)
    
    def train(self, train_data, test_data, n_iters=15, save_on_metric='roc_auc') :
        best_metric = -1.0
        last_few_metrics = deque(maxlen=3)
        for _ in tqdm_notebook(range(n_iters)) :
            self.model.train(train_data)

            outputs = self.model.evaluate(test_data)
            predictions = np.array(outputs["predictions"])
            test_metrics = self.metrics(test_data.y, predictions)
            if self.display_metrics :
                print_metrics(test_metrics)
            self.model.tensorboard_writer.add_metrics(self.model.epoch, {'test' : test_metrics})

            metric = test_metrics[save_on_metric]
            last_few_metrics.append(metric)
                
            if metric > best_metric:
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

            if len(last_few_metrics) >= 3 and best_metric not in last_few_metrics :
                print(best_metric, last_few_metrics)
                break


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
            
        test_data.outputs = outputs

        if save_results :
            f = open(self.model.dirname + '/evaluate.json', 'w')
            json.dump(test_metrics, f)
            f.close()

        return outputs

    
