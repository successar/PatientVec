from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
import numpy as np
from pandas.io.json.normalize import nested_to_record 

def calc_metrics(target, predictions) :
    predict_classes = (predictions > 0.5)
    rep = nested_to_record(classification_report(target, predict_classes, output_dict=True), sep='/')
    rep.update({'accuracy' : accuracy_score(target, predict_classes)})
    rep.update({'roc_auc' : roc_auc_score(target, predictions)})
    return rep

class ClassificationTrainer() :
    def __init__(self, Model, config) :
        self.model = Model(config)
    
    def train(self, train_data, test_data, n_iters=30, save_on_metric='1/f1-score') :
        best_metric = 0.0
        for i in tqdm_notebook(range(n_iters)) :
            self.model.train(train_data.X, train_data.y)

            predictions = self.model.evaluate(test_data.X)
            predictions = np.array(predictions)
            test_metrics = calc_metrics(test_data.y, predictions)
            print(test_metrics)
            self.model.tensorboard_writer._add_metrics(self.model.epoch, {'test' : test_metrics})

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

class ClassificationEvaluator() :
    def __init__(self, Model, dirname) :
        self.model = Model.init_from_config(dirname)

    def evaluate(self, test_data, average='binary') :
        predictions = self.model.evaluate(test_data.X)
        predictions = np.array(predictions)
        test_metrics = calc_metrics(test_data.y, predictions)
        print(test_metrics)
        return predictions