from sklearn.metrics import *
import numpy as np
from pandas.io.json.normalize import nested_to_record
from collections import defaultdict
import pandas as pd
from IPython.display import display


def calc_metrics_classification(target, predictions) :
    if predictions.shape[-1] == 1 :
        predictions = predictions[:, 0]
        predictions = np.array([1 - predictions, predictions]).T

    predict_classes = np.argmax(predictions, axis=-1)
    rep = nested_to_record(classification_report(target, predict_classes, output_dict=True), sep='/')
    rep.update({'accuracy' : accuracy_score(target, predict_classes)})
    if predictions.shape[-1] == 2 :
        rep.update({'roc_auc' : roc_auc_score(target, predictions[:, 1])})
        rep.update({"pr_auc" : average_precision_score(target, predictions[:, 1])})
    return rep

def calc_metrics_regression(target, predictions) :
    rep = {}
    rep['rmse'] = np.sqrt(mean_squared_error(target, predictions))
    rep['mae'] = mean_absolute_error(target, predictions)
    rep['r2'] = r2_score(target, predictions)

    return rep

def calc_metrics_multilabel(target, predictions) :
    rep = {}
    target = np.array(target)
    nlabels = target.shape[1]
    predict_classes = np.where(predictions > 0.5, 1, 0)
    for i in range(nlabels) :
        rep_i = nested_to_record(classification_report(target[:, i], predict_classes[:, i], output_dict=True), sep='/')
        rep_i.update({'accuracy' : accuracy_score(target[:, i], predict_classes[:, i])})
        rep_i.update({'roc_auc' : roc_auc_score(target[:, i], predictions[:, i])})
        rep_i.update({"pr_auc" : average_precision_score(target[:, i], predictions[:, i])})
        for k in list(rep_i.keys()) :
            rep_i['label_' + str(i) + '/' + k] = rep_i[k]
            del rep_i[k]
            
        rep.update(rep_i)
    
    macro_roc_auc = np.mean([v for k, v in rep.items() if 'roc_auc' in k])
    macro_pr_auc = np.mean([v for k, v in rep.items() if 'pr_auc' in k])
    
    rep['macro_roc_auc'] = macro_roc_auc
    rep['macro_pr_auc'] = macro_pr_auc
    
    return rep

def calc_metrics_multitask(target, predictions) :
    n_tasks = len(target[0]) // 2
    target = np.array(target).reshape(-1, n_tasks, 2)
    
    tasks_reps = {}
    for t in range(n_tasks) :
        idxs = np.where(target[:, t, ])
    rep = {}
    target = np.array(target)
    nlabels = target.shape[1]
    predict_classes = np.where(predictions > 0.5, 1, 0)
    for i in range(nlabels) :
        rep_i = nested_to_record(classification_report(target[:, i], predict_classes[:, i], output_dict=True), sep='/')
        rep_i.update({'accuracy' : accuracy_score(target[:, i], predict_classes[:, i])})
        rep_i.update({'roc_auc' : roc_auc_score(target[:, i], predictions[:, i])})
        rep_i.update({"pr_auc" : average_precision_score(target[:, i], predictions[:, i])})
        for k in list(rep_i.keys()) :
            rep_i['label_' + str(i) + '/' + k] = rep_i[k]
            del rep_i[k]
            
        rep.update(rep_i)
    
    macro_roc_auc = np.mean([v for k, v in rep.items() if 'roc_auc' in k])
    macro_pr_auc = np.mean([v for k, v in rep.items() if 'pr_auc' in k])
    
    rep['macro_roc_auc'] = macro_roc_auc
    rep['macro_pr_auc'] = macro_pr_auc
    
    return rep

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
        if not k.startswith('label_') :
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v
        if '/1/' in k :
            d[k.split('/', 1)[0]][k.split('/', 1)[1]] = v

    df = pd.DataFrame(d)
    with pd.option_context('display.max_columns', 30):
        display(df.round(3))
        