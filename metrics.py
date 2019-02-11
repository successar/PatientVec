from sklearn.metrics import *
import numpy as np
from pandas.io.json.normalize import nested_to_record

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
        