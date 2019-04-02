import os

from PatientVec.common import get_latest_model
from PatientVec.models.Vanilla import ClassificationTrainer as BasicCT
from PatientVec.models.Hierarchical import ClassificationTrainer as HierCT
from PatientVec.trainer import Trainer, Evaluator

def get_evaluator(dataset, model_name, trainer='basic', basename='outputs') :
    model_dirname = get_latest_model(os.path.join(basename, dataset.name, model_name))
    evaluator = Evaluator(BasicCT, model_dirname, _type=dataset.metrics_type, display_metrics=True)
    return evaluator