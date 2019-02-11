from dataset import Dataset
import numpy as np

from common import *

import argparse
parser = argparse.ArgumentParser(description='Run Diagnosis experiments')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument("--output_dir", type=str)

args = parser.parse_args()

data = Dataset(name='Diagnosis', dirname=os.path.join(args.data_dir, 'preprocess/Diagnosis/'))
labellist = [x for x in data.dataframe.columns if x.startswith('y_')]
data.generate_labels(labellist, len(labellist), 'multilabel')

data.generate_encoded_field('gender_y', 'onehot')
data.generate_encoded_field('age_y', 'onehot')
data.generate_encoded_field('ethnicity_y', 'onehot')

features = [x for x in data.dataframe.columns if x.startswith('feature')]
for f in features :
    data.generate_encoded_field(f, 'trivial')
    
data.set_structured_params(regexs=[r'^feature', 'gender_y', 'age_y', 'ethnicity_y'])

# from trainer import Trainer, Evaluator
# from models.Vanilla import ClassificationTrainer as BasicCT
# from models.Hierarchical import ClassificationTrainer as HierCT

# from Experiments.experiments import experiments, hierarchical_experiments, structured_experiments

# train_data = data.get_data('train', structured=True)
# dev_data = data.get_data('dev', structured=True)

# train_data = data.filter_data_length(train_data, truncate=90)
# dev_data = data.filter_data_length(dev_data, truncate=90)

# for e in experiments :
#     config = e(data, structured=True)
#     print(config)
#     trainer = Trainer(BasicCT, config, _type='multilabel', display_metrics=args.display)
#     trainer.train(train_data, dev_data, save_on_metric='macro_roc_auc')

#     evaluator = Evaluator(BasicCT, trainer.model.dirname, _type='multilabel', display_metrics=args.display)
#     _ = evaluator.evaluate(dev_data, save_results=True)
#     print('='*300)
    
# for e in experiments :
#     for use_structured in [True, False] :
#         config = e(data, structured=use_structured)
#         print(config)
#         trainer = Trainer(BasicCT, config, _type='multilabel', display_metrics=args.display)
#         trainer.train(train_data, dev_data, save_on_metric='macro_roc_auc')

#         evaluator = Evaluator(BasicCT, trainer.model.dirname, _type='multilabel', display_metrics=args.display)
#         _ = evaluator.evaluate(dev_data, save_results=True)
#         print('='*300)
        
# for e in hierarchical_experiments :
#     for use_structured in [True, False] :
#         config = e(data, structured=use_structured)
#         print(config)
#         trainer = Trainer(HierCT, config, display_metrics=args.display)
#         trainer.train(train_data, dev_data)

#         evaluator = Evaluator(HierCT, trainer.model.dirname, display_metrics=args.display)
#         _ = evaluator.evaluate(dev_data, save_results=True)
#         print('='*30)
        
# train_data = data.get_data('train', structured=True, encodings=data.structured_columns)
# dev_data = data.get_data('dev', structured=True, encodings=data.structured_columns)

# train_data = data.filter_data_length(train_data, truncate=90)
# dev_data = data.filter_data_length(dev_data, truncate=90)

# for e in structured_experiments :
#     for use_structured in [True, False] :
#         config = e(data, structured=use_structured, encodings=data.structured_columns)
#         print(config)

#         trainer = Trainer(BasicCT, config, display_metrics=args.display)
#         trainer.train(train_data, dev_data)

#         evaluator = Evaluator(BasicCT, trainer.model.dirname, display_metrics=args.display)
#         _ = evaluator.evaluate(dev_data, save_results=True)
#         print('='*30)