from PatientVec.dataset import Dataset
import os

def readmission_dataset(args) :
    data = Dataset(name='Readmission', dirname=os.path.join(args.data_dir, 'preprocess/Readmission/'))

    labellist = ['y']
    data.generate_labels(labellist, len(labellist), 'binary')
    data.save_on_metric = 'roc_auc'
    data.metrics_type = 'classifier'

    data.generate_encoded_field('gender_y', 'onehot')
    data.generate_encoded_field('age_y', 'onehot')
    data.generate_encoded_field('ethnicity_y', 'onehot')
    features = [x for x in data.dataframe.columns if x.startswith('feature')]
    for f in features :
        data.generate_encoded_field(f, 'trivial')
    data.set_structured_params(regexs=[r'^feature', 'gender_y', 'age_y', 'ethnicity_y'])
    
    data.keys_to_use = ['1/precision', '1/recall', '1/f1-score', 'accuracy', 'roc_auc', 'pr_auc']
    
    return data

def diagnosis_dataset(args) :
    data = Dataset(name='Diagnosis', dirname=os.path.join(args.data_dir, 'preprocess/Diagnosis/'))

    labellist = [x for x in data.dataframe.columns if x.startswith('y_')]
    data.generate_labels(labellist, len(labellist), 'multilabel')
    data.save_on_metric = 'macro_roc_auc'
    data.metrics_type = 'multilabel'

    data.generate_encoded_field('gender_y', 'onehot')
    data.generate_encoded_field('age_y', 'onehot')
    data.generate_encoded_field('ethnicity_y', 'onehot')
    features = [x for x in data.dataframe.columns if x.startswith('feature')]
    for f in features :
        data.generate_encoded_field(f, 'trivial')
    data.set_structured_params(regexs=[r'^feature', 'gender_y', 'age_y', 'ethnicity_y'])
    
    data.keys_to_use = ['macro_roc_auc', 'macro_pr_auc']

    return data