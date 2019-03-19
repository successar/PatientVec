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
    
    data.keys_to_use = ['accuracy', 'roc_auc', 'pr_auc']
    
    return data

def mortality_dataset(args, _type) :
    data = Dataset(name='Mortality_'+_type, dirname=os.path.join(args.data_dir, 'preprocess/Mortality/'))

    labellist = ['mortality_' + _type]
    data.generate_labels(labellist, len(labellist), 'binary')
    data.save_on_metric = 'roc_auc'
    data.metrics_type = 'classifier'

    data.generate_encoded_field('gender_y', 'onehot')
    data.generate_encoded_field('age_y', 'onehot')
    data.generate_encoded_field('ethnicity_y', 'onehot')
    features = [x for x in data.dataframe.columns if x.startswith('feature')]
    for f in features :
        if 'sapsii' not in f :
            data.generate_encoded_field(f, 'trivial')
        else :
            data.generate_encoded_field(f, 'scale', {'m' : 0, 'M' : 163})
    data.set_structured_params(regexs=[r'^feature', 'gender_y', 'age_y', 'ethnicity_y'])
    
    data.keys_to_use = ['accuracy', 'roc_auc', 'pr_auc']
    
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

def hip_dataset(args, yr=3) :
    data = Dataset(name='HipSurgery_' + str(yr), dirname=os.path.join(args.data_dir, 'preprocess/HipSurgery/', str(yr) + '_yrs'))

    labellist = ['Target']
    data.generate_labels(labellist, len(labellist), 'binary')
    data.save_on_metric = 'roc_auc'
    data.metrics_type = 'classifier'

#     data.generate_encoded_field('gender_y', 'onehot')
#     data.generate_encoded_field('age_y', 'onehot')
#     data.generate_encoded_field('ethnicity_y', 'onehot')
#     features = [x for x in data.dataframe.columns if x.startswith('feature')]
#     for f in features :
#         data.generate_encoded_field(f, 'trivial')
#     data.set_structured_params(regexs=[r'^feature', 'gender_y', 'age_y', 'ethnicity_y'])
    
    return data

def knee_dataset(args, yr=3) :
    data = Dataset(name='KneeSurgery_' + str(yr), dirname=os.path.join(args.data_dir, 'preprocess/KneeSurgery/', str(yr) + '_yrs'))

    labellist = ['Target']
    data.generate_labels(labellist, len(labellist), 'binary')
    data.save_on_metric = 'roc_auc'
    data.metrics_type = 'classifier'

#     data.generate_encoded_field('gender_y', 'onehot')
#     data.generate_encoded_field('age_y', 'onehot')
#     data.generate_encoded_field('ethnicity_y', 'onehot')
#     features = [x for x in data.dataframe.columns if x.startswith('feature')]
#     for f in features :
#         data.generate_encoded_field(f, 'trivial')
#     data.set_structured_params(regexs=[r'^feature', 'gender_y', 'age_y', 'ethnicity_y'])
    
    return data

dataloaders = {
    'readmission' : readmission_dataset,
    'mortality_30day' : lambda x : mortality_dataset(x, '30day'),
    'mortality_1yr' : lambda x : mortality_dataset(x, '1yr'),
    'diagnosis' : diagnosis_dataset,
    'hip' : hip_dataset,
    'knee' : knee_dataset
}