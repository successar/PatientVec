from PatientVec.dataset import Dataset
import os
import numpy as np

def pneumonia_dataset(args) :
    data = Dataset(name='Pneumonia', dirname=args.data_dir)

    labellist = [args.label_field]
    data.generate_labels(labellist, len(labellist), 'binary')
    data.save_on_metric = 'roc_auc'
    data.metrics_type = 'classifier'

#     features = [x for x in data.dataframe.columns if x.startswith('feature')]
#     for f in features :
#         data.generate_encoded_field(f, 'trivial')
#     data.set_structured_params(regexs=[r'^feature'])
    
    data.keys_to_use = ['accuracy', 'roc_auc', 'pr_auc']
    
    return data

def immunosuppressed_dataset(args) :
    data = Dataset(name='Immunosuppressed', dirname=args.data_dir)

    labellist = [args.label_field]
    data.generate_labels(labellist, len(labellist), 'binary')
    data.save_on_metric = 'roc_auc'
    data.metrics_type = 'classifier'

#     features = [x for x in data.dataframe.columns if x.startswith('feature')]
#     for f in features :
#         data.generate_encoded_field(f, 'trivial')
#     data.set_structured_params(regexs=[r'^feature'])
    
    data.keys_to_use = ['accuracy', 'roc_auc', 'pr_auc']
    
    return data

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

def readmission_hcup_dataset(args) :
    data = Dataset(name='Readmission_hcup', dirname=os.path.join(args.data_dir, 'preprocess/Readmission/'))

    cols = [x for x in data.dataframe.columns if x.startswith('hcup')]
    topcols = np.array(cols)[(data.dataframe[cols].mean() > 0.10).values]
    labellist = list(topcols)
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

def mortality_hcup_dataset(args, _type) :
    data = Dataset(name='Mortality_'+_type + '_hcup', dirname=os.path.join(args.data_dir, 'preprocess/Mortality/'))

    cols = [x for x in data.dataframe.columns if x.startswith('hcup')]
    topcols = np.array(cols)[(data.dataframe[cols].mean() > 0.10).values]
    labellist = list(topcols)
    data.generate_labels(labellist, len(labellist), 'multilabel')
    data.save_on_metric = 'macro_roc_auc'
    data.metrics_type = 'multilabel'

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
    
    data.keys_to_use = ['macro_roc_auc', 'macro_pr_auc']
    
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
    if args is None :
        data_dir='.'
    else :
        data_dir = args.data_dir
    data = Dataset(name='HipSurgery_' + str(yr), dirname=os.path.join(data_dir, 'preprocess/HipSurgery/', str(yr) + '_yrs'))

    labellist = ['Target']
    data.generate_labels(labellist, len(labellist), 'binary')
    data.save_on_metric = 'roc_auc'
    data.metrics_type = 'classifier'

    features = [x for x in data.dataframe.columns if x in ['Meds', 'Surgery', 'continous', 'Binarized']]
    for f in features :
        data.generate_encoded_field(f, 'trivial')
    data.set_structured_params(regexs=['Meds', r'Surgery$', 'continous', 'Binarized'])
    
    return data

def knee_dataset(args, yr=3) :
    data = Dataset(name='KneeSurgery_' + str(yr), dirname=os.path.join(args.data_dir, 'preprocess/KneeSurgery/', str(yr) + '_yrs'))

    labellist = ['Target']
    data.generate_labels(labellist, len(labellist), 'binary')
    data.save_on_metric = 'roc_auc'
    data.metrics_type = 'classifier'

    features = [x for x in data.dataframe.columns if x in ['Meds', 'Surgery', 'continous', 'Binarized']]
    for f in features :
        data.generate_encoded_field(f, 'trivial')
    data.set_structured_params(regexs=['Meds', r'Surgery$', 'continous', 'Binarized'])
    
    return data

def multitask_surgery_dataset(args, yr=3) :
    data = Dataset(name='MultiTaskSurgery_' + str(yr), dirname=os.path.join(args.data_dir, 'preprocess/MultiTaskSurgery/', str(yr) + '_yrs'))

    labellist = ['Target_Hip', 'Task_Hip', 'Target_Knee', 'Task_Knee']
    data.generate_labels(labellist, len(labellist), 'multitask')
    data.save_on_metric = 'macro_roc_auc'
    data.metrics_type = 'multitask'

    features = [x for x in data.dataframe.columns if x in ['Meds', 'Surgery', 'continous', 'Binarized']]
    for f in features :
        data.generate_encoded_field(f, 'trivial')
    data.set_structured_params(regexs=['Meds', r'Surgery$', 'continous', 'Binarized'])
    
    return data

dataloaders = {
    'readmission' : readmission_dataset,
    'mortality_30day' : lambda x : mortality_dataset(x, '30day'),
    'mortality_1yr' : lambda x : mortality_dataset(x, '1yr'),
    'diagnosis' : diagnosis_dataset,
    'hip' : hip_dataset,
    'knee' : knee_dataset,
    'hip_1yr' : lambda x : hip_dataset(x, 1),
    'knee_1yr' : lambda x : knee_dataset(x, 1),
    'hip_2yr' : lambda x : hip_dataset(x, 2),
    'knee_2yr' : lambda x : knee_dataset(x, 2),
    'hip_3yr' : lambda x : hip_dataset(x, 3),
    'knee_3yr' : lambda x : knee_dataset(x, 3),
    'hip_0.5yr' : lambda x : hip_dataset(x, 0.5),
    'knee_0.5yr' : lambda x : knee_dataset(x, 0.5),
    'hip_0.25yr' : lambda x : hip_dataset(x, 0.25),
    'knee_0.25yr' : lambda x : knee_dataset(x, 0.25),
    'hip_-1yr' : lambda x : hip_dataset(x, -1),
    'knee_-1yr' : lambda x : knee_dataset(x, -1),
    'both_1yr' : lambda x : multitask_surgery_dataset(x, 1),
    'both_2yr' : lambda x : multitask_surgery_dataset(x, 2),
    'both_3yr' : lambda x : multitask_surgery_dataset(x, 3),
    'both_0.5yr' : lambda x : multitask_surgery_dataset(x, 0.5),
    'both_0.25yr' : lambda x : multitask_surgery_dataset(x, 0.25),
    'both_-1yr' : lambda x : multitask_surgery_dataset(x, -1),
    'readmission_hcup' : readmission_hcup_dataset,
    'mortality_30day_hcup' : lambda x : mortality_hcup_dataset(x, '30day'),
    'pneumonia' : pneumonia_dataset,
    'immuno' : immunosuppressed_dataset
}