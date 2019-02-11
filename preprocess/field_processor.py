from sklearn.preprocessing import *
import numpy as np
import pandas as pd

class NoEncoding() :
    def __init__(self) :
        pass
    
    def fit(self, X) :
        if len(X.shape) == 1 :
            X = X.reshape(-1, 1)
        self.output_dim = X.shape[-1]

    def transform(self, X) :
        X = np.array(X)
        if len(X.shape) == 1 :
            X = X.reshape(-1, 1)
        return X
    
    def get_output_dim(self) :
        return self.output_dim

class OneHotEncoding() :
    def __init__(self) :
        self.encoder = LabelBinarizer()

    def fit(self, X) :
        X = np.array(X)
        self.encoder.fit(X.reshape(-1, 1))
        l = len(self.encoder.classes_)
        self.output_dim = 1 if l == 2 else l

    def transform(self, X) :
        X = np.array(X)
        return self.encoder.transform(X.reshape(-1, 1))
        
    def get_output_dim(self) :
        return self.output_dim

class BinnedEncoding() :
    def __init__(self, **kwargs) :
        self.encoder = KBinsDiscretizer(**kwargs)

    def fit(self, X) :
        X = np.array(X)
        self.encoder.fit(X.reshape(-1, 1))
        self.output_dim = self.encoder.n_bins_[0]

    def transform(self, X) :
        X = np.array(X)
        return self.encoder.transform(X.reshape(-1, 1))
            
    def get_output_dim(self) :
        return self.output_dim

class ICDEncoding() :
    def __init__(self, level='chapter', split=';') :
        icd9_df = pd.read_csv("preprocess/common data/icd9_R.csv")
        self.code_to_level = {k:v for k, v in zip(list(icd9_df.code), list(icd9_df[level]))}
        self.split_char = split

        self.encoder = MultiLabelBinarizer(classes=sorted(list(set(self.code_to_level.values()))))
        self.encoder.fit([])
        
        self.output_dim = len(self.encoder.classes_)

    def fit(self, X) :
        pass

    def transform(self, X) :
        X = [x.split(self.split_char) if type(x) == str else [] for x in X]
        levels = [list(set([self.code_to_level[x] for x in y])) for y in X]
        return self.encoder.transform(levels)
    
    def get_output_dim(self) :
        return self.output_dim

field_processors = {
    'trivial' : lambda x : NoEncoding(),
    'onehot' : lambda x : OneHotEncoding(),
    'binned' : lambda x : BinnedEncoding(**x),
    'icd9' : lambda x  :ICDEncoding(**x)
}