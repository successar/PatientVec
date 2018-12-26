import pandas as pd 
import pickle
import numpy as np
from PatientVec.preprocess.vocabulary import Vocabulary
from PatientVec.preprocess.embedder import PretrainedEmbedding
from PatientVec.preprocess.vectorizer import Vectorizer, DataHolder

from sklearn.model_selection import train_test_split
from copy import deepcopy

class Dataset() :
    def __init__(self, name, dirname, labelfield, train_size=0.8) :
        self.name = name
        self.vocab = Vocabulary().load(dirname)
        self.embedding = PretrainedEmbedding().load(dirname)
        self.sequences = Vectorizer().load_sequences(dirname + '/note_sequences.p')
        self.dataframe = pd.read_csv(dirname + '/data_nonotes.csv')

        self.labels = np.array(self.dataframe[labelfield])

        self.idxs = {}
        self.idxs['train'], self.idxs['test'] = train_test_split(list(range(len(self.labels))), stratify=self.labels, train_size=train_size, random_state=1298)

    def get_data(self, field) :
        filtered_idxs = [i for i in self.idxs[field] if len(self.sequences[i]) < 10000]
        X = [self.sequences[i] for i in filtered_idxs]
        y = [self.labels[i] for i in filtered_idxs]
        print("Pos Percentage", sum(y)/len(y))
        return DataHolder(X=X, y=y)

    def generate_label_map_for_probing_field(self, probing_field) :
        probing_labels = sorted(self.dataframe[probing_field].unique())
        return {k:i for i, k in enumerate(probing_labels)}

    def get_data_for_probing(self, probing_field, field) :
        probing_labels = list(self.dataframe[probing_field])
        label_map = self.generate_label_map_for_probing_field(probing_field)
        mapped_probing_labels = list(map(lambda s : label_map[s], probing_labels))
        
        filtered_idxs = [i for i in self.idxs[field] if len(self.sequences[i]) < 10000]
        X = [self.sequences[i] for i in filtered_idxs]
        y = [mapped_probing_labels[i] for i in filtered_idxs]
        print("Pos Percentage", sum(y)/len(y))
        return DataHolder(X=X, y=y)


def get_basic_model_config(data, exp_name, prober=False) :
    config = {
        'exp_name' : exp_name,
        'training' : {
            'bsize' : 32,
            'weight_decay' : 1e-4,
            'class_weight' : True
        },
        'model' : {
            'embedder' : {
                'name' : 'token',
                'params' : {
                    'vocab_size' : data.vocab.vocab_size,
                    'embed_size' : data.embedding.word_dim,
                    'pre_embed' : data.embedding.embeddings
                }
            },
            'encoder' : {
                'name' : 'LSTM', 
                'params' : {
                    'hidden_size' : 128
                }
            },
            'decoder' : {
                'name' : 'MLP',
                'params' : {
                    'hidden_size' : 128,
                    'output_size' : 1
                },
                'predictor' : {
                    'name' : 'binary'
                }
            }
        }
    }

    if prober :
        config['model']['prober'] = {
            'name' : 'MLP',
            'params' : {
                'hidden_size' : 128,
                'output_size' : 1
            },
            'predictor' : {
                'name' : 'binary'
            }
        }

    return config