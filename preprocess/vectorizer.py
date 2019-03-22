import numpy as np
import os, pickle
from math import ceil

from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

import nltk
from nltk.corpus import stopwords

from collections import Counter

class Vectorizer :
    def __init__(self, vocab=None, boundaries=True):
        self.vocab = vocab
        self.boundaries = boundaries

    def map_to_idxs(self, text, boundaries=True) :
        tokens = self.vocab.tokenizer(text) if type(text) == str else text
        if boundaries :
            tokens = self.vocab.add_sentence_boundaries(tokens)

        indices = self.vocab.map_to_idxs(tokens)
        return indices

    def map_to_words(self, indices) :
        tokens = self.vocab.map_to_words(indices)
        return tokens

    def texts_to_sequences(self, texts, boundaries=True):
        sequences = []
        for t in texts :
            sequences.append(self.map_to_idxs(t, boundaries))
        return sequences

    def sequences_to_texts(self, sequences, concat=False) :
        texts = [self.map_to_words(s) for s in sequences]
        if concat :
            texts = [" ".join(t) for t in texts]

        return texts

    def save_sequences(self, sequences, filename) :
        pickle.dump(sequences, open(filename, 'wb'))

    def load_sequences(self, filename) :
        return pickle.load(open(filename, 'rb'))

from scipy.sparse.linalg import norm as sparse_norm
from numpy.linalg import norm as dense_norm
from scipy.sparse import issparse

class BoWder :
    def __init__(self, vocab=None, stop_words=False, norm=None, constant_mul=1.0) :
        self.vocab = vocab 
        self.norm = norm
        self.constant_mul = constant_mul

        self.words_to_remove = set([self.vocab.PAD, self.vocab.SOS, self.vocab.UNK, self.vocab.EOS])
        if stop_words :
            self.words_to_remove = (set(stopwords.words('english')) & set(self.vocab.word2idx.keys())) | self.words_to_remove

        self.idxs_to_remove = set([self.vocab.word2idx[x] for x in list(self.words_to_remove)])
        self.words_to_keep = list(set(self.vocab.word2idx.keys()) - self.words_to_remove)

        self.map_vocab_to_bow = {self.vocab.word2idx[k]:i for i, k in enumerate(self.words_to_keep)}
        self.map_bow_to_vocab = {v:k for k, v in self.map_vocab_to_bow.items()}

        self.tfidftransform = TfidfTransformer(norm=None)

    def generate_bow(self, X) :
        bow = np.zeros((len(X), len(self.words_to_keep)))
        for i, x in enumerate(tqdm_notebook(X)) :
            x = set(x) - self.idxs_to_remove
            counts = Counter(x)
            for w, c in counts.items() :
                bow[i, self.map_vocab_to_bow[w]] += c

        bow = csr_matrix(bow)
        return bow

    def fit_tfidf(self, X) :
        bow = self.generate_bow(X)
        self.tfidftransform.fit(bow)

    def get_tfidf(self, X) :
        bow = self.generate_bow(X)
        print("TFIDFing ...")
        bow = self.tfidftransform.transform(bow)
        bow = self.normalise_bow(bow)
        if issparse(bow) :
            bow = bow.todense()
        return bow
    
    def get_bow(self, X) :
        bow = self.generate_bow(X)
        bow = self.normalise_bow(bow) 
        if issparse(bow) :
            bow = bow.todense()
        return bow

    def get_binary_bow(self, X) :
        bow = self.generate_bow(X)
        print("Clipping ...")
        bow = np.clip(bow.todense(), 0, 1)
        assert (bow > 1).sum() == 0, (bow > 1).sum()
        bow = self.normalise_bow(bow)
        if issparse(bow) :
            bow = bow.todense()
        return bow

    def normalise_bow(self, bow, use_norm=None) :
        if self.norm is not None or use_norm is not None:
            use_norm = use_norm if use_norm is not None else self.norm
            print("Normalising using " + str(use_norm))
            if use_norm.startswith('l') :
                print('Using Norm from linalg')
                if issparse(bow) :
                    norm_l = sparse_norm(bow, int(use_norm[1]), axis=1)
                else :
                    norm_l = dense_norm(bow, int(use_norm[1]), axis=1)
                norm_l = np.where(norm_l == 0, 1.0, norm_l)
                bow = bow / norm_l[:, None]
                print("Multiplying by constant , ", self.constant_mul)
                bow = bow * self.constant_mul
            else :
                bow = normalize(bow, norm=use_norm, copy=False)

        return bow

class DataHolder() :
    def __init__(self, **kwargs) :
        for n, v in kwargs.items() :
            assert hasattr(v, '__iter__'), n + " Value is not Iterable"
            setattr(self, n, v)

        self.attributes = list(kwargs.keys())

    def save(self, dirname) :
        os.makedirs(dirname, exist_ok=True)
        pickle.dump(self, open(dirname + '/data.p', 'wb'))

    def load(self, dirname) :
        data = pickle.load(open(dirname + '/data.p', 'rb'))
        return data

    def add_fields(self, **kwargs) :
        for n, v in kwargs.items() :
            assert hasattr(v, '__iter__'), n + " Value is not Iterable"
            setattr(self, n, v)

        self.attributes += list(kwargs.keys())

    def mock(self, n=200) :
        data_kwargs = { key: getattr(self, key)[:n] for key in self.attributes}
        return DataHolder(**data_kwargs)

    def sample(self, n) :
        sample = np.random.choice(range(len(self.X)), size=n, replace=False)
        return self.filter(sample)

    def filter(self, idxs) :
        data_kwargs = { key: [getattr(self, key)[i] for i in idxs] for key in self.attributes}
        return DataHolder(**data_kwargs)