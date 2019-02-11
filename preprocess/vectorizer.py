import numpy as np
import os, pickle
from math import ceil

from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

import nltk
from nltk.corpus import stopwords

from collections import Counter

class Vectorizer :
    def __init__(self, vocab=None, boundaries=True):
        self.vocab = vocab
        self.boundaries = True

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
        for t in tqdm_notebook(texts) :
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

class BoWder :
    def __init__(self, vocab=None, stop_words=False) :
        self.vocab = vocab 

        self.words_to_remove = set([self.vocab.PAD, self.vocab.SOS, self.vocab.UNK, self.vocab.EOS])
        if stop_words :
           self.words_to_remove = (set(stopwords.words('english')) & set(self.vocab.word2idx.keys())) | self.words_to_remove

        self.idxs_to_remove = set([self.vocab.word2idx[x] for x in list(self.words_to_remove)])
        self.words_to_keep = list(set(self.vocab.word2idx.keys()) - self.words_to_remove)

        self.map_vocab_to_bow = {self.vocab.word2idx[k]:i for i, k in enumerate(self.words_to_keep)}
        self.map_bow_to_vocab = {v:k for k, v in self.map_vocab_to_bow.items()}

    def get_bow(self, X) :
        bow = np.zeros((len(X), len(self.words_to_keep)))
        for i, x in enumerate(tqdm_notebook(X)) :
            x = set(x) - self.idxs_to_remove
            counts = Counter(x)
            for w, c in counts.items() :
                bow[i, self.map_vocab_to_bow[w]] += c

        return csr_matrix(bow)

    def fit_tfidf(self, X) :
        bow = self.get_bow(X)
        self.tfidftransform = TfidfTransformer()
        self.tfidftransform.fit(bow)

    def get_tfidf(self, X) :
        bow = self.get_bow(X)
        return self.tfidftransform.transform(bow)


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

    def filter(self, idxs) :
        data_kwargs = { key: [getattr(self, key)[i] for i in idxs] for key in self.attributes}
        return DataHolder(**data_kwargs)