import numpy as np
import os, pickle
from math import ceil

from tqdm import tqdm_notebook

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
        sequences = [self.map_to_idxs(t, boundaries) for t in texts]
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

class DataHolder() :
    def __init__(self, **kwargs) :
        for n, v in kwargs.items() :
            setattr(self, n, v)

    def save(self, dirname) :
        os.makedirs(dirname, exist_ok=True)
        pickle.dump(self, open(dirname + '/data.p', 'wb'))

    def load(self, dirname) :
        data = pickle.load(open(dirname + '/data.p', 'rb'))
        return data