import re
import json
import numpy as np

import pickle
from sklearn.feature_extraction.text import CountVectorizer

import os

class Vocabulary:
    def __init__(self, num_words=None, min_df=None, boundaries=True, padding=True, unk=True):
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.PAD = '<000>'
        self.UNK = '<UNK>'

        self.boundaries = boundaries
        self.padding = padding
        self.unk = unk

        self.num_words = num_words
        self.min_df = min_df

        self.config = {
            'boundaries' : self.boundaries,
            'padding' : self.padding,
            'unk' : self.unk,
            'num_words' : self.num_words,
            'min_df' : self.min_df
        }

    def tokenizer(self, text) :
        return text.split(' ')

    def extend_word_idx(self, n) :
        for word in self.word2idx :
            self.word2idx[word] += n        
    
    def fit(self, texts):
        if self.min_df is not None :    
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, min_df=self.min_df, lowercase=False)
        else :
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, lowercase=False)

        self.cvec.fit_transform(texts)           
        self.word2idx = self.cvec.vocabulary_

        if self.boundaries :
            self.extend_word_idx(2)
            self.word2idx[self.SOS] = 0
            self.word2idx[self.EOS] = 1

        if self.unk :
            self.extend_word_idx(1)
            self.word2idx[self.UNK] = 0

        if self.padding :
            self.extend_word_idx(1)
            self.word2idx[self.PAD] = 0       
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        self.cvec.stop_words_ = None
        
    def add_word(self, word) :
        if word not in self.word2idx :
            idx = max(self.word2idx.values()) + 1
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.vocab_size += 1

    def map_to_words(self, idxlist) :
        return [self.idx2word[x] for x in idxlist]
    
    def map_to_idxs(self, wordlist) :
        return [self.word2idx[x] if x in self.word2idx else self.word2idx[self.UNK] for x in wordlist]

    def add_sentence_boundaries(self, wordlist) :
        assert self.boundaries, "No Boundaries present in vocabulary"
        if type(wordlist[0]) == int :
            return [self.word2idx[self.SOS]] + wordlist + [self.word2idx[self.EOS]]

        return [self.SOS] + wordlist + [self.EOS]

    def remove_sentence_boundaries(self, wordlist) :
        assert self.boundaries, "No Boundaries present in vocabulary"
        if type(wordlist[0]) == int :
            return [x for x in wordlist if x not in (self.word2idx[self.SOS], self.word2idx[self.EOS])]

        return [x for x in wordlist if x not in (self.SOS, self.EOS)]

    def get_word_list(self) :
        return list(self.word2idx.keys())

    def save(self, dirname) :
        os.makedirs(dirname, exist_ok=True)
        data = {'config' : self.config, 'word2idx' : self.word2idx}
        filepath = os.path.join(dirname, 'vocabulary.p')
        pickle.dump(data, open(filepath, 'wb'))

    def load(self, dirname) :
        filepath = os.path.join(dirname, 'vocabulary.p')
        data = pickle.load(open(filepath, 'rb'))

        config = data['config']
        self.__dict__.update(config)
        self.config = config

        self.word2idx = data['word2idx']
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        return self        