import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from math import ceil

from tqdm import tqdm_notebook

import spacy, re
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

def cleaner(text, spacy=True, qqq=True, lower=True) :
    text = re.sub(r'\s+', ' ', text.strip())
    if spacy :
        text = [t.text for t in nlp(text)]
    else :
        text = text.split()
    if lower :
        text = [t.lower() for t in text]
    if qqq :
        text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]
    return " ".join(text)

class Vocabulary:
    def __init__(self, num_words=None, min_df=None):
        self.embeddings = None
        self.word_dim = None
        self.num_words = num_words
        self.min_df = min_df

        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.PAD = '<0>'
        self.UNK = '<UNK>'

    def tokenizer(self, text) :
        return text.split(' ')
    
    def fit(self, texts):
        if self.min_df is not None :    
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, min_df=self.min_df, lowercase=False)
        else :
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, lowercase=False)

        self.cvec.fit_transform(texts)           
        self.word2idx = self.cvec.vocabulary_
        
        for word in self.cvec.vocabulary_ :
            self.word2idx[word] += 4
            
        self.word2idx[self.PAD] = 0
        self.word2idx[self.UNK] = 1
        self.word2idx[self.SOS] = 2
        self.word2idx[self.EOS] = 3
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        self.cvec.stop_words_ = None
        
    def add_word(self, word) :
        if word not in self.word2idx :
            idx = max(self.word2idx.values()) + 1
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.vocab_size += 1

    def extract_embeddings(self, model):
        self.word_dim, self.vocab_size = model.vector_size, len(self.word2idx)
        self.embeddings = np.zeros([self.vocab_size, self.word_dim])
        in_pre = 0
        for i, word in sorted(self.idx2word.items()):
            if word in model :
                self.embeddings[i] = model[word] 
                in_pre += 1
            else :
                self.embeddings[i] = np.random.randn(self.word_dim)
                
        self.embeddings[0] = np.zeros(self.word_dim)
                
        print("Found " + str(in_pre) + " words in model out of " + str(len(self.idx2word)))
        return self.embeddings

    def map_to_words(self, idxlist) :
        return [self.idx2word[x] for x in idxlist]
    
    def map_to_idxs(self, wordlist) :
        return [self.word2idx[x] if x in self.word2idx else self.word2idx[self.UNK] for x in wordlist]

    def add_sentence_boundaries(self, wordlist) :
        return [self.SOS] + wordlist + [self.EOS]

class Vectorizer :
    def __init__(self, vocab):
        self.vocab = vocab

    def convert_to_sequence(self, texts) :
        texts_tokenized = map(self.vocab.tokenizer, texts)
        texts_tokenized = map(lambda s : self.vocab.map_to_idxs(self.vocab.add_sentence_boundaries(s)), texts_tokenized)
        texts_tokenized = list(texts_tokenized)
        return texts_tokenized

    def texts_to_sequences(self, texts):
        unpad_X = self.convert_to_sequence(texts)
        return unpad_X