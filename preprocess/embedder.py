import os
import pickle

import numpy as np

class PretrainedEmbedding() :
    def __init__(self, vocab=None) :
        if vocab is not None :
            self.vocab = vocab
            self.idx2word = self.vocab.idx2word
            self.vocab_size = max(self.idx2word.keys()) + 1
            self.word_dim = None

    def extract_embeddings_from_gensim_model(self, model):
        self.word_dim = model.vector_size
        self.embeddings = np.zeros([self.vocab_size, self.word_dim])

        in_pre = 0
        for i, word in self.idx2word.items():
            if word in model :
                self.embeddings[i] = model[word] 
                in_pre += 1
            else :
                self.embeddings[i] = np.random.randn(self.word_dim)
                
        if self.vocab.padding :
            padding_idx = self.vocab.word2idx[self.vocab.PAD]
            self.embeddings[padding_idx] = np.zeros(self.word_dim)
                
        print("Found " + str(in_pre) + " words in model out of " + str(self.vocab_size))

    def save(self, dirname) :
        os.makedirs(dirname, exist_ok=True)
        matrix_filepath = os.path.join(dirname, 'embedding_matrix.npy')
        np.save(matrix_filepath, self.embeddings)
        
        mapping_filepath = os.path.join(dirname, 'embedding_mapping.p')
        pickle.dump(self.idx2word, open(mapping_filepath, 'wb'))

    def load(self, dirname) :
        self.embedding_file = os.path.join(dirname, 'embedding_matrix.npy')
        self.embeddings = np.load(self.embedding_file)
        
        mapping_filepath = os.path.join(dirname, 'embedding_mapping.p')
        self.idx2word = pickle.load(open(mapping_filepath, 'rb'))
        
        self.vocab_size = max(self.idx2word.keys()) + 1
        self.word_dim = self.embeddings.shape[1]
        assert self.vocab_size == self.embeddings.shape[0], "Size of Vocab and embeddings don't match"

        return self
