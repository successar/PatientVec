import torch
import torch.nn as nn

import numpy as np

from allennlp.common import Registrable

class Embedder(nn.Module, Registrable) :
    pass

@Embedder.register("token_embedder")
class TokenEmbedder(Embedder) :
    def __init__(self, vocab_size, embed_size, embedding_file=None) :
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.config = {
            'embed_size' : embed_size,
            'vocab_size' : vocab_size
        }
        
        if embedding_file is not None :
            pre_trained_embedding = np.load(embedding_file)
            print("Setting Embedding")
            weight = torch.Tensor(pre_trained_embedding)
            weight[0, :].zero_()
            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            
    def forward(self, seq) :
        return self.embedding(seq)