import torch
import torch.nn as nn

class TokenEmbedder(nn.Module) :
    def __init__(self, vocab_size, embed_size, pre_embed=None) :
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.config = {
            'embed_size' : embed_size,
            'vocab_size' : vocab_size
        }
        
        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()
            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            
    def forward(self, seq) :
        return self.embedding(seq)
    
embedders = { 'token' : TokenEmbedder }

def get_embedder(embedder_params) :
    embedder_name = embedder_params['name']
    embedder = embedders[embedder_name](**embedder_params['params'])
    embedder_config = {'name' : embedder_name, 'params' : embedder.config}
    return embedder, embedder_config