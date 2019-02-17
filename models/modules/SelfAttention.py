import torch
import torch.nn as nn

from .UniSimilarity import UniSimilarity
from allennlp.common.from_params import FromParams
from .sparsemax import Sparsemax

class SelfAttention(nn.Module, FromParams) :
    def __init__(self, similarity:UniSimilarity, normaliser:str = 'softmax') :
        super(SelfAttention, self).__init__()
        self.similarity = similarity
        self.norm_type = normaliser
        if normaliser == 'softmax' :
            self.normaliser = nn.Softmax(dim=-1)
        elif normaliser == 'sparsemax' :
            self.normaliser = Sparsemax(dim=-1)

    def forward(self, tensor_1, mask) :
        sim_1 = self.similarity(tensor_1) #(B, *)
        if self.norm_type == 'sparsemax' :
            sim_1.masked_fill_(mask.byte(), -1e-20)
        else :
            sim_1.masked_fill_(mask.byte(), float("-inf"))
        
        attn_1 = self.normaliser(sim_1)

        return attn_1
