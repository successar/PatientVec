import torch
import torch.nn as nn

from .UniSimilarity import UniSimilarity
from allennlp.common.from_params import FromParams
from .sparsemax import Sparsemax

class DimAttention(nn.Module, FromParams) :
    def __init__(self, similarity:UniSimilarity, normaliser:str = 'softmax') :
        super(DimAttention, self).__init__()
        self.similarity = similarity
        self.norm_type = normaliser
        if normaliser == 'softmax' :
            self.normaliser = nn.Softmax(dim=-1)

    def forward(self, tensor_1, mask) :
        sim_1 = self.similarity(tensor_1) #(B, L, H)
        sim_1.masked_fill_(mask.byte().unsqueeze(-1), float("-inf"))
        
        attn_1 = self.normaliser(sim_1)

        return attn_1
