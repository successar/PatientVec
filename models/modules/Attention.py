import torch
import torch.nn as nn

from allennlp.common.from_params import FromParams

from .BiSimilarity import BiSimilarity
from .sparsemax import Sparsemax

class Attention(nn.Module, FromParams) :
    def __init__(self, similarity:BiSimilarity, normaliser:str = 'softmax') :
        super(Attention, self).__init__()
        self.similarity = similarity
        if normaliser == 'softmax' :
            self.normaliser = nn.Softmax(dim=-1)
        elif normaliser == 'sparsemax' :
            self.normaliser = Sparsemax(dim=-1)

    def forward(self, tensor_1, tensor_2, mask) :
        sim_1 = self.similarity(tensor_1, tensor_2) #(B, *)
        sim_1.masked_fill_(mask.byte(), float("-inf"))
        attn_1 = self.normaliser(sim_1)

        return attn_1
