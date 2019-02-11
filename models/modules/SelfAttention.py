import torch
import torch.nn as nn

from .UniSimilarity import UniSimilarity
from allennlp.common.from_params import FromParams

class SelfAttention(nn.Module, FromParams) :
    def __init__(self, similarity:UniSimilarity) :
        super(SelfAttention, self).__init__()
        self.similarity = similarity

    def forward(self, tensor_1, mask) :
        sim_1 = self.similarity(tensor_1) #(B, *)
        sim_1.masked_fill_(mask.byte(), float("-inf"))
        attn_1 = nn.Softmax(dim=-1)(sim_1)

        return attn_1
