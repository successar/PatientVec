import torch
import torch.nn as nn

from allennlp.common.from_params import FromParams

from .BiSimilarity import BiSimilarity

class Attention(nn.Module, FromParams) :
    def __init__(self, similarity:BiSimilarity) :
        super(Attention, self).__init__()
        self.similarity = similarity

    def forward(self, tensor_1, tensor_2, mask) :
        sim_1 = self.similarity(tensor_1, tensor_2) #(B, *)
        sim_1.masked_fill_(mask.byte(), float("-inf"))
        attn_1 = nn.Softmax(dim=-1)(sim_1)

        return attn_1
