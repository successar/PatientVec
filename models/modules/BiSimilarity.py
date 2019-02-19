import torch
import torch.nn as nn

from allennlp.common import Registrable
from allennlp.nn.activations import Activation

class BiSimilarity(nn.Module, Registrable) :
    pass

@BiSimilarity.register("additive")
class BiAdditiveSimilarity(BiSimilarity) :
    def __init__(self, tensor_1_dim, tensor_2_dim, hidden_size, activation:Activation = Activation.by_name('tanh')) :
        super(BiAdditiveSimilarity, self).__init__()
        self.tensor_1_dim = tensor_1_dim
        self.tensor_2_dim = tensor_2_dim
        self.hidden_size = hidden_size

        self.linear_1 = nn.Linear(tensor_1_dim, hidden_size)
        self.linear_c = nn.Linear(tensor_2_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1, bias=False)

        self.activation = activation()

    def forward(self, tensor_1, tensor_2) :
        sim_1 = self.linear_1(tensor_1) + self.linear_c(tensor_2)
        sim_2 = self.linear_2(self.activation(sim_1)).squeeze(-1)

        return sim_2

@BiSimilarity.register("bilinear")
class BiBilinearSimilarity(BiSimilarity) :
    def __init__(self, tensor_1_dim, tensor_2_dim, hidden_size, activation:Activation = Activation.by_name('tanh')) :
        super(BiBilinearSimilarity, self).__init__()
        self.tensor_1_dim = tensor_1_dim
        self.tensor_2_dim = tensor_2_dim
        self.hidden_size = hidden_size

        self.linear_1 = nn.Bilinear(tensor_1_dim, tensor_2_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1, bias=False)

        self.activation = activation()

    def forward(self, tensor_1, tensor_2) :
        sim_1 = self.linear_1(tensor_1, tensor_2.expand(-1, tensor_1.shape[1], -1))
        sim_2 = self.linear_2(self.activation(sim_1)).squeeze(-1)

        return sim_2