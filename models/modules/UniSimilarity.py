import torch
import torch.nn as nn

from allennlp.common import Registrable
from allennlp.nn.activations import Activation

class UniSimilarity(nn.Module, Registrable) :
    pass

@UniSimilarity.register("additive")
class UniAdditiveSimilarity(UniSimilarity) :
    def __init__(self, tensor_1_dim, hidden_size, output_size:int=1, activation:Activation = Activation.by_name('tanh')) :
        super(UniAdditiveSimilarity, self).__init__()
        self.tensor_1_dim = tensor_1_dim
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_1 = nn.Linear(tensor_1_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size, bias=False)

        self.activation = activation()

    def forward(self, tensor_1) :
        sim_1 = self.linear_1(tensor_1)
        sim_2 = self.linear_2(self.activation(sim_1))
        
        if self.output_size == 1 :
            sim_2 = sim_2.squeeze(-1)

        return sim_2

@UniSimilarity.register("scaled_dot")
class UniScaledDotSimilarity(UniSimilarity) :
    def __init__(self, tensor_1_dim, hidden_size=128) :
        super(UniScaledDotSimilarity, self).__init__()
        self.tensor_1_dim = tensor_1_dim
        self.linear_1 = nn.Linear(tensor_1_dim, 1, bias=False)

    def forward(self, tensor_1) :
        sim_1 = self.linear_1(tensor_1).squeeze(-1) / self.tensor_1_dim**0.5

        return sim_1