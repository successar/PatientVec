from allennlp.common import Registrable
import torch.nn as nn

class Model(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        pass

from .Classifier import *