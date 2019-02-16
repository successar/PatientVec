import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common import Registrable

class Predictor(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        pass

@Predictor.register("binary")
class Binary_Predictor(Predictor) :
    def forward(self, potential, target, weight) :
        predictions = torch.sigmoid(potential)
        if target is not None :
            pos_weight = weight[0][1]/weight[0][0] if weight is not None else None
            loss = F.binary_cross_entropy_with_logits(potential, target, pos_weight=pos_weight)
        else :
            loss = 0.0
            
        return predictions, loss

@Predictor.register("multiclass")
class Multiclass_predictor(Predictor) :
    def forward(self, potential, target, weight) :
        predictions = nn.Softmax(dim=-1)(potential)
        if target is not None :
            loss = F.cross_entropy(potential, target.long(), weight=weight)
        else :
            loss = 0.0

        return predictions, loss

@Predictor.register("regression")
class Regression_predictor(Predictor) :
    def forward(self, potential, target, weight) :
        predictions = potential
        if target is not None :
            loss = F.mse_loss(potential, target)
        else :
            loss = 0.0

        return predictions, loss

@Predictor.register("multilabel")
class Multilabel_predictor(Predictor) :
    def forward(self, potential, target, weight) :
        # weight : (C, 2), potential : (N, C), target : (N, C)
        predictions = torch.sigmoid(potential)
        if target is not None :
            loss = F.binary_cross_entropy_with_logits(potential, target, reduction='none')
            if weight is not None :
                new_weight = torch.gather(weight.unsqueeze(0).expand(potential.shape[0], -1, -1), 2, target.unsqueeze(-1).long()).squeeze(-1)
                loss = (loss * new_weight).mean()
            else :
                loss = loss.mean()
        else :
            loss = 0.0

        return predictions, loss