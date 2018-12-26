import torch
import torch.nn as nn
import torch.nn.functional as F

class Binary_Predictor(nn.Module) :
    def __init__(self) :
        super().__init__()

    def forward(self, potential, target, weight) :
        predictions = torch.sigmoid(potential)
        if target is not None :
            pos_weight = weight[1]/weight[0] if weight is not None else None
            loss = F.binary_cross_entropy_with_logits(potential, target.unsqueeze(-1), pos_weight=pos_weight)
        else :
            loss = 0.0
            
        return predictions, loss

class Multiclass_predictor(nn.Module) :
    def __init__(self) :
        super().__init__()

    def forward(self, potential, target, weight) :
        predictions = nn.Softmax(dim=-1)(potential)
        if target is not None :
            loss = F.cross_entropy(potential, target, weight=weight)
        else :
            loss = 0.0

        return predictions, loss

predictors = { 'binary' : Binary_Predictor, 'multiclass' : Multiclass_predictor }

def get_predictors(params) :
    return predictors[params['name']](), {'name' : params['name']}