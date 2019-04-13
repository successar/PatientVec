import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common import Registrable
from .Feedforward import FeedForward

class Predictor(nn.Module, Registrable) :
    def __init__(self, replicate=False, alpha=0.5) :
        self.replicate = replicate
        self.alpha = alpha
        super().__init__()

    def forward(self, **kwargs) :
        pass

@Predictor.register("binary")
class Binary_Predictor(Predictor) :
    def forward(self, potential, target, weight, masks=None, potential_seq=None) :
        predictions = torch.sigmoid(potential)
        
        if target is not None :
            pos_weight = weight[0][1]/weight[0][0] if weight is not None else None
            loss = F.binary_cross_entropy_with_logits(potential, target, pos_weight=pos_weight)
            if self.replicate :
                assert ((masks is not None) & (potential_seq is not None))
                target_repl = target.unsqueeze(1).repeat(1, potential_seq.shape[1], 1)
                loss_seq = F.binary_cross_entropy_with_logits(potential_seq, target_repl, pos_weight=pos_weight, reduction='none').squeeze(-1)
                loss_seq = (loss_seq * (1 - masks).float())
                loss_seq = loss_seq.mean(1).mean()
                loss = self.alpha * loss_seq + (1 - self.alpha) * loss
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

from copy import deepcopy
@Predictor.register('multitask')
class MultiTask_predictor(Predictor) :
    def __init__(self, n_tasks:int, task_decoder:FeedForward) :
        super(MultiTask_predictor, self).__init__()
        assert n_tasks > 1
        self.decoders = nn.ModuleList([deepcopy(task_decoder) for i in range(n_tasks)])

    def forward(self, potential, target, weight) :
        # potential : (N, H), target : (N, 2*T), weight : (T, 2)
        N, T = target.shape[0], target.shape[1]//2
        target = target.reshape(N, T, 2) #(N, T, 2)
        assert len(self.decoders) == T
        loss = 0.0
        predictions = []
        for i in range(len(self.decoders)) :
            potential_task = self.decoders[i](potential) #(B, 1)
            prediction_task = torch.sigmoid(potential_task) #(B, 1)
            predictions.append(prediction_task.unsqueeze(1))
            if target is not None :
                pos_weight = weight[i][1]/weight[i][0] if weight is not None else None
                loss_task = F.binary_cross_entropy_with_logits(potential_task, target[:, i, 0].unsqueeze(-1), pos_weight=pos_weight, reduction='none')
                loss += (loss_task * target[:, i, 1].unsqueeze(-1)).mean()

        predictions = torch.cat(predictions, dim=1)
        return predictions, loss


@Predictor.register("multilabel")
class Multilabel_predictor(Predictor) :
    def forward(self, potential, target, weight, masks=None, potential_seq=None) :
        # weight : (C, 2), potential : (N, C), target : (N, C), masks : (N, L), potential_seq : (N, L, C)
        predictions = torch.sigmoid(potential)
        if target is not None :
            loss = F.binary_cross_entropy_with_logits(potential, target, reduction='none')
            if weight is not None :
                pos_weight = weight[:, 1] / weight[:, 0] #(C, )
                new_weight = target * pos_weight + (1 - target) #(N, C)
            else :
                new_weight = 1.0
            loss = (loss * new_weight).mean()
            if self.replicate :
                assert ((masks is not None) & (potential_seq is not None))
                target_repl = target.unsqueeze(1).repeat(1, potential_seq.shape[1], 1) #(N, L, C)
                loss_seq = F.binary_cross_entropy_with_logits(potential_seq, target_repl, reduction='none') #(N, L, C)
                loss_seq = (loss_seq * new_weight.unsqueeze(1)).mean(-1) #(N, L)
                loss_seq = (loss_seq * (1 - masks).float()) #(N, L)
                loss_seq = loss_seq.mean(1).mean() #()
                loss = self.alpha * loss_seq + (1 - self.alpha) * loss
        else :
            loss = 0.0

        return predictions, loss