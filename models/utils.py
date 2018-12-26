import numpy as np
import torch

from tensorboardX import SummaryWriter

class Holder() :
    
    def __init__(self, data, do_sort=False) :
        maxlen = max([len(x) for x in data])
        self.maxlen = maxlen
        self.B = len(data)
        lengths = []
        expanded = []
        masks = []
        masks_no_ends = []

        for _, d in enumerate(data) :
            rem = maxlen - len(d)
            expanded.append(d + [0]*rem)
            lengths.append(len(d))
            masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))
            masks_no_ends.append([0]*len(d) + [1]*(rem))

        lengths = np.array(lengths)  
        self.orig_lengths = lengths.copy()

        expanded = np.array(expanded, dtype='int64')
        idxs = np.flip(np.argsort(lengths), axis=0).copy()

        self.do_sort = do_sort

        if do_sort :
            lengths = lengths[idxs]
            expanded = expanded[idxs]

        self.lengths = torch.LongTensor(lengths).cuda()
        self.seq = torch.LongTensor(expanded).cuda()

        masks = np.array(masks)
        self.masks = torch.ByteTensor(masks).cuda()

        self.masks_no_ends = torch.ByteTensor(np.array(masks_no_ends)).cuda()

        self.correcting_idxs = torch.LongTensor(np.argsort(idxs)).cuda()
        self.sorting_idxs = torch.LongTensor(idxs).cuda()

    def sort(self, seq) :
        return seq[self.sorting_idxs]

    def correct(self, seq) :
        return seq[self.correcting_idxs]

class TensorboardWriter:
    def __init__(self, dirname: str) -> None:
        self._train_log = SummaryWriter(log_dir=dirname + '/log')

    @staticmethod
    def _item(value):
        if hasattr(value, 'item'):
            val = value.item()
        else:
            val = value
        return val

    def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), global_step)

    def add_train_histogram(self, name: str, values: torch.Tensor, global_step: int) -> None:
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, global_step)


class ModelTensorboardWriter :
    def __init__(self, dirname) :
        self._tensorboard = TensorboardWriter(dirname)

    def _add_model_params_stats(self, epoch, model, model_name):
        for name, param in model.named_parameters():
            self._tensorboard.add_train_scalar(model_name + "/parameter_mean/" + name, param.data.mean(), epoch)
            self._tensorboard.add_train_scalar(model_name + "/parameter_std/" + name, param.data.std(), epoch)
            if param.grad is not None:
                grad_data = param.grad.data
                self._tensorboard.add_train_scalar(model_name + "/gradient_mean/" + name, grad_data.mean(), epoch)
                self._tensorboard.add_train_scalar(model_name + "/gradient_std/" + name, grad_data.std(), epoch)
                self._tensorboard.add_train_scalar(model_name + "/gradient_norm/" + name, torch.norm(grad_data), epoch)

    def _add_metrics(self, epoch, metrics) :
        for key in metrics :
            keyed_metrics = metrics[key]
            for name, metric in keyed_metrics.items() :
                self._tensorboard.add_train_scalar(key + "/metric/" + name, metric, epoch)
                