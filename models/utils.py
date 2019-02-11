import numpy as np
import torch

from tensorboardX import SummaryWriter

class Holder() : 
    def __init__(self, data) :
        maxlen = max([len(x) for x in data])
        self.maxlen = maxlen
        self.B = len(data)

        lengths = []
        expanded = []
        masks = []

        for _, d in enumerate(data) :
            rem = maxlen - len(d)
            expanded.append(d + [0]*rem)
            lengths.append(len(d))
            masks.append([1] + [0]*(len(d)-2) + [1]*(rem+1))

        self.lengths = torch.LongTensor(np.array(lengths)).cuda()
        self.seq = torch.LongTensor(np.array(expanded, dtype='int64')).cuda()
        self.masks = torch.ByteTensor(np.array(masks)).cuda()

    def depad(self, seq) :
        depadded_seq = []
        for i in range(len(seq)) :
            depadded_seq.append(seq[i][:self.lengths[i]].cpu().data.numpy())

        return depadded_seq

class HierHolder() :
    def __init__(self, data) :
        self.doclens = [len(x) for x in data]
        flatten_data = [y for d in data for y in d]
        self.flatten_holder = Holder(flatten_data) 
        self.flatten_index, self.flatten_mask, self.B, self.maxdoclen = self.generate_index(self.doclens)

    def generate_index(self, lengths) :
        maxlen = max(lengths)
        indices = []
        masks = []
        for i in range(len(lengths)) :
            start = i * maxlen
            indices += [start + j for j in range(lengths[i])]
            masks.append([0] * lengths[i] + [1] * (maxlen - lengths[i]))
            
        return torch.LongTensor(indices).cuda(), torch.ByteTensor(masks).cuda(), len(lengths), maxlen

    def depad(self, seq) :
        depadded_seq = []
        for i in range(len(seq)) :
            depadded_seq.append(seq[i][:self.doclens[i]].cpu().data.numpy())

        return depadded_seq

    def unflatten(self, tensor) :
        tensor_shape = tensor.shape[1:]
        new_shape = tuple([self.B * self.maxdoclen] + list(tensor_shape))
        new_tensor = torch.zeros(new_shape, requires_grad=tensor.requires_grad, dtype=tensor.dtype).cuda()
        new_tensor = new_tensor.index_copy(0, self.flatten_index, tensor)
        new_tensor = new_tensor.reshape(tuple([self.B, self.maxdoclen] + list(tensor_shape)))
        return new_tensor

    def unflatten_list(self, list_of_tensors) :
        #(BL, )
        list_of_lists = []
        idx = 0
        for i in range(len(self.doclens)) :
            list_of_lists.append(list_of_tensors[idx:idx+self.doclens[i]])
            idx += self.doclens[i]

        return list_of_lists

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
        name = name.replace(' ', '_')
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), global_step)

    def add_train_histogram(self, name: str, values: torch.Tensor, global_step: int) -> None:
        name = name.replace(' ', '_')
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, global_step)

class ModelTensorboardWriter :
    def __init__(self, dirname) :
        self._tensorboard = TensorboardWriter(dirname)

    def add_model_params_stats(self, epoch, model, model_name):
        for name, param in model.named_parameters():
            self._tensorboard.add_train_scalar(model_name + "/parameter_mean/" + name, param.data.mean(), epoch)
            self._tensorboard.add_train_scalar(model_name + "/parameter_std/" + name, param.data.std(), epoch)
            if param.grad is not None:
                grad_data = param.grad.data
                self._tensorboard.add_train_scalar(model_name + "/gradient_mean/" + name, grad_data.mean(), epoch)
                self._tensorboard.add_train_scalar(model_name + "/gradient_std/" + name, grad_data.std(), epoch)
                self._tensorboard.add_train_scalar(model_name + "/gradient_norm/" + name, torch.norm(grad_data), epoch)

    def add_metrics(self, epoch, metrics) :
        for key in metrics :
            keyed_metrics = metrics[key]
            for name, metric in keyed_metrics.items() :
                self._tensorboard.add_train_scalar(key + "/metric/" + name, metric, epoch)

def get_sorting_index_with_noise(X, noise_frac) :
    lengths = [len(x) for x in X]
    if noise_frac > 0 :
        noisy_lengths = [x + np.random.randint(np.floor(-x*noise_frac), np.ceil(x*noise_frac)) for x in lengths]
    else :
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)

def get_sorting_index_with_noise_from_lengths(lengths, noise_frac) :
    if noise_frac > 0 :
        noisy_lengths = [x + np.random.randint(np.floor(-x*noise_frac), np.ceil(x*noise_frac)) for x in lengths]
    else :
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)