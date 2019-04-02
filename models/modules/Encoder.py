import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import Registrable
from typing import List
from allennlp.nn.activations import Activation

class Encoder(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")


def wrap_pytorch_rnn(mod, name) :
    @Encoder.register(name)
    class BiRNNEncoder(Encoder) :
        def __init__(self, input_size, hidden_size) :
            super(BiRNNEncoder, self).__init__()
            self.output_size = hidden_size*2
            self.rnn = mod(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            
        def forward(self, seqs, lengths) :
            packseq = nn.utils.rnn.pack_padded_sequence(seqs, lengths, batch_first=True, enforce_sorted=False)
            output, h = self.rnn(packseq)
            output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)
            if type(h) == tuple :
                h, c = h
                
            h = torch.cat([h[0], h[1]], dim=-1)
            return h, output
        
    return BiRNNEncoder
        
wrap_pytorch_rnn(nn.LSTM, "lstm")
wrap_pytorch_rnn(nn.GRU, "gru")

@Encoder.register("cnn")
class CNNEncoder(Encoder) :
    def __init__(self, input_size, hidden_size, kernel_sizes, activation:Activation=Activation.by_name('relu')) :
        super(CNNEncoder, self).__init__()
        convs = {}
        for i in range(len(kernel_sizes)) :
            convs[str(i)] = nn.Conv1d(input_size, hidden_size, kernel_sizes[i], padding=int((kernel_sizes[i] - 1)//2))

        self.convolutions = nn.ModuleDict(convs)
        self.activation = activation

        self.output_size = hidden_size * len(kernel_sizes)

    def forward(self, seqs, lengths) :
        seq_t = seqs.transpose(1, 2)
        outputs = [self.convolutions[i](seq_t) for i in sorted(self.convolutions.keys())]

        output = self.activation(torch.cat(outputs, dim=1))
        h = F.max_pool1d(output, kernel_size=output.size(-1)).squeeze(-1)

        return h, output.transpose(1, 2)

@Encoder.register("average")
class AverageEncoder(Encoder) :
    def __init__(self, input_size, projection, hidden_size=None, activation:Activation=Activation.by_name('linear')) :
        super(AverageEncoder, self).__init__()
        if projection :
            self.projection = nn.Linear(input_size, hidden_size)
            self.output_size = hidden_size
        else :
            self.projection = lambda s : s
            self.output_size = input_size

        self.activation = activation

    def forward(self, seqs, lengths) :
        output = self.activation(self.projection(seqs))
        h = output.mean(1)

        return h, output

from sru import SRU
@Encoder.register("sru")
class SRUEncoder(Encoder) :
    def __init__(self, input_size, hidden_size) :
        super(SRUEncoder, self).__init__()
        self.output_size = hidden_size * 2
        self.rnn = SRU(input_size, hidden_size, num_layers=2, bidirectional=True, rescale=True)

    def forward(self, seqs, lengths) :
        seqs = seqs.transpose(0, 1) #(L, B, E)
        output, h = self.rnn(seqs) #(L, B, H*2), #(Layers, B, H*2)
        h = h[1] #(B, H*2)            
        return h, output.transpose(0, 1)