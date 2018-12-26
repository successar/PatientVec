import torch
import torch.nn as nn

def wrap_pytorch_rnn(mod) :
    class BiRNNEncoder(nn.Module) :
        def __init__(self, input_size, hidden_size) :
            super().__init__()
            self.output_size = hidden_size*2
            
            self.config = {
                'input_size' : input_size,
                'hidden_size' : hidden_size
            }

            self.rnn = mod(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            
        def forward(self, seqs, lengths, **kwargs) :
            packseq = nn.utils.rnn.pack_padded_sequence(seqs, lengths, batch_first=True)
            output, h = self.rnn(packseq)
            output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)
            if type(h) == tuple :
                h, c = h
                
            h = torch.cat([h[0], h[1]], dim=-1)
            return h, output
        
    return BiRNNEncoder
        
encoders = {'LSTM' : wrap_pytorch_rnn(nn.LSTM), 'GRU' : wrap_pytorch_rnn(nn.GRU)}

def get_encoder(params) :
    encoder_name = params['name']
    encoder = encoders[encoder_name](**params['params'])
    encoder_config = { 'name' : encoder_name, 'params' : encoder.config}
    return encoder, encoder_config