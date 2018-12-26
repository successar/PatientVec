import torch
import torch.nn as nn

from .predictors import get_predictors

class MLP(nn.Module) :
    def __init__(self, input_size, hidden_size, output_size) :
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.config = {
            'input_size' : input_size,
            'hidden_size' : hidden_size,
            'output_size' : output_size
        }

        self.MLP = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_size)
                    )
        
    def forward(self, x) :
        return self.MLP(x)

class Logistic(nn.Module) :
    def __init__(self, input_size, output_size) :
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.config = {
            'input_size' : input_size,
            'output_size' : output_size
        }

        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x) :
        return self.linear(x)
    
decoders = {
    'MLP' : MLP, 
    'Logistic' : Logistic
}

def get_decoder(params) :
    decoder_name = params['name']
    decoder = decoders[decoder_name](**params['params'])

    predictor, predictor_config = get_predictors(params['predictor'])
    decoder_config = {'name' : decoder_name, 'params' : decoder.config, 'predictor' : predictor_config}

    return (decoder, predictor), decoder_config