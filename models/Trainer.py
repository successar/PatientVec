import json
import os
import re
import shutil
import time
from copy import deepcopy

import torch.optim as O
from allennlp.common import Params

from .Model import Model
from .utils import ModelTensorboardWriter

class Trainer :
    def __init__(self, configuration) :
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        exp_config = configuration['exp_config']
        self.time_str = time.ctime().replace(' ', '_')
        self.exp_name = exp_config['exp_name']
        basepath = exp_config.get('basepath', 'outputs')
        
        self.dirname = os.path.join(basepath, self.exp_name, self.time_str)
        if 'evaluate' not in exp_config or not exp_config['evaluate'] :
            self.tensorboard_writer = ModelTensorboardWriter(self.dirname)

        self.file_name = exp_config['filename']

        self.model = Model.from_params(Params(configuration['model'])).cuda()
        self.training_config = configuration['training_config']

        self.generate_optimizer()

    def generate_optimizer(self) :
        self.parameters = dict(self.model.named_parameters())

        self.training_common_config = self.training_config['common']
        self.bsize = self.training_common_config.get("bsize", 32)

        param_groups = []
        for param_regex, param_optim in self.training_config['groups'] :
            param_optim.update({
                "params" : [v for k, v in self.parameters.items() if re.match(param_regex, k)]
            })
            param_groups.append(param_optim)

            self.parameters = {k:v for k, v in self.parameters.items() if not re.match(param_regex, k)}

        self.optimizer = O.__dict__[self.training_config['type']](param_groups)
        self.epoch = 0
        
    @classmethod
    def init_from_config(cls, dirname, **kwargs) :
        config = json.load(open(dirname + '/config.json', 'r'))
        for n, v in kwargs.items() :
            config['exp_config'][n] = v
        obj = cls(config)
        obj.load_values(dirname)
        return obj

    def train(self, **kwargs) :
        raise NotImplementedError("Initialise Method train")

    def evaluate(self, **kwargs) :
        raise NotImplementedError("Initialise Method evaluate")

    def save_model(self, dirname) :
        raise NotImplementedError("Initialise save_model")

    def load_model(self, dirname) :
        raise NotImplementedError("Initialise load_model")
  
    def save_values(self, use_dirname=None, save_model=True) :
        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname

        os.makedirs(dirname, exist_ok=True)

        shutil.copy2(self.file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            self.save_model(dirname)

        return dirname

    def load_values(self, dirname) :
        self.load_model(dirname)
