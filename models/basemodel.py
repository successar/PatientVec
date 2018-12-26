import torch
import torch.nn as nn

from .utils import Holder, ModelTensorboardWriter

import numpy as np

import os,shutil,json,time

from tqdm import tqdm_notebook

class BaseModel() :
    def __init__(self, params, **kwargs) :
        self.initialize_model(params['model'])
        self.initialize_training(params['training'])
        self.time_str = time.ctime().replace(' ', '')
        self.exp_name = params['exp_name']
        self.dirname = 'outputs/' + kwargs['classname'] + '/' + self.exp_name + '/' + self.time_str
        self.tensorboard_writer = ModelTensorboardWriter(self.dirname)
        self.file_name = kwargs['filename']

    def get_config(self) :
        config = { 'exp_name' : self.exp_name, 'model' : self.model_config, 'training' : self.training_config }
        return config
        
    def initialize_training(self, training_params) :
        raise NotImplementedError("Initialise Training Function")

    def initialize_model(self, model_params) :
        raise NotImplementedError("Initialise Model Function")
        
    @classmethod
    def init_from_config(cls, dirname, **kwargs) :
        config = json.load(open(dirname + '/config.json', 'r'))
        obj = cls(config, **kwargs)
        obj.load_values(dirname)
        return obj

    def get_batch_variable(self, data) :
        data = Holder(data, do_sort=True)
        return data

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
        json.dump(self.get_config(), open(dirname + '/config.json', 'w'))

        if save_model :
            self.save_model(dirname)

        return dirname

    def load_values(self, dirname) :
        self.load_model(dirname)
