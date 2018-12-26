import torch
import torch.nn as nn
from .decoders import get_decoder
from .embedders import get_embedder
from .encoders import get_encoder

from .classifier import SequenceClassifier

from .basemodel import BaseModel

import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import os,shutil,json,time

file_name = os.path.abspath(__file__)

from tqdm import tqdm_notebook

class Model(BaseModel) :
    def __init__(self, params) :
        super(Model, self).__init__(params, classname='lstm_probing', filename=file_name)

    def initialize_training(self, training_params) :
        self.epoch = training_params.get('epoch', 0)
        self.bsize = training_params.get('bsize', 32)
        self.class_weight = training_params.get('class_weight', False)
        self.weight_decay = training_params.get('weight_decay', 1e-5)
        self.lr = training_params.get('lr', 0.0001)

        self.probe_params = list(self.probe.parameters())
        self.prober_optim = torch.optim.Adam(self.probe_params, lr=self.lr, weight_decay=self.weight_decay)

        self.training_config = {
            'bsize' : self.bsize,
            'class_weight' : self.class_weight,
            'weight_decay' : self.weight_decay,
            'lr' : self.lr,
            'epoch' : self.epoch
        }
        
    def initialize_model(self, model_params) :
        embedder, embedder_config = get_embedder(model_params['embedder'])

        model_params['encoder']['params']['input_size'] = embedder_config['params']['embed_size']
        encoder, encoder_config = get_encoder(model_params['encoder'])

        model_params['decoder']['params']['input_size'] = encoder.output_size
        (decoder, decoder_pred), decoder_config = get_decoder(model_params['decoder'])

        model_params['prober']['params']['input_size'] = encoder.output_size
        (self.probe, probe_pred), probe_config = get_decoder(model_params['prober'])

        for v in encoder.parameters() :
            v.requires_grad = False

        for v in embedder.parameters() :
            v.requires_grad = False 
        
        self.main_classifier = SequenceClassifier(embedder, encoder, decoder, decoder_pred).cuda()
        self.probe_classifier = SequenceClassifier(embedder, encoder, self.probe, probe_pred).cuda()

        self.model_config = {
            'embedder' : embedder_config,
            'encoder' : encoder_config,
            'decoder' : decoder_config,
            'prober' : probe_config
        } 

    def train(self, data_in, target_in, train=True) :
        sorting_idx = np.argsort([len(x) for x in data_in])
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        class_weight = torch.Tensor(compute_class_weight('balanced', np.sort(np.unique(target)), target)).cuda() if self.class_weight else None
        
        self.probe_classifier.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)
        num_batches = self.epoch * len(batches)

        for n in tqdm_notebook(batches) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)
            batch_target = target[n:n+bsize]
            batch_target = torch.Tensor(batch_target).cuda()
            batch_data.target = batch_target
            batch_data.weight = class_weight
            
            self.probe_classifier(batch_data)

            if train :
                self.prober_optim.zero_grad()
                batch_data.loss.backward()
                self.prober_optim.step()
                self.tensorboard_writer._add_model_params_stats(num_batches, self.probe_classifier, 'probe_classifier')

            loss_total += float(batch_data.loss.data.cpu().item())
            num_batches += 1

        loss_total = loss_total*bsize/N
        self.tensorboard_writer._add_metrics(self.epoch, {'train' : {'loss' :  loss_total}})

        self.epoch += 1

    def evaluate(self, data) :
        self.probe_classifier.train()
        bsize = self.bsize
        N = len(data)

        outputs = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)

            self.probe_classifier(batch_data)

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        
        return outputs

    def evaluate_main(self, data) :
        self.main_classifier.eval()
        bsize = self.bsize
        N = len(data)

        outputs = []

        for n in tqdm_notebook(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = self.get_batch_variable(batch_doc)

            self.main_classifier(batch_data)

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        
        return outputs
  
    def save_model(self, dirname) :
        torch.save(self.main_classifier.state_dict(), dirname + '/main_classifier.th')
        torch.save(self.probe_classifier.state_dict(), dirname + '/probe_classifier.th')

    def load_model(self, dirname) :
        self.main_classifier.load_state_dict(torch.load(dirname + '/main_classifier.th'))
        self.probe_classifier.load_state_dict(torch.load(dirname + '/probe_classifier.th'))

    def load_encoder_values(self, dirname) :
        self.main_classifier.load_state_dict(torch.load(dirname + '/main_classifier.th'))
