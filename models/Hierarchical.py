import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm_notebook

from .Trainer import Trainer
from .Iterator import Hierarchical_Generator

file_name = os.path.abspath(__file__)

class ClassificationTrainer(Trainer) :
    def __init__(self,  configuration) :
        configuration['exp_config']['filename'] = file_name
        super(ClassificationTrainer, self).__init__(configuration=configuration)

    def generate_optimizer(self) :
        self.class_weight = self.training_config['common'].get("class_weight", False)
        super().generate_optimizer()
        
    def train(self, train_data) :
        generator = Hierarchical_Generator(train_data, batch_size=self.bsize, sort_and_shuffle=True)
        target = generator.train_data.y

        class_weight = None
        if self.class_weight :
            class_weight = []
            for i in range(len(target[0])) :
                class_weight.append(torch.Tensor(compute_class_weight('balanced', np.sort(np.unique(target[:, i])), target[:, i])).cuda().unsqueeze(0))

            class_weight = torch.cat(class_weight, dim=0)
        
        self.model.train()
        N = generator.N
        loss_total = 0
        num_batches = 0

        for batch in tqdm_notebook(generator) :
            torch.cuda.empty_cache()
            batch.weight = class_weight
            
            self.model(batch)
            loss = batch.outputs["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            if num_batches % 10 == 0 :
                self.tensorboard_writer.add_model_params_stats(num_batches, self.model, 'model')

            loss_total += float(loss.data.cpu().item())
            num_batches += 1

        loss_total = loss_total*self.bsize/N
        self.tensorboard_writer.add_metrics(self.epoch, {'train' : {'loss' :  loss_total}})

        self.epoch += 1

    def evaluate(self, test_data) :
        generator = Hierarchical_Generator(test_data, self.bsize, sort_and_shuffle=False)

        predictions = []
        word_attentions = []
        sentence_attentions = []

        for batch in tqdm_notebook(generator) :
            torch.cuda.empty_cache()

            self.model(batch)
            predict = batch.outputs['predict']

            predict = predict.cpu().data.numpy()
            predictions.append(predict)

            word_attentions.append(batch.outputs['word_attention'])
            sentence_attentions.append(batch.outputs['sentence_attention'])

        predictions = [x for y in predictions for x in y]
        word_attentions = [x for y in word_attentions for x in y]
        sentence_attentions = [x for y in sentence_attentions for x in y]

        return { 
            "predictions" : predictions, 
            "word_attentions" : word_attentions, 
            "sentence_attentions" : sentence_attentions 
        }
  
    def save_model(self, dirname) :
        torch.save(self.model.state_dict(), dirname + '/model.th')

    def load_model(self, dirname) :
        self.model.load_state_dict(torch.load(dirname + '/model.th'))
