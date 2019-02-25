import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm_notebook

from .Trainer import Trainer
from .Iterator import Concatenated_Generator

file_name = os.path.abspath(__file__)

class ClassificationTrainer(Trainer) :
    def __init__(self,  configuration) :
        configuration['exp_config']['filename'] = file_name
        super(ClassificationTrainer, self).__init__(configuration=configuration)

    def generate_optimizer(self) :
        self.class_weight = self.training_config['common'].get("class_weight", False)
        self.balanced = self.training_config['common'].get('balanced', False)
        assert not (self.class_weight & self.balanced), "Both class weight and balanced set ..."
        super().generate_optimizer()
        
    def train(self, train_data) :
        generator = Concatenated_Generator(train_data, batch_size=self.bsize, balanced=self.balanced)
        target = np.array(generator.train_data.y)

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
        self.model.eval()
        generator = Concatenated_Generator(test_data, self.bsize, sort_and_shuffle=False)

        predictions = []
        attentions = []

        for batch in tqdm_notebook(generator) :
            torch.cuda.empty_cache()

            self.model(batch)
            predict = batch.outputs['predict']

            predict = predict.cpu().data.numpy()
            predictions.append(predict)

            if "attention" in batch.outputs :
                attention = batch.outputs['attention']
                attentions.append(attention)

        predictions = [x for y in predictions for x in y]

        if len(attentions) > 0:
            attentions = [x for y in attentions for x in y]

        return { "predictions" : predictions, "attentions" : attentions }
  
    def save_model(self, dirname) :
        torch.save(self.model.state_dict(), dirname + '/model.th')

    def load_model(self, dirname) :
        self.model.load_state_dict(torch.load(dirname + '/model.th'))

    def load_filtered_model(self, dirname) :
        pretrained_dict = {k:v for k, v in torch.load(dirname + '/model.th').items() if not k.startswith('decoder')}
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

