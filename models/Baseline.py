import os
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm_notebook

from .Trainer import Trainer
from .Iterator import Vector_Generator

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
        generator = Vector_Generator(train_data, batch_size=self.bsize, balanced=self.balanced)
        target = np.array(generator.train_data.y)

        class_weight = self.compute_class_weight(target)
    
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
        generator = Vector_Generator(test_data, self.bsize, sort_and_shuffle=False)

        predictions = []

        for batch in tqdm_notebook(generator) :
            torch.cuda.empty_cache()

            self.model(batch)
            predict = batch.outputs['predict']

            predict = predict.cpu().data.numpy()
            predictions.append(predict)

        predictions = [x for y in predictions for x in y]

        return { "predictions" : predictions }
  
    def save_model(self, dirname) :
        torch.save(self.model.state_dict(), dirname + '/model.th')

    def load_model(self, dirname) :
        self.model.load_state_dict(torch.load(dirname + '/model.th'))

    def load_filtered_model(self, dirname) :
        pretrained_dict = {k:v for k, v in torch.load(dirname + '/model.th').items() if not k.startswith('decoder')}
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

