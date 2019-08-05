import os
import datetime

from parameter import Parameter
from model import Model
from trainer import Trainer
from evaluator import Evaluator

class Experiment(Parameter):
    param_type = 'Experiment'

    def __init__(self):
        self.model = Model()

        self.trainer = Trainer()
        self.evaluator = Evaluator()

    def update_parameters(self):
        self.params = {
            'param_type': Experiment.param_type,

            'model': self.model,

            'trainer': self.trainer,
            'evaluator': self.evaluator,
        }

    def set_name(self, name):
        currentDT = datetime.datetime.now()
        self.name = name + '_' + currentDT.strftime("%Y_%m_%d_%H_%M_%S")
        os.mkdir(self.name)

        self.trainer.exp_name = self.name
        self.evaluator.exp_name = self.name

    def save(self):
        super().save(self.name)

    def train(self):
        self.trainer.train(self.model)

    def val(self):
        self.evaluator.val(self.model)

    def test(self):
        self.evaluator.test(self.model)
