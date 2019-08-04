from trainer import Trainer

from __future__ import absolute_import, division, print_function, unicode_literals
!pip install -q tensorflow==2.0.0-beta1
import tensorflow as tf

import glob

class TFTrainer(Trainer):
    def __init__(self, data_loc):
        self.data_loc = data_loc
        super().__init__()

    def init_parameters(self):
        self.params = {
            'data_loc': self.data_loc,
        }

    def load_data(self):
        if not hasattr(self, data):
            filenames = glob.glob(self.data_loc + '*')
            self.data = tf.data.TFRecordDataset(filenames=filenames)

    def train(self, model, log_file):
        self.load_data()
        prediction = model.predict()
        raise NotImplementedError
