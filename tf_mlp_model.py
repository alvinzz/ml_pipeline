from model import Model

import tensorflow as tf

class TF_MLP_Model(Model):
    param_type = "TF_MLP_Model"

    def __init__(self):
        self.in_size = 1
        self.hidden_sizes = [1]
        self.out_size = 1

        self.activation = 'relu'

    def update_parameters(self):
        self.params = {
            'param_type': TF_MLP_Model.param_type,

            'in_size': self.in_size,
            'hidden_sizes': self.hidden_sizes,
            'out_size': self.out_size,

            'activation': self.activation,
        }

    def build_model(self):
        if not hasattr(self, 'model'):
            layers = []

            for layer in range(len(self.hidden_sizes) + 1):
                if layer == 0:
                    layers.append(tf.keras.layers.Dense(self.hidden_sizes[0],
                        activation=self.activation, input_shape=[self.in_size]))
                elif layer == len(self.hidden_sizes):
                    layers.append(tf.keras.layers.Dense(self.out_size))
                else:
                    layers.append(tf.keras.layers.Dense(self.hidden_sizes[layer]))
            
            self.model = tf.keras.Sequential(layers)

    def predict(self, input):
        self.build_model()

        return self.model.predict(input['feature'])