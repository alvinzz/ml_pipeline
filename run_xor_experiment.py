import tensorflow as tf
import numpy as np
import glob

from experiment import Experiment
from tf_mlp_model import TF_MLP_Model
from tf_trainer import TF_Trainer
from tf_losses import mse_loss

# create dataset & example parser
def create_xor_dataset():
	x = np.array([[0,0], [0,1], [1,0], [1,1]]).astype(np.float32)
	y = np.array([[0], [1], [1], [0]]).astype(np.float32)

	def gen_example(x, y):
		feature = {
		  'feature': tf.train.Feature(float_list=tf.train.FloatList(value=x.tolist())),
		  'label': tf.train.Feature(float_list=tf.train.FloatList(value=y.tolist())),
		}
		return tf.train.Example(features=tf.train.Features(feature=feature))

	record_file = 'xor.tfrecords'
	with tf.io.TFRecordWriter(record_file) as writer:
	    for (_x, _y) in zip(x, y):
	        example = gen_example(_x, _y)
	        writer.write(example.SerializeToString())

def xor_example_parser(example):
	feature_description = {
	    'feature': tf.io.FixedLenFeature([2], tf.float32),
	    'label': tf.io.FixedLenFeature([1], tf.float32),
	}
	return tf.io.parse_single_example(example, feature_description)

# define experiment
def setup_experiment():
	XOR_experiment = Experiment()

	XOR_model = TF_MLP_Model()
	XOR_experiment.model = XOR_model

	XOR_trainer = TF_Trainer()
	XOR_trainer.example_parser = xor_example_parser
	XOR_trainer.loss = mse_loss
	XOR_experiment.trainer = XOR_trainer

def get_hyperparams():
	if glob.glob('xor.params'):
		XOR_experiment.load('xor.params')
	else:
		XOR_model.in_size = 2
		XOR_model.hidden_sizes = [20, 20]
		XOR_model.out_size = 1

		XOR_model.activation = 'relu'



		XOR_trainer.data_loc = './'
        XOR_trainer.load_checkpoint = True

        XOR_trainer.random_seed = 0
        XOR_trainer.dataset_shuffle_buffer_size = 1000
        
        XOR_trainer.optimizer_type = 'Adam'
        XOR_trainer.learning_rate = 0.1
        
        XOR_trainer.n_epochs = 100
        XOR_trainer.batch_size = 4

        XOR_trainer.start_epoch = 0
        XOR_trainer.log_period = 10
        XOR_trainer.save_period = 10

if __name__ == '__main__':
	create_xor_dataset()
	setup_experiment()
	get_hyperparams()

	XOR_experiment.train('xor')

	XOR_experiment.save('xor')