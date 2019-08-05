from trainer import Trainer

import tensorflow as tf
import glob

class TF_Trainer(Trainer):
    param_type = 'TF_Trainer'

    optimizer_type_dict = {
        'Adam': tf.keras.optimizers.Adam,
    }

    def __init__(self):
        self.data_loc = './'
        self.load_checkpoint = False

        self.random_seed = 0
        self.dataset_shuffle_buffer_size = 1000
        
        self.optimizer_type = 'Adam'
        self.learning_rate = 0.1
        
        self.n_epochs = 100
        self.batch_size = 10

        self.start_epoch = 0
        self.log_period = 10
        self.save_period = 10

    def update_parameters(self):
        self.params = {
            'param_type': TF_Trainer.param_type,

            'data_loc': self.data_loc,
            'load_checkpoint': self.load_checkpoint,

            'random_seed': self.random_seed,
            'dataset_shuffle_buffer_size': self.dataset_shuffle_buffer_size,

            'optimizer_type': self.optimizer,
            'learning_rate': self.learning_rate,
            
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,

            'start_epoch': self.start_epoch,
            'log_period': self.log_period,
            'save_period': self.save_period,
        }

    def load_data(self):
        if not hasattr(self, data):
            filenames = glob.glob(self.data_loc + '*')
            with tf.device('/cpu:0'):
                dataset = tf.data.TFRecordDataset(filenames=filenames)
                dataset = dataset.shuffle(buffer_size=self.dataset_shuffle_buffer_size)
                dataset = dataset.map(self.example_parser)
                dataset = dataset.batch(self.batch_size)
                dataset = dataset.prefetch(1)
                self.data = dataset

    def example_parser(self, example):
        print("TODO: define example_parser method of TF_Trainer to parse TFExamples from a TFRecordDataset")
        raise NotImplementedError

    @tf.function
    def train(self, model, log_dir):
        if not self.load_checkpoint:
            assert self.start_epoch == 0, "If not loading from checkpoint, start_epoch should be 0"
        if self.load_checkpoint:
            assert self.start_epoch != 0, "If loading from checkpoint, start_epoch should not be 0"

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        self.load_data()

        self.optimizer = TF_Trainer.optimizer_type_dict[self.optimizer_type](self.learning_rate)

        self.checkpoint = tf.train.Checkpoint(model=model)
        if self.load_checkpoint:
            self.checkpoint.restore(log_dir + '/ckpt.{}'.format(self.start_epoch))
        self.checkpoint.save_counter = self.start_epoch

        self.summary_writer = tf.summary.create_file_writer(log_dir + '/train_log')
        self.metrics = {
            'loss': tf.keras.metrics.Mean(name='loss', dtype=tf.float32),
            'grad_mag': tf.keras.metrics.Mean(name='grad_mag', dtype=tf.float32),
            'optim_stepsize': tf.keras.metrics.Mean(name='optim_stepsize', dtype=tf.float32)
        }

        with summary_writer.as_default():
            for epoch in range(self.start_epoch, self.start_epoch+self.n_epochs):
                for data_batch in self.data:
                    self.train_step(model, data_batch)
                
                if (epoch + 1) % self.save_period == 0:
                    self.checkpoint.save(file_prefix=(log_dir + '/ckpt'))
                
                if (epoch + 1) % self.log_period == 0:
                    for (metric_name, tf_metric) in self.metrics.items():
                        tf.summary.scalar(metric_name, tf_metric.result(), step=epoch)
                        tf_metric.reset_states()

    def get_loss(self, data_batch, model_pred):
        print("TODO: define get_loss method of TF_Trainer")
        raise NotImplementedError

    def train_step(self, model, data_batch):
        with tf.GradientTape() as grad_tape:
            model_pred = model.predict(data_batch)
            loss = self.get_loss(data_batch, model_pred)
        gradients = grad_tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        self.metrics['loss'].update_state(loss)
        self.metrics['grad_mag'].update_state(
            tf.math.reduce_mean([tf.linalg.norm(gradient) for gradient in gradients])
        self.metrics['optim_stepsize'].update_state(
            tf.math.reduce_mean([tf.linalg.norm(weight) for weight in self.optimizer.weights])