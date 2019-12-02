import os
import pdb

import layers

class TDNN():
    def __init__(self, data_configs, model_configs, training_configs):
        self.data_configs  = data_configs
        self.model_configs = model_configs
        self.training_configs = training_configs

    def initialize(self):
        '''
            build graph of TDNN models
        '''
        # get input tensor
        record_path = os.path.join('training_data', self.data_configs.speaker, 'train.record')
        assert os.path.exists(record_path), "No train.record found in " + record_path
        dataset = layers.Dataset(record_path, self.data_configs, self.model_configs, self.training_configs)
        dataset.load_tfrecord()
        input_tensor, output_tensor = dataset.next_batch()

        # add TDNN layers
        input_tensor = tf.expand_dims(input_tensor, -1)
        tdnn_out = layers.Subsample_TDNN(input_tensor, self.model_configs.num_layers, self.model_configs.layer_info)

        # add sofmax layer
        self.output = Softmax(tdnn_out, output_dim)

        # add loss
        self.loss = tf.losses.CategoricalCrossentropy(output_tensor, self.output)

        # add optimizer
        self.optimizer = tf.train.AdamOptimizer(self.training_configs.learning_rate, self.training_configs.adam_beta1,
                    self.training_configs.adam_beta2, self.training_configs.adam_epsilon)

    def train(self):
        with tf.InteractiveSession() as sess:
            

    def predict(self):
        pass