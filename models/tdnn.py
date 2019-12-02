import os
import pdb
import tensorflow as tf 

from . import layers

class SubsampleTDNN():
    def __init__(self, data_configs, model_configs, training_configs):
        self.data_configs  = data_configs
        self.model_configs = model_configs
        self.training_configs = training_configs

    def initialize(self):
        '''
            build graph of TDNN models
        '''
        # get input tensor
        train_record_path = os.path.join('training_data', self.data_configs.speaker, 'train.record')
        valid_record_path = os.path.join('training_data', self.data_configs.speaker, 'valid.record')
        assert os.path.exists(train_record_path), "No train.record found in " + train_record_path
        assert os.path.exists(valid_record_path), "No valid.record found in " + valid_record_path
        self.train_dataset = layers.Dataset(train_record_path, self.data_configs, self.model_configs, self.training_configs)
        self.valid_dataset = layers.Dataset(valid_record_path, self.data_configs, self.model_configs, self.training_configs)
        self.train_dataset.load_tfrecord(shuffle = True)
        self.valid_dataset.load_tfrecord(repeat = 1)
        input_tensor, output_tensor = self.train_dataset.next_batch()

        # add TDNN layers
        input_tensor = tf.expand_dims(input_tensor, -1)
        tdnn_out = layers.SubsampleTDNN(input_tensor, self.model_configs.num_layers, self.model_configs.layer_info, self.model_configs.input_dim)

        # add sofmax layer
        self.output = layers.Softmax(tdnn_out, self.model_configs.layer_info[-1].num_filters, self.model_configs.output_dim)

        # add loss
        self.loss = tf.keras.losses.categorical_crossentropy(output_tensor, self.output)

        # add optimizer
        self.optimizer = tf.train.AdamOptimizer(self.training_configs.learning_rate, self.training_configs.adam_beta1,
                                                self.training_configs.adam_beta2, self.training_configs.adam_epsilon).minimize(self.loss)


    def train(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        temp = sess.run(self.train_dataset.next_batch())
        pdb.set_trace()
        # for idx in range(self.training_configs.training_step):
        #     loss, _ = sess.run([self.loss, self.optimizer])
        #     if idx % self.training_configs.save_step == 0:
        #         while 1:
        #             print(self.valid_dataset.next_batch())
            

    def predict(self):
        pass