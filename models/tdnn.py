import os
import pdb
import math
import numpy as np 
import tensorflow as tf 

from tqdm import tqdm

from infolog import log
from . import layers

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
        train_record_path = os.path.join('training_data', self.data_configs.speaker, 'train.record')
        cnt = 0
        for record in tf.python_io.tf_record_iterator(train_record_path):
            cnt += 1
        self.train_record_size = cnt
        log("Number of train samples:\t" + str(self.train_record_size))
        cnt = 0
        valid_record_path = os.path.join('training_data', self.data_configs.speaker, 'valid.record')
        for record in tf.python_io.tf_record_iterator(valid_record_path):
            cnt += 1
        self.valid_record_size = cnt
        log("Number of valid samples:\t" + str(self.valid_record_size))
        assert os.path.exists(train_record_path), "No train.record found in " + train_record_path
        assert os.path.exists(valid_record_path), "No valid.record found in " + valid_record_path
        self.train_dataset = layers.Dataset(train_record_path, self.data_configs, self.model_configs, self.training_configs)
        self.valid_dataset = layers.Dataset(valid_record_path, self.data_configs, self.model_configs, self.training_configs)
        self.train_dataset.load_tfrecord()
        self.valid_dataset.load_tfrecord()
        input_tensor, self.output_tensor = self.train_dataset.next_batch()

        # add TDNN layers
        input_tensor = tf.expand_dims(input_tensor, -1)
        if self.model_configs.subsample:
            self.tdnn_out = layers.SubsampleTDNN(input_tensor, self.model_configs.num_layers, self.model_configs.layer_info, 
                                                self.model_configs.input_dim, max_left_frame=self.data_configs.left_frames, 
                                                max_right_frame=self.data_configs.right_frames)
        else:
            self.tdnn_out = layers.TDNN(input_tensor, self.model_configs.num_layers, self.model_configs.layer_info, 
                                        self.model_configs.input_dim, subsample=self.model_configs.subsample)

        # add sofmax layer
        self.output = layers.Softmax(self.tdnn_out, self.model_configs.layer_info[-1].num_filters, self.model_configs.output_dim)

        # add loss
        self.loss = tf.keras.losses.categorical_crossentropy(self.output_tensor, self.output)

        # add optimizer
        self.optimizer = tf.train.AdamOptimizer(self.training_configs.learning_rate, self.training_configs.adam_beta1,
                                                self.training_configs.adam_beta2, self.training_configs.adam_epsilon).minimize(self.loss)

    def train(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep = 1000)
        ckpt_path = os.path.join('checkpoint', self.data_configs.speaker)
        if tf.train.latest_checkpoint(ckpt_path) != None:
            log("load last checkpoint from " + tf.train.latest_checkpoint(ckpt_path))
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        for epoch in range(self.training_configs.train_epochs):
            log("epoch: " + str(epoch))
            try:
                num_iters = math.ceil(self.train_record_size/self.training_configs.batch_size)
                for idx in tqdm(range(num_iters)):
                    loss, _ = sess.run([self.loss, self.optimizer])
                    if idx % 10 == 0:
                        print("\rEpoch: ", epoch, " Step ", idx, " Loss: ", np.average(loss))
            except tf.errors.OutOfRangeError:
                print('tf.errors.OutOfRangeError')
            # save checkpoint
            saver.save(sess, os.path.join(ckpt_path, 'model-' + str(epoch) + '.ckpt'))
            # evaluate on valid set
            log("Calculate accuracy on valid set")
            num_iters = math.ceil(self.valid_record_size/self.training_configs.batch_size)
            true_pred = 0
            for j in tqdm(range(num_iters)):
                input_tensor, target_tensor = self.valid_dataset.next_batch()
                _input, target = sess.run([input_tensor, target_tensor])
                input_tensor = tf.get_default_graph().get_tensor_by_name("IteratorGetNext:0")
                logit = sess.run(self.output, feed_dict = {input_tensor: _input})
                true_pred += np.sum(np.equal(np.argmax(logit, axis=-1), np.argmax(target, axis=-1)))
            log("Frame accuracy on valid set: " + str(true_pred/(num_iters*self.training_configs.batch_size)))
            

    def predict(self):
        pass