import os
import pdb
import math
import random
import numpy as np 
import soundfile as sf
import tensorflow as tf 

from tqdm import tqdm

from infolog import log
from . import layers
from speech_features.base import mfcc, delta

mfcc_mean = [3.78006526e-01, -4.00190337e+01, -3.31905641e+00, -2.28594191e+00,
            -1.68875260e+00, -1.31968740e+01, -6.83837268e+00, -3.45659099e+00,
            -1.03483508e+01, -9.10794440e+00, -4.15393851e+00, -1.24307398e+01,
            -8.86608480e+00, -5.19131484e+00, -4.25165477e-02,  1.16097964e-02,
            7.78610982e-03,  1.81413237e-02,  1.18418127e-03,  1.81433919e-02,
            3.38739085e-02,  1.30856064e-02, -9.01942923e-03,  4.77990246e-03,
            -4.53386717e-03,  6.93330284e-03, -5.08295724e-03, -7.96265489e-03,
            -3.32165606e-03,  5.48567253e-04, -6.72682988e-04,  5.07762323e-03,
            4.81159653e-03, -1.40227998e-03,  5.95164952e-03,  3.04054412e-03,
            3.14334514e-03,  3.75045053e-03,  2.68109706e-03,  1.41634137e-03]

mfcc_std = [1.01826931, 65.7136309,  15.46226975, 17.15669019, 16.71445839, 25.98081295,
            20.63222354, 14.79783584, 21.85685645, 20.26518489, 14.83555884, 23.87387081,
            17.95320907, 13.10238271,  3.07169386,  3.27137234,  3.5204676,   3.35330126,
            3.56747124,  3.80083351,  3.60261127,  3.63155241,  3.68597123,  3.34682958,
            3.39926061,  3.08231115,  2.78381945,  1.08731157,  1.24452376,  1.32273193,
            1.29808822,  1.42086541,  1.5482317,   1.537888,    1.51716503,  1.56346654,
            1.4424433,  1.4575192,   1.34579971,  1.211822]

class Dataset():
    def __init__(self, mode, data_configs, batch_size, max_seqlen):
        self.data_configs = data_configs
        self.data_path = os.path.join('data', self.data_configs.speaker, mode)
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.filenames = [file[:-4] for file in os.listdir(os.path.join(self.data_path, 'dur'))]
        self.max_step = int(math.ceil(len(self.filenames)/batch_size))
        self.dataset_size = len(self.filenames)
        self.mapping = {}
        with open('data/ossian_mapping.txt', 'r') as f:
            for i, row in enumerate(f):
                _, char = row[:-1].split('\t')
                self.mapping[char] = i
        self.init_dataset()

    def init_dataset(self):
        self.current_step = 0
        random.shuffle(self.filenames)

    def get_feature(self, filename):
        # prepare input features and labels
        features = []
        labels   = []
        # compute 40 dimension MFCCs = Energy + mfcc + delta + delta-deltas = 1 + 13 + 13 + 13
        signal, sr = sf.read(os.path.join(self.data_path, 'wav', filename + '.wav'))
        # compute mfcc of each frame in signal
        mfccs, energy = mfcc(signal = signal, appendEnergy = self.data_configs.appendEnergy, samplerate = self.data_configs.sample_rate, \
                             winlen = self.data_configs.winlen, winstep=self.data_configs.winstep)
        energy = np.reshape(energy, (len(energy), 1))
        # compute delta coefficient, from frame t computed in terms of the static coefficients of t-N and t+N frames
        delta_mfccs = delta(mfccs, N = 2)
        # compute delta delta coefficient
        delta_deltas_mfccs = delta(delta_mfccs, N = 2)
        # concate them to 40 dimension MFCCs
        mfcc_feats = np.concatenate((energy, mfccs, delta_mfccs, delta_deltas_mfccs), axis = 1)
        # add padding to list of features
        # features = mfcc_feats
        features = (mfcc_feats - mfcc_mean)/mfcc_std
        features = list(np.concatenate((np.zeros((self.data_configs.left_frames, 40)), features, np.zeros((self.data_configs.right_frames, 40))), axis = 0))
        # read duration info file
        durations = []
        with open(os.path.join(self.data_path, 'dur', filename + '.dur'), 'r') as f:
            for row in f:
                char, dur_s, dur_e = row[:-1].split('\t')
                durations.append((char, int(dur_e)))
        # get list labels
        labels = []
        feat_idx = self.data_configs.left_frames
        dur_idx = 0
        while feat_idx < len(features) - self.data_configs.right_frames:
            if ((feat_idx - self.data_configs.left_frames)*self.data_configs.winstep*1000 < durations[dur_idx][1] - 0.5*self.data_configs.winlen*1000) or (dur_idx == len(durations) - 1):
                label_val = self.mapping[durations[dur_idx][0]]
                one_hot_label = np.zeros(len(self.mapping))
                one_hot_label[label_val] = 1
                labels.append(one_hot_label)
            else:
                dur_idx += 1
                label_val = self.mapping[durations[dur_idx][0]]
                one_hot_label = np.zeros(len(self.mapping))
                one_hot_label[label_val] = 1
                labels.append(one_hot_label)
            feat_idx += 1
        features = features[:min(len(features), self.max_seqlen + self.data_configs.left_frames + self.data_configs.right_frames)]
        labels = labels[:min(len(labels), self.max_seqlen)]
        seqlen = len(labels)
        # add padding
        while len(features) < self.max_seqlen + self.data_configs.left_frames + self.data_configs.right_frames:
            features.append(np.zeros_like(features[0]))
            label = np.zeros_like(labels[0])
            label[0] = 1
            labels.append(label)
        # return
        return features, labels, seqlen

    def get_next(self):
        input_data = []
        seqlen_data = []
        target_data = []
        files = self.filenames[self.current_step*self.batch_size: (self.current_step + 1)*self.batch_size]
        for file in files:
            _input, target, seqlen = self.get_feature(file)
            input_data.append(_input)
            seqlen_data.append(seqlen)
            target_data.append(target)
        self.current_step += 1
        if self.current_step >= self.max_step:
            self.init_dataset()
        # return input_data, target_data, seqlen_data
        return np.expand_dims(input_data, axis=-1), target_data, seqlen_data


class TDNN_LSTM():
    def __init__(self, data_configs, model_configs, training_configs):
        self.data_configs  = data_configs
        self.model_configs = model_configs
        self.training_configs = training_configs

    def _learning_rate_decay(self):
        #Compute natural exponential decay
        lr = tf.train.exponential_decay(self.training_configs.initial_learning_rate, self.global_step - self.training_configs.start_decay, 
                                        self.training_configs.decay_steps, self.training_configs.decay_rate, name='lr_exponential_decay')

        #clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, self.training_configs.final_learning_rate), self.training_configs.initial_learning_rate)

    def initialize(self):
        """
            build graph for TDNN + LSTM model
        """
        # define placeholder
        self.input_tensor  = tf.placeholder(name = 'input', shape = [None, self.model_configs.max_seqlen + 
                                            self.data_configs.left_frames + self.data_configs.right_frames, self.model_configs.input_dim, 1], 
                                            dtype = tf.float32)
        self.seqlen_tensor = tf.placeholder(name = 'sequence_length', shape = [None], dtype = tf.int32)
        self.target_tensor = tf.placeholder(name = 'target', shape = [None, self.model_configs.max_seqlen, 
                                            self.model_configs.output_dim], dtype = tf.float32)
        self.global_step   = tf.Variable(0, name='global_step', trainable=False)
        # add layer to graph
        self.tdnn_lstm_output = layers.TDNN_LSTM(self.input_tensor, self.seqlen_tensor, self.model_configs.num_layers, self.model_configs.layer_info, 
                                                self.model_configs.input_dim, max_seqlen=self.model_configs.max_seqlen, 
                                                max_left_frame=self.data_configs.left_frames, max_right_frame=self.data_configs.right_frames)
        # add softmax
        self.output = layers.Softmax(self.tdnn_lstm_output, self.model_configs.layer_info[-1].lstm_num_units, self.model_configs.output_dim)
        # add loss
        self.loss = tf.keras.losses.categorical_crossentropy(self.target_tensor, self.output)
        # self.loss = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(self.target_tensor * tf.log(self.output), axis = 2), axis = 1), axis = 0)

        # add optimizer
        self.learning_rate = self._learning_rate_decay()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.training_configs.adam_beta1, self.training_configs.adam_beta2, 
                                                self.training_configs.adam_epsilon).minimize(self.loss, global_step = self.global_step)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)   


    def train(self):
        # define session and load last checkpoint
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep = 1000)
        ckpt_path = os.path.join('checkpoint', self.data_configs.speaker)
        if tf.train.latest_checkpoint(ckpt_path) != None:
            log("load last checkpoint from " + tf.train.latest_checkpoint(ckpt_path))
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        # train model
        train_dataset = Dataset('train', self.data_configs, self.training_configs.batch_size, self.model_configs.max_seqlen)
        valid_dataset = Dataset('valid', self.data_configs, self.training_configs.batch_size, self.model_configs.max_seqlen)

        # mean_val = np.zeros(40)
        # num_labels = 0
        # for idx in tqdm(range(train_dataset.max_step)):
        #     input_data, target_data, seqlen_data = train_dataset.get_next()
        #     mean_val += np.sum(np.sum(input_data, axis = 1), axis = 0)
        #     num_labels += np.sum(seqlen_data)
        # mean_val = mean_val/num_labels
        # print(mean_val)

        # std_val = np.zeros(40)
        # num_labels = 0
        # for idx in tqdm(range(train_dataset.max_step)):
        #     input_data, target_data, seqlen_data = train_dataset.get_next()
        #     std_val += np.sum(np.sum(np.abs(np.subtract(input_data, mfcc_mean))**2, axis = 1), axis = 0)
        #     num_labels += np.sum(seqlen_data)
        # std_val = np.sqrt(std_val/num_labels)
        # print(std_val)
        # pdb.set_trace()

        # input_data, target_data, seqlen_data = train_dataset.get_next()
        # temp = sess.run(self.tdnn_lstm_output, feed_dict={self.input_tensor: input_data, self.seqlen_tensor: seqlen_data, self.target_tensor: target_data})
        # pdb.set_trace()
        for epoch in range(self.training_configs.train_epochs):
            # for idx in tqdm(range(train_dataset.max_step)):
            #     input_data, target_data, seqlen_data = train_dataset.get_next()
            #     loss, _, global_step = sess.run([self.loss, self.optimizer, self.global_step], feed_dict={self.input_tensor: input_data, self.seqlen_tensor: seqlen_data, 
            #                                                                                  self.target_tensor: target_data})
            #     if idx % 10 == 0:
            #         print("\rEpoch: ", epoch, " Step ", global_step, " Loss: ", np.average(loss))
            # # save model
            # saver.save(sess, os.path.join(ckpt_path, 'model.ckpt'), global_step=global_step)
            # run evaluate on valid set
            true_pred = 0
            num_pred  = 0
            for idx in tqdm(range(train_dataset.max_step)):
                input_data, target_data, seqlen_data = train_dataset.get_next()
                logit = sess.run(self.output, feed_dict={self.input_tensor: input_data, self.seqlen_tensor: seqlen_data, 
                                                                   self.target_tensor: target_data})
                pdb.set_trace()
                for i in range(len(seqlen_data)):
                    true_pred += np.sum(np.equal(np.argmax(logit[i][:seqlen_data[i]], axis=-1), np.argmax(target_data[i][:seqlen_data[i]], axis=-1)))
                    num_pred  += seqlen_data[i]
            log("Frame accuracy on valid set: " + str(true_pred/num_pred))



