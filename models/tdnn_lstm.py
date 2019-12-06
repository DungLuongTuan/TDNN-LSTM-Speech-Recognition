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
        features = mfcc_feats
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
        seqlen = len(features)  
        # add padding
        while len(features) < self.max_seqlen + self.data_configs.left_frames + self.data_configs.right_frames:
            features.append(np.zeros_like(features[0]))
            labels.append(np.zeros_like(labels[0]))
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
        return np.expand_dims(input_data, axis=-1), target_data, seqlen_data


class TDNN_LSTM():
    def __init__(self, data_configs, model_configs, training_configs):
        self.data_configs  = data_configs
        self.model_configs = model_configs
        self.training_configs = training_configs

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
        # add layer to graph
        self.tdnn_lstm_output = layers.TDNN_LSTM(self.input_tensor, self.model_configs.num_layers, self.model_configs.layer_info, 
                                                self.model_configs.input_dim, max_seqlen=self.model_configs.max_seqlen, 
                                                max_left_frame=self.data_configs.left_frames, max_right_frame=self.data_configs.right_frames)
        # add softmax
        if self.model_configs.layer_info[-1].layer_name == 'TDNN':
            self.output = layers.Softmax(self.tdnn_lstm_output, self.model_configs.layer_info[-1].num_filters, self.model_configs.output_dim)
        elif self.model_configs.layer_info[-1].layer_name == 'LSTM':
            self.output = layers.Softmax(self.tdnn_lstm_output, self.model_configs.layer_info[-1].num_units, self.model_configs.output_dim)
        # add loss
        self.loss = tf.keras.losses.categorical_crossentropy(self.target_tensor, self.output)

        # add optimizer
        self.optimizer = tf.train.AdamOptimizer(self.training_configs.learning_rate, self.training_configs.adam_beta1,
                                                self.training_configs.adam_beta2, self.training_configs.adam_epsilon).minimize(self.loss)      


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
        input_data, target_data, seqlen_data = train_dataset.get_next()
        temp = sess.run(self.tdnn_lstm_output, feed_dict={self.input_tensor: input_data, self.seqlen_tensor: seqlen_data, self.target_tensor: target_data})
        pdb.set_trace()

        for epoch in training_configs.train_epochs:
            for idx in tqdm(range(train_dataset.max_step)):
                input_data, seqlen_data, target_data = train_dataset.get_next()
                loss, _ = sess.run([self.loss, self.optimizer], feed_dict={self.input_tensor: input_data, self.seqlen_tensor: seqlen_data, 
                                                                           self.target_tensor: target_data})
                if idx % 10 == 0:
                    print("\rEpoch: ", epoch, " Step ", idx, " Loss: ", np.average(loss))

            # run evaluate on valid set
            true_pred = 0
            for idx in tqdm(range(valid_dataset.max_step)):
                input_data, seqlen_data, target_data = train_dataset.get_next()
                logit = sess.run(self.output, feed_dict={self.input_tensor: input_data, self.seqlen_tensor: seqlen_data, 
                                                                   self.target_tensor: target_data})
                true_pred += np.sum(np.equal(np.argmax(logit, axis=-1), np.argmax(target, axis=-1)))
            log("Frame accuracy on valid set: " + str(true_pred/valid_dataset.dataset_size))



