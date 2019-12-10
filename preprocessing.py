"""
    TODO:
    - how to handle audio has length < num_frames*winstep+winlen
"""
import os
import sys
import pdb
import logging
import argparse
import numpy as np 
import soundfile as sf
import tensorflow as tf 

from tqdm import tqdm

from configs.base import data_configs
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

def serialize_example(feat, label):
    input_features = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[val.tostring()])) for val in feat]
    output_features = [tf.train.Feature(float_list=tf.train.FloatList(value=[val])) for val in label]
    feature_list = {
        'input_feature': tf.train.FeatureList(feature=input_features),
        'output_class' : tf.train.FeatureList(feature=output_features),
    }
    context_dict = {
        'feat_dim0' : tf.train.Feature(int64_list=tf.train.Int64List(value=[np.shape(feat)[0]])),
        'feat_dim1' : tf.train.Feature(int64_list=tf.train.Int64List(value=[np.shape(feat)[1]])),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_list), context = tf.train.Features(feature=context_dict))
    return example_proto.SerializeToString()

def main():
    # make training data dir
    os.makedirs(os.path.join('training_data', data_configs.speaker), exist_ok = True)
    # initialize logging file
    logging.basicConfig(filename=os.path.join('training_data', data_configs.speaker, 'preprocessing.log'), format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # get mapping character to index
    mapping = {}
    with open('data/ossian_mapping.txt', 'r') as f:
        for i, row in enumerate(f):
            _, char = row[:-1].split('\t')
            mapping[char] = i

    # generate Train TF Record
    logging.info('generate training TFRecord')
    train_path = os.path.join('data', data_configs.speaker, 'train')
    # get all filenames
    filenames = ['.'.join(file.split('.')[:-1]) for file in os.listdir(os.path.join(train_path, 'time_lab'))]
    # write training record
    j = 0
    with tf.io.TFRecordWriter(os.path.join('training_data', data_configs.speaker, 'train-temp.record')) as writer:
        for filename in tqdm(filenames):
            # prepare input features and labels
            features = []
            labels   = []
            # compute 40 dimension MFCCs = Energy + mfcc + delta + delta-deltas = 1 + 13 + 13 + 13
            signal, sr = sf.read(os.path.join(train_path, 'wav', filename + '.wav'))
            # compute mfcc of each frame in signal
            mfccs, energy = mfcc(signal = signal, appendEnergy = data_configs.appendEnergy, samplerate = data_configs.sample_rate, \
                                winlen = data_configs.winlen, winstep=data_configs.winstep)
            energy = np.reshape(energy, (len(energy), 1))
            # compute delta coefficient, from frame t computed in terms of the static coefficients of t-N and t+N frames
            delta_mfccs = delta(mfccs, N = 2)
            # compute delta delta coefficient
            delta_deltas_mfccs = delta(delta_mfccs, N = 2)
            # concate them to 40 dimension MFCCs
            mfcc_feats = np.concatenate((energy, mfccs, delta_mfccs, delta_deltas_mfccs), axis = 1)
            # add padding to list of features
            features = mfcc_feats
            # features = (mfcc_feats - mfcc_mean)/mfcc_std
            features = np.concatenate((np.zeros((data_configs.left_frames, 40)), features, np.zeros((data_configs.right_frames, 40))), axis = 0)
            # read duration info file
            durations = []
            with open(os.path.join(train_path, 'dur', filename + '.dur'), 'r') as f:
                for row in f:
                    char, dur_s, dur_e = row[:-1].split('\t')
                    durations.append((char, int(dur_e)))
            # get list labels
            seq_features = []
            labels = []
            feat_idx = data_configs.left_frames
            dur_idx = 0
            while feat_idx < len(features) - data_configs.right_frames:
                if ((feat_idx - data_configs.left_frames)*data_configs.winstep*1000 < durations[dur_idx][1] - 0.5*data_configs.winlen*1000) or (dur_idx == len(durations) - 1):
                    labels.append(mapping[durations[dur_idx][0]])
                    seq_features.append(features[feat_idx - data_configs.left_frames: feat_idx + data_configs.right_frames + 1])
                else:
                    dur_idx += 1
                    seq_features.append(features[feat_idx - data_configs.left_frames: feat_idx + data_configs.right_frames + 1])
                    labels.append(mapping[durations[dur_idx][0]])
                feat_idx += 1    
            # write to TFRecord file
            for feat, label in zip(seq_features, labels):
                one_hot_label = np.zeros(len(mapping))
                one_hot_label[label] = 1
                example = serialize_example(feat, one_hot_label)
                writer.write(example)

    # generate Valid TF Record
    logging.info('generate valid TFRecord')
    valid_path = os.path.join('data', data_configs.speaker, 'valid')
    # get all filenames
    filenames = ['.'.join(file.split('.')[:-1]) for file in os.listdir(os.path.join(valid_path, 'time_lab'))]
    # write valid record
    j = 0
    with tf.io.TFRecordWriter(os.path.join('training_data', data_configs.speaker, 'valid-temp.record')) as writer:
        for filename in tqdm(filenames):
            # prepare input features and labels
            features = []
            labels   = []
            # compute 40 dimension MFCCs = Energy + mfcc + delta + delta-deltas = 1 + 13 + 13 + 13
            signal, sr = sf.read(os.path.join(valid_path, 'wav', filename + '.wav'))
            # compute mfcc of each frame in signal
            mfccs, energy = mfcc(signal = signal, appendEnergy = data_configs.appendEnergy, samplerate = data_configs.sample_rate, \
                                winlen = data_configs.winlen, winstep=data_configs.winstep)
            energy = np.reshape(energy, (len(energy), 1))
            # compute delta coefficient, from frame t computed in terms of the static coefficients of t-N and t+N frames
            delta_mfccs = delta(mfccs, N = 2)
            # compute delta delta coefficient
            delta_deltas_mfccs = delta(delta_mfccs, N = 2)
            # concate them to 40 dimension MFCCs
            mfcc_feats = np.concatenate((energy, mfccs, delta_mfccs, delta_deltas_mfccs), axis = 1)
            # add padding to list of features
            features = mfcc_feats
            # features = (mfcc_feats - mfcc_mean)/mfcc_std
            features = np.concatenate((np.zeros((data_configs.left_frames, 40)), features, np.zeros((data_configs.right_frames, 40))), axis = 0)
            # read duration info file
            durations = []
            with open(os.path.join(valid_path, 'dur', filename + '.dur'), 'r') as f:
                for row in f:
                    char, dur_s, dur_e = row[:-1].split('\t')
                    durations.append((char, int(dur_e)))
            # get list labels
            seq_features = []
            labels = []
            feat_idx = data_configs.left_frames
            dur_idx = 0
            while feat_idx < len(features) - data_configs.right_frames:
                if ((feat_idx - data_configs.left_frames)*data_configs.winstep*1000 < durations[dur_idx][1] - 0.5*data_configs.winlen*1000) or (dur_idx == len(durations) - 1):
                    labels.append(mapping[durations[dur_idx][0]])
                    seq_features.append(features[feat_idx - data_configs.left_frames: feat_idx + data_configs.right_frames + 1])
                else:
                    dur_idx += 1
                    seq_features.append(features[feat_idx - data_configs.left_frames: feat_idx + data_configs.right_frames + 1])
                    labels.append(mapping[durations[dur_idx][0]])
                feat_idx += 1    
            # write to TFRecord file
            for feat, label in zip(seq_features, labels):
                one_hot_label = np.zeros(len(mapping))
                one_hot_label[label] = 1
                example = serialize_example(feat, one_hot_label)
                writer.write(example)

if __name__ == '__main__':
    main()