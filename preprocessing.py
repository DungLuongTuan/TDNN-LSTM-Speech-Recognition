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


def serialize_example(feat, label):
    features = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[val.tostring()])) for val in feat]
    feature_list = {
        'input_feature': tf.train.FeatureList(feature=features)
    }
    context_dict = {
        'output_class' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
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
    with tf.io.TFRecordWriter(os.path.join('training_data', data_configs.speaker, 'train.record')) as writer:
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
                example = serialize_example(feat, label)
                writer.write(example)

    # generate Valid TF Record
    logging.info('generate valid TFRecord')
    valid_path = os.path.join('data', data_configs.speaker, 'valid')
    # get all filenames
    filenames = ['.'.join(file.split('.')[:-1]) for file in os.listdir(os.path.join(valid_path, 'time_lab'))]
    # write valid record
    j = 0
    with tf.io.TFRecordWriter(os.path.join('training_data', data_configs.speaker, 'valid.record')) as writer:
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
                example = serialize_example(feat, label)
                writer.write(example)

if __name__ == '__main__':
    main()