# TODO
# make training_data folder contain data-label pairs for training
# 

import os
import sys
import pdb
import logging
import argparse
import numpy as np 
import soundfile as sf
import tensorflow as tf 

from configs.base import data_configs
from speech_features.base import mfcc, delta

def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(mfccs, label):
    feature = {
        'audio/mfcc': _bytes_feature(image_encoded),
        'audio/label': _bytes_feature(image_format)
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def main():
    # make training data dir
    os.makedirs(os.path.join('training_data', data_configs.speaker), exist_ok = True)
    # initialize logging file
    logging.basicConfig(filename='preprocessing.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # generate TF Record
    logging.info('generate training TFRecord')
    train_path = os.path.join('data', data_configs.speaker, 'train')
    filenames = ['.'.join(file.split('.')[:-1]) for file in os.listdir(os.path.join(train_path, 'time_lab'))]
    with tf.python_io.TFRecordWriter(os.path.join('training_data', data_configs.speaker, 'train.record')) as writer:
        for filename in filenames:
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
            # get list labels
            
            # write to TFRecord file
            pdb.set_trace()
            example = serialize_example(mfcc, label)
            writer.write(example)


if __name__ == '__main__':
    main()