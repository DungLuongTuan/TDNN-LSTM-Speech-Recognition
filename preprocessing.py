# TODO
# make training_data folder contain data-label pairs for training
# 

import os
import sys
import logging
import argparse
import numpy as np 
import tensorflow as tf 

from configs.base import data_configs

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
    os.makedirs(os.path.join('training_data', data_configs.speaker))
    # initialize logging file
    logging.basicConfig(filename='preprocessing.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # generate TF Record
    logging.generate("generate training TFRecord")
    filenames = [file.split('.')[0] for file in os.listdir(os.path.join("data", data_configs.speaker, "train", time_lab))]
    with tf.python_io.TFRecordWriter(os.path.join("training", data_configs.speaker, "train.record"))
    for filename in filenames:
        # prepare mfccs and labels
        # TODO

        # write to TFRecord file
        example = serialize_example(mfcc, label)
        writer.write(example)


if __name__ == '__main__':
    main()