import pdb
import tensorflow as tf 

class Dataset(object):
    def __init__(self, record_path, data_configs, model_configs, training_configs):
        self.record_path = record_path
        self.data_configs = data_configs
        self.model_configs = model_configs
        self.training_configs = training_configs
        self.feature_dict = {
            'input_feature': tf.FixedLenSequenceFeature([], dtype=tf.string)
        }
        self.context_dict = {
            'output_class' : tf.FixedLenFeature([], dtype=tf.int64),
            'feat_dim0'    : tf.FixedLenFeature([], dtype=tf.int64),
            'feat_dim1'    : tf.FixedLenFeature([], dtype=tf.int64)
        }

    def parse_tfrecord(self, example):
        context, features = tf.parse_single_sequence_example(example, sequence_features=self.feature_dict, context_features=self.context_dict)
        dim0 = context['feat_dim0']
        dim1 = context['feat_dim1']
        output_class = context['output_class']
        input_feature = tf.decode_raw(features['input_feature'], tf.float32)
        input_feature = tf.reshape(input_feature, [dim0, dim1])
        return input_feature, output_class

    def load_tfrecord(self, shuffle = None, repeat = None):
        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.map(self.parse_tfrecord)
        dataset = dataset.batch(self.training_configs.batch_size)
        if shuffle != None:
            dataset = dataset.shuffle(shuffle)
        if repeat != None:
            dataset = dataset.repeat(repeat)
        self.iterator = dataset.make_one_shot_iterator()

    def next_batch(self):
        return self.iterator.get_next()

