import pdb
import tensorflow as tf 

class Dataset(object):
    def __init__(self, record_path, data_configs, model_configs, training_configs):
        self.record_path = record_path
        self.data_configs = data_configs
        self.model_configs = model_configs
        self.training_configs = training_configs
        self.feature_dict = {
            'input_feature': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'output_class' : tf.FixedLenSequenceFeature([], dtype=tf.float32)
        }
        self.context_dict = {
            'feat_dim0'    : tf.FixedLenFeature([], dtype=tf.int64),
            'feat_dim1'    : tf.FixedLenFeature([], dtype=tf.int64)
        }

    def parse_tfrecord(self, example):
        context, features = tf.parse_single_sequence_example(example, sequence_features=self.feature_dict, context_features=self.context_dict)
        dim0 = context['feat_dim0']
        dim1 = context['feat_dim1']
        output_class = features['output_class']
        input_feature = tf.decode_raw(features['input_feature'], tf.float64)
        input_feature = tf.reshape(input_feature, [dim0, dim1])
        input_feature = tf.cast(input_feature, tf.float32)
        return input_feature, output_class

    def load_tfrecord(self, repeat = None):
        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.map(self.parse_tfrecord)
        dataset = dataset.batch(self.training_configs.batch_size)
        dataset = dataset.repeat(count=repeat)
        self.iterator = dataset.make_one_shot_iterator()

    def next_batch(self):
        return self.iterator.get_next()

