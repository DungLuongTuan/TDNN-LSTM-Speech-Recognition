"""
    define model and train
"""
import tensorflow as tf 
# tf.enable_eager_execution()

from models import TDNN, TDNN_LSTM
from configs.base import data_configs, model_configs, training_configs

def main():
    if model_configs.model_name == 'TDNN':
        model = TDNN(data_configs, model_configs, training_configs)
        model.initialize()
        model.train()

if __name__ == '__main__':
    main()