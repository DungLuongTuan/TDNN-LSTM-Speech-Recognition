"""
    define model and train
"""
import os
import tensorflow as tf 

import infolog

from models import SubsampleTDNN, TDNN_LSTM
from configs.base import data_configs, model_configs, training_configs

log = infolog.log

def main():
    # init log
    log_dir = os.path.join('checkpoint', data_configs.speaker)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), model_configs.model_name, None)
    # train model
    if model_configs.model_name == 'TDNN':
        model = SubsampleTDNN(data_configs, model_configs, training_configs)
        model.initialize()
        model.train()

if __name__ == '__main__':
    main()