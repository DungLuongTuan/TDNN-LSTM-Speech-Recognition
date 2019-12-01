"""
	define model and train
"""
from models import TDNN, TDNN_LSTM
from configs.base import data_configs, model_configs, training_configs

def main():
	if model_configs.model_name == 'TDNN':
		model = TDNN(data_configs)
		model.initialize()

if __name__ == '__main__':
	main()