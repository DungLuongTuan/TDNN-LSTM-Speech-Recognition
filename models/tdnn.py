from layers.dataset import Dataset

class TDNN():
	def __init__(self, data_configs, model_configs):
		self.data_configs  = data_configs
		self.model_configs = model_configs

	def initialize(self):
		'''
			build graph of TDNN models
		'''
		# get input tensor
		record_path = os.path.join('training_data', self.data_configs.speaker, 'train.record')
		assert os.path.exists(record_path), "No train.record found in " + record_path
		dataset = Dataset(record_path, data_configs, model_configs)
		input_tensor, output_tensor = dataset.next_batch()

		# add TDNN layers
