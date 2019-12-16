import os
import subprocess

from tqdm import tqdm

# train
filenames = [file[:-4] for file in os.listdir('data/ngoc_anh_vov/train/txt')]

for filename in tqdm(filenames):
	subprocess.call(['cp data/ngoc_anh_vov/dur/' + filename + '.dur data/ngoc_anh_vov/train/dur'], shell = True)

# valid
filenames = [file[:-4] for file in os.listdir('data/ngoc_anh_vov/valid/txt')]

for filename in tqdm(filenames):
	subprocess.call(['cp data/ngoc_anh_vov/dur/' + filename + '.dur data/ngoc_anh_vov/valid/dur'], shell = True)