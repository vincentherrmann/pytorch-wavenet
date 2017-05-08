import time
import torch
from WaveNetModel2 import WaveNetModel2
from model import Optimizer, AudioFileLoader

data_loader = AudioFileLoader(['../train_samples/violin.wav', '../train_samples/piano.wav', '../train_samples/bell.wav'],
							  classes=256,
							  max_load_duration=0.9,
							  dtype=torch.FloatTensor,
							  ltype=torch.LongTensor,
							  sampling_rate=11025)

for i in range(100):
	data_loader.get_wavenet_minibatch([1+i, 5+i, 9+i, 12+i],
									  receptive_field=1000,
									  target_length=256)
	if i % 10 == 9:
		data_loader.use_new_chunk()

